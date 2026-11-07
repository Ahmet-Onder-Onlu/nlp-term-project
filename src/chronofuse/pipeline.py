"""End-to-end ChronoFuse pipeline.

Offline (``build``), per domain:
    corpus → chunk → BM25 index ─┐
                  → dense index  ├─ data/indexes/<domain>/
    corpus → metadata profiles ──┤
           → temporal KG ────────┤
           → SQLite doc store ───┘

Online (``run``), per query (1a) or step (1b):
    query analysis (rules | guidance-distilled LLM)
      → channels: BM25(query), BM25(keywords), BM25(sub-queries),
                  dense(query), dense(HyDE), KG-PPR(entities, years)
      → weighted RRF → temporal prior → cross-encoder → LLM listwise → TREC run
"""

from __future__ import annotations

import logging
from pathlib import Path

from tqdm import tqdm

from .config import Config, resolve_path
from .data.reteco import Query, load_corpus, load_guidance, load_queries, load_steps
from .indexing.base import Ranking
from .indexing.bm25_index import BM25Index
from .metadata.chunking import chunk_document
from .metadata.profile import build_profile, read_profiles, write_profiles
from .query.analyzer import LLMQueryAnalyzer, QueryAnalysis, RuleQueryAnalyzer
from .retrieval.fusion import rrf
from .retrieval.temporal_scorer import apply_temporal_prior
from .store import DocStore

log = logging.getLogger("chronofuse")


def index_dir(cfg: Config, domain: str) -> Path:
    return resolve_path(cfg.get("paths.indexes", "data/indexes")) / domain


# =============================================================== build =====
def build_domain(cfg: Config, domain: str, force: bool = False) -> None:
    root = resolve_path(cfg.get("paths.data", "data/raw/reteco"))
    out = index_dir(cfg, domain)
    out.mkdir(parents=True, exist_ok=True)
    ch = cfg.get("chunking", {})
    pids, texts, profiles = [], [], []
    store = DocStore(out / "docs.sqlite")
    batch: list[tuple[str, str]] = []

    need_bm25 = force or not (out / "bm25").exists()
    need_prof = force or not (out / "profiles.jsonl").exists()
    for doc in tqdm(load_corpus(root, domain), desc=f"[{domain}] read+chunk"):
        if need_bm25 or cfg.get("dense.enabled", False):
            for p in chunk_document(
                doc.id,
                doc.content,
                ch.get("size", 200),
                ch.get("overlap", 40),
                ch.get("max_passages", 64),
            ):
                pids.append(p.pid)
                texts.append(p.text)
        if need_prof:
            profiles.append(build_profile(doc))
            batch.append((doc.id, doc.content))
            if len(batch) >= 5000:
                store.add_many(batch)
                batch = []
    if batch:
        store.add_many(batch)
    log.info("[%s] %d passages", domain, len(pids))

    if need_bm25:
        bm = cfg.get("bm25", {})
        BM25Index(k1=bm.get("k1", 0.9), b=bm.get("b", 0.4)).build(pids, texts).save(out / "bm25")
    if need_prof:
        write_profiles(profiles, out / "profiles.jsonl")
    if cfg.get("kg.enabled", False) and (force or not (out / "kg.pkl").exists()):
        from .kg.temporal_kg import TemporalKG

        prof = profiles or list(read_profiles(out / "profiles.jsonl").values())
        TemporalKG(max_entity_df=cfg.get("kg.max_entity_df", 0.05)).build(prof).save(out / "kg.pkl")
    if cfg.get("dense.enabled", False) and (force or not (out / "dense").exists()):
        from .indexing.dense_index import DenseIndex

        d = cfg.get("dense")
        DenseIndex(
            model_name=d.get("model"),
            query_prefix=d.get("query_prefix", ""),
            doc_prefix=d.get("doc_prefix", ""),
            batch_size=d.get("batch_size", 64),
            max_seq_length=d.get("max_seq_length", 256),
        ).build(pids, texts).save(out / "dense")


# ================================================================= run =====
class DomainRetriever:
    """Loads one domain's resources and executes the online pipeline."""

    def __init__(self, cfg: Config, domain: str, llm=None):
        self.cfg, self.domain = cfg, domain
        d = index_dir(cfg, domain)
        self.bm25 = BM25Index.load(d / "bm25")
        self.profiles = read_profiles(d / "profiles.jsonl")
        self.store = DocStore(d / "docs.sqlite")
        self.dense = self.kg = self.ce = self.llm_rr = None
        if cfg.get("dense.enabled", False):
            from .indexing.dense_index import DenseIndex

            dc = cfg.get("dense")
            self.dense = DenseIndex.load(
                d / "dense",
                query_prefix=dc.get("query_prefix", ""),
                batch_size=dc.get("batch_size", 64),
            )
        if cfg.get("kg.enabled", False):
            from .kg.temporal_kg import TemporalKG

            self.kg = TemporalKG.load(d / "kg.pkl")
        if cfg.get("rerank.cross_encoder.enabled", False):
            from .rerank.cross_encoder import CrossEncoderReranker

            self.ce = CrossEncoderReranker(cfg.get("rerank.cross_encoder.model"))
        if cfg.get("rerank.llm.enabled", False) and llm is not None:
            from .rerank.llm_listwise import LLMListwiseReranker

            self.llm_rr = LLMListwiseReranker(
                llm, window=cfg.get("rerank.llm.window", 20), step=cfg.get("rerank.llm.step", 10)
            )

    def retrieve(self, text: str, qa: QueryAnalysis, k: int) -> Ranking:
        cfg, ch = self.cfg, self.cfg.get("channels", {})
        runs: dict[str, Ranking] = {}
        if ch.get("bm25_query", 1.0):
            runs["bm25_query"] = self.bm25.search([text], k)[0]
        if ch.get("bm25_keywords", 0) and qa.keyword_query:
            runs["bm25_keywords"] = self.bm25.search([qa.keyword_query], k)[0]
        if ch.get("bm25_subqueries", 0) and qa.sub_queries:
            sub = self.bm25.search(qa.sub_queries, k)
            runs["bm25_subqueries"] = rrf({str(i): r for i, r in enumerate(sub)}, top=k)
        if self.dense is not None:
            if ch.get("dense_query", 0):
                runs["dense_query"] = self.dense.search([text], k)[0]
            if ch.get("dense_hyde", 0) and qa.hyde_passage:
                runs["dense_hyde"] = self.dense.search([qa.hyde_passage], k)[0]
        if self.kg is not None and ch.get("kg", 0):
            runs["kg"] = self.kg.search(qa.entities, qa.years, k)

        fused = (
            next(iter(runs.values()))
            if len(runs) == 1
            else rrf(runs, weights=dict(ch), k=cfg.get("fusion.rrf_k", 60), top=k)
        )
        fused = apply_temporal_prior(
            fused,
            qa,
            self.profiles,
            lam=cfg.get("temporal.lambda", 0.0),
            tolerance=cfg.get("temporal.tolerance", 1),
        )
        if self.ce is not None or self.llm_rr is not None:
            top_n = max(cfg.get("rerank.cross_encoder.top_n", 100), cfg.get("rerank.llm.top_n", 40))
            texts = self.store.get_many([d for d, _ in fused[:top_n]])
            if self.ce is not None:
                fused = self.ce.rerank(
                    text,
                    fused,
                    texts,
                    self.profiles,
                    top_n=cfg.get("rerank.cross_encoder.top_n", 100),
                )
            if self.llm_rr is not None:
                fused = self.llm_rr.rerank(
                    text, qa, fused, texts, self.profiles, top_n=cfg.get("rerank.llm.top_n", 40)
                )
        return fused


def make_analyzer(cfg: Config, domain: str, llm=None):
    if cfg.get("query.analyzer", "rule") == "llm" and llm is not None:
        root = resolve_path(cfg.get("paths.data", "data/raw/reteco"))
        guidance = load_guidance(root, domain, "train")
        demo_q = [q for q in load_queries(root, domain, "train") if q.id in guidance]
        n = cfg.get("query.num_demos", 3)
        demos = [(q, guidance[q.id]) for q in demo_q[:n]]
        return LLMQueryAnalyzer(llm, demos=demos)
    return RuleQueryAnalyzer()


def run_domain(cfg: Config, domain: str, split: str, subtrack: str, llm=None) -> dict[str, Ranking]:
    root = resolve_path(cfg.get("paths.data", "data/raw/reteco"))
    k = cfg.get("retrieval.depth", 100)
    dr = DomainRetriever(cfg, domain, llm)
    analyzer = make_analyzer(cfg, domain, llm)
    run: dict[str, Ranking] = {}
    if subtrack == "1a":
        for q in tqdm(load_queries(root, domain, split), desc=f"[{domain}/{split}/1a]"):
            run[q.id] = dr.retrieve(q.text, analyzer.analyze(q), k)
    elif subtrack == "1b":
        cache: dict[str, QueryAnalysis] = {}
        for st in tqdm(load_steps(root, domain, split), desc=f"[{domain}/{split}/1b]"):
            if st.query_id not in cache:
                cache[st.query_id] = analyzer.analyze(st.query)
            qa = cache[st.query_id]
            step_q = Query(
                id=st.id,
                text=f"{st.query.title} {st.instruction}",
                title=st.query.title,
                raw=st.query.raw,
            )
            text = step_q.text if cfg.get("steps.include_title", True) else st.instruction
            run[st.id] = dr.retrieve(text, qa, k)
    else:
        raise ValueError(f"unknown subtrack {subtrack!r}")
    return run

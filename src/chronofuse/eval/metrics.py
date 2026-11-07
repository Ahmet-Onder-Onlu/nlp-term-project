"""Official RETECO scoring.

* nDCG@10 via ``pytrec_eval`` (``ndcg_cut_10``), binary relevance.
* Topics present in qrels but absent from the run score **0** (official rule;
  note the starter-kit ``scorer.py`` silently skips them).
* Macro average: mean within each domain, then mean over domains.
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pytrec_eval

Run = dict[str, list[tuple[str, float]]]  # qid -> ranking


def to_trec_dict(run: Run) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for qid, ranking in run.items():
        seen: dict[str, float] = {}
        n = len(ranking)
        for r, (d, _) in enumerate(ranking):
            if d not in seen:  # duplicate ids: first occurrence wins
                seen[d] = float(n - r)  # rank-derived score avoids tie-order ambiguity
        out[qid] = seen
    return out


def per_query_ndcg(run: Run, qrels: dict[str, dict[str, int]], k: int = 10) -> dict[str, float]:
    ev = pytrec_eval.RelevanceEvaluator(qrels, {f"ndcg_cut.{k}"})
    scored = ev.evaluate(to_trec_dict(run))
    return {qid: scored.get(qid, {}).get(f"ndcg_cut_{k}", 0.0) for qid in qrels}


def recall_at(run: Run, qrels: dict[str, dict[str, int]], k: int = 100) -> float:
    vals = []
    for qid, rel in qrels.items():
        gold = {d for d, r in rel.items() if r > 0}
        top = {d for d, _ in run.get(qid, [])[:k]}
        vals.append(len(gold & top) / len(gold) if gold else 0.0)
    return sum(vals) / len(vals) if vals else 0.0


def macro_average(per_domain: dict[str, float]) -> float:
    return sum(per_domain.values()) / len(per_domain) if per_domain else 0.0


def write_trec(run: Run, path: str | Path, tag: str, depth: int = 1000) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for qid, ranking in run.items():
            seen = set()
            rank = 0
            for d, s in ranking:
                if d in seen:
                    continue
                seen.add(d)
                rank += 1
                if rank > depth:
                    break
                f.write(f"{qid} Q0 {d} {rank} {s:.6f} {tag}\n")


def read_trec(path: str | Path) -> Run:
    rows: dict[str, list[tuple[int, str, float]]] = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                qid, _, d, rank, score, _tag = line.split()
                rows.setdefault(qid, []).append((int(rank), d, float(score)))
    return {q: [(d, s) for _, d, s in sorted(v)] for q, v in rows.items()}


def summarize(per_domain_scores: dict[str, dict[str, float]]) -> dict:
    """``per_domain_scores``: domain -> qid -> nDCG. Returns per-domain means and macro."""
    dom = {d: (sum(v.values()) / len(v) if v else 0.0) for d, v in per_domain_scores.items()}
    return {"per_domain": {d: round(s, 4) for d, s in dom.items()},
            "macro_ndcg@10": round(macro_average(dom), 4),
            "num_topics": sum(len(v) for v in per_domain_scores.values())}


def iter_domains(selected: Iterable[str] | str, all_domains: list[str]) -> list[str]:
    if selected in ("all", ["all"], None):
        return list(all_domains)
    return [selected] if isinstance(selected, str) else list(selected)

# ChronoFuse

**Metadata- and time-aware hybrid retrieval for RETECO, SemEval-2027 Task 1, Track 1 (TEMPO).**

ChronoFuse targets temporal reasoning-intensive retrieval: ranking passages that are both on-topic
and *temporally aligned* with the question. It combines sparse, dense and knowledge-graph retrieval
over a metadata-enriched corpus with LLM-based temporal query understanding and reranking.

| | |
|---|---|
| Task | [RETECO](https://datascienceuibk.github.io/RETECO/), Sub-tracks **1a** (whole query) and **1b** (step-wise) |
| Data | [`DataScience-UIBK/RETECO-SemEval2027`](https://huggingface.co/datasets/DataScience-UIBK/RETECO-SemEval2027) v1.1: 13 domains, 1.17M docs, 1,730 queries, 3,976 steps |
| Metric | nDCG@10 (`pytrec_eval`), mean per domain, then macro-averaged over the 13 domains |
| Baseline | Official BM25: 1a **0.1147**, 1b **0.1177** (dev) |
| Paper | ACL format, [`paper/`](paper/) |

---

## Architecture

```
OFFLINE (per domain)                                    ONLINE (per query / step)
────────────────────                                    ─────────────────────────
documents.jsonl                                         query
  ├─ chunk (200w, 40 overlap) ─┬─ BM25 (bm25s)            └─ Query analyzer ── rules | LLM + guidance-distilled few-shot
  │                            └─ Dense (bge) → FAISS/Qdrant      → intent, year window, anchors, events,
  ├─ metadata profile ──────── years, granularity, entities,        entities, keywords, sub-queries, HyDE
  │                            title, short description       ┌──────────── channels ────────────┐
  ├─ temporal KG ───────────── doc–entity–year graph (NetworkX/Neo4j)
  └─ SQLite doc store                                     BM25(q) BM25(kw) BM25(subq) Dense(q) Dense(HyDE) KG-PPR
                                                                 └──── weighted RRF ────┘
                                                                 → temporal-alignment prior (λ)
                                                                 → cross-encoder (metadata-augmented)
                                                                 → temporal-aware LLM listwise rerank
                                                                 → TREC run
```

Key ideas:

1. **Metadata everywhere.** Each document gets a normalised temporal profile (years, ranges, decades,
   granularity, relative expressions), entities, proxy title and short description. These feed the
   KG, the vector-DB payload filter, the temporal prior and the reranker prompts.
2. **Guidance distillation.** TEMPO ships per-query *guidance* (temporal intent, anchors, events,
   retrieval plan) for train/dev only. We use train guidance as few-shot demonstrations so the LLM
   produces the same schema for unseen (test) queries.
3. **Hybrid multi-channel retrieval + RRF**, then a **temporal-alignment prior** and two-stage
   reranking.

## Project layout

```
.
├── configs/                  # YAML configs; experiments inherit from base.yaml
│   ├── base.yaml
│   └── experiments/          # e0_bm25 → e1_bm25_temporal → e2_hybrid → e3_full
├── src/chronofuse/
│   ├── data/                 # RETECO loaders (queries, steps, guidance, qrels, corpus)
│   ├── metadata/             # temporal extraction, chunking, document profiles
│   ├── indexing/             # BM25, dense (FAISS), Qdrant vector store
│   ├── kg/                   # temporal knowledge graph (NetworkX, Neo4j export)
│   ├── query/                # rule-based and guidance-distilled LLM query analyzers
│   ├── llm/                  # OpenAI-compatible client with disk cache
│   ├── retrieval/            # RRF / CombSUM fusion, temporal-alignment prior
│   ├── rerank/               # cross-encoder, LLM listwise reranker
│   ├── eval/                 # official nDCG@10 scoring, TREC I/O
│   ├── store.py              # SQLite document store
│   ├── pipeline.py           # build + run orchestration
│   └── cli.py                # `chronofuse build | run | evaluate`
├── scripts/                  # data download, official-scorer cross-check
├── tests/                    # pytest unit tests
├── paper/                    # ACL-format LaTeX paper (acl.sty, sections/, references.bib)
├── data/                     # raw/ (downloaded), indexes/ (built) — git-ignored
└── results/                  # runs/ (TREC files), metrics/ (JSON) — git-ignored
```

## Setup

```bash
conda env create -f environment.yml      # Python 3.11 + JDK 21 + requirements.txt
conda activate reteco
pip install -e .
cp .env.example .env                     # add an LLM endpoint for e3_full
make starter-kit                         # official scorer + format checker (third_party/)
```

## Data

```bash
python scripts/download_data.py --domains iota hsm   # small smoke-test subset
python scripts/download_data.py                      # all 13 Track 1 domains
```

## Usage

```bash
# 1) build indexes, metadata profiles, KG (per domain; cached)
chronofuse build --config configs/experiments/e0_bm25.yaml --domains iota

# 2) retrieve, write TREC runs, score
chronofuse run --config configs/experiments/e0_bm25.yaml --split dev --subtrack 1a --domains iota

# 3) cross-check one domain with the organisers' scorer
scripts/official_score.sh results/runs/e0_bm25/1a_dev iota dev
```

Runs go to `results/runs/<experiment>/<subtrack>_<split>/<domain>.trec` (six-column TREC format)
and metrics to `results/metrics/<experiment>__<subtrack>_<split>.json`.

## Experiments

| ID | Config | Components |
|---|---|---|
| E0 | `e0_bm25.yaml` | passage BM25 + MaxP |
| E1 | `e1_bm25_temporal.yaml` | E0 + temporal-alignment prior |
| E2 | `e2_hybrid.yaml` | E1 + dense + temporal KG, weighted RRF |
| E3 | `e3_full.yaml` | E2 + LLM query analysis (keywords, sub-queries, HyDE) + cross-encoder + LLM listwise |

Protocol: tune on **train**, report on **dev**, never use guidance or gold labels for the evaluated query.

## Tests

```bash
make test
```

## Licence and attribution

Code: MIT. RETECO text is CC BY-SA 4.0 (Stack Exchange); annotations are CC BY 4.0.
Please cite TEMPO and RECOR (see `paper/reteco_refs.bib`).

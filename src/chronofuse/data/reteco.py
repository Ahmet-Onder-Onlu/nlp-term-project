"""Loaders for the RETECO Track 1 (TEMPO) release, v1.1.

Layout (per domain)::

    track1_tempo/<domain>/
        documents.jsonl          {id, content}
        examples_{split}.jsonl   {id, query, gold_ids, gold_answers}       -> 1a
        steps_{split}.jsonl      {id, query, steps:[{step_id, step, step_instruction, gold_ids}]} -> 1b
        guidance_{split}.jsonl   {id, query_guidance, gold_passage_annotations}
        qrels_{split}.txt        qid 0 docid rel   (1a)
        qrels_steps_{split}.txt  step_id 0 docid rel (1b)

``guidance`` is only available for train/dev. It is used as supervision /
few-shot material for the query analyzer, never as an input at test time.
"""

from __future__ import annotations

import html
import re
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import orjson

TRACK1_DOMAINS = [
    "bitcoin",
    "cardano",
    "economics",
    "genealogy",
    "history",
    "hsm",
    "iota",
    "law",
    "monero",
    "politics",
    "quant",
    "travel",
    "workplace",
]

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def clean_html(text: str) -> str:
    """StackExchange queries carry HTML; strip tags and entities."""
    return _WS_RE.sub(" ", html.unescape(_TAG_RE.sub(" ", text))).strip()


@dataclass
class Document:
    id: str
    content: str


@dataclass
class Query:
    id: str
    text: str  # cleaned text (title + body)
    title: str  # first line of the StackExchange post
    raw: str
    gold_ids: list[str] = field(default_factory=list)


@dataclass
class Step:
    id: str  # <query_id>_step<n>
    query_id: str
    step: int
    instruction: str
    query: Query
    gold_ids: list[str] = field(default_factory=list)


def domain_dir(data_root: str | Path, domain: str) -> Path:
    return Path(data_root) / "track1_tempo" / domain


def _iter_jsonl(path: Path) -> Iterator[dict]:
    with open(path, "rb") as f:
        for line in f:
            if line.strip():
                yield orjson.loads(line)


def load_corpus(data_root: str | Path, domain: str) -> Iterator[Document]:
    for rec in _iter_jsonl(domain_dir(data_root, domain) / "documents.jsonl"):
        yield Document(id=rec.get("id") or rec["doc_id"], content=rec["content"])


def _make_query(rec: dict) -> Query:
    raw = rec["query"]
    title = raw.split("\n", 1)[0].strip()
    return Query(
        id=rec["id"],
        text=clean_html(raw),
        title=clean_html(title),
        raw=raw,
        gold_ids=list(rec.get("gold_ids", [])),
    )


def load_queries(data_root: str | Path, domain: str, split: str) -> list[Query]:
    return [
        _make_query(r)
        for r in _iter_jsonl(domain_dir(data_root, domain) / f"examples_{split}.jsonl")
    ]


def load_steps(data_root: str | Path, domain: str, split: str) -> list[Step]:
    out: list[Step] = []
    for rec in _iter_jsonl(domain_dir(data_root, domain) / f"steps_{split}.jsonl"):
        q = _make_query(rec)
        for st in rec.get("steps", []):
            out.append(
                Step(
                    id=st["step_id"],
                    query_id=rec["id"],
                    step=int(st["step"]),
                    instruction=st.get("step_instruction", ""),
                    query=q,
                    gold_ids=list(st.get("gold_ids", [])),
                )
            )
    return out


def load_guidance(data_root: str | Path, domain: str, split: str) -> dict[str, dict]:
    path = domain_dir(data_root, domain) / f"guidance_{split}.jsonl"
    if not path.exists():
        return {}
    return {r["id"]: r for r in _iter_jsonl(path)}


def load_qrels(
    data_root: str | Path, domain: str, split: str, steps: bool = False
) -> dict[str, dict[str, int]]:
    name = f"qrels_steps_{split}.txt" if steps else f"qrels_{split}.txt"
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with open(domain_dir(data_root, domain) / name) as f:
        for line in f:
            if line.strip():
                qid, _, docid, rel = line.split()
                qrels[qid][docid] = int(rel)
    return dict(qrels)

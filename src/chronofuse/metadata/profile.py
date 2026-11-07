"""Per-document metadata store.

For every document we persist a compact profile used by filtering, temporal
scoring, the knowledge graph and the LLM reranker:

* ``title``        first line / sentence (proxy title)
* ``description``  short extractive description (lead sentences); can be
                   replaced by an LLM-written summary for top candidates
* ``years``, ``granularity``, ``explicit``, ``relative``  temporal profile
* ``entities``     capitalised multi-word spans (cheap NER); spaCy optional
* ``n_words``      length
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import orjson

from ..data.reteco import Document
from .temporal import extract_temporal

_SENT_RE = re.compile(r"(?<=[.!?])\s+")
_ENT_RE = re.compile(r"\b([A-Z][a-zA-Z0-9]+(?:\s+(?:of|de|the|and|&)?\s*[A-Z][a-zA-Z0-9]+){0,4})\b")
_STOP_ENTS = {"The", "This", "That", "It", "In", "On", "A", "An", "If", "I", "We", "You", "He", "She",
              "They", "But", "And", "For", "As", "At", "To", "Is", "Are", "What", "When", "How", "Why"}


@dataclass
class DocProfile:
    id: str
    title: str
    description: str
    years: list[int] = field(default_factory=list)
    granularity: str = "none"
    explicit: list[str] = field(default_factory=list)
    relative: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    n_words: int = 0


def extract_entities(text: str, limit: int = 30) -> list[str]:
    counts: dict[str, int] = {}
    for m in _ENT_RE.finditer(text[:20000]):
        ent = m.group(1).strip()
        if ent in _STOP_ENTS or len(ent) < 3:
            continue
        counts[ent] = counts.get(ent, 0) + 1
    return [e for e, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:limit]]


def build_profile(doc: Document, desc_sentences: int = 2) -> DocProfile:
    text = doc.content.strip()
    sents = _SENT_RE.split(text[:4000])
    title = (text.split("\n", 1)[0] if "\n" in text[:300] else sents[0])[:200]
    tp = extract_temporal(text)
    return DocProfile(
        id=doc.id, title=title, description=" ".join(sents[:desc_sentences])[:500],
        years=tp.years, granularity=tp.granularity, explicit=tp.explicit[:20],
        relative=tp.relative[:20], entities=extract_entities(text), n_words=len(text.split()),
    )


def write_profiles(profiles: Iterable[DocProfile], path: str | Path) -> int:
    n = 0
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        for p in profiles:
            f.write(orjson.dumps(asdict(p)) + b"\n")
            n += 1
    return n


def read_profiles(path: str | Path) -> dict[str, DocProfile]:
    out: dict[str, DocProfile] = {}
    with open(path, "rb") as f:
        for line in f:
            d = orjson.loads(line)
            out[d["id"]] = DocProfile(**d)
    return out

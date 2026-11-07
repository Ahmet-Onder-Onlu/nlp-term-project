"""Passage chunking.

TEMPO documents range from one paragraph to multi-megabyte dumps. We split each
document into word windows with overlap; passage ids are ``<doc_id>#<k>`` and
document scores are aggregated with MaxP (max over passages) at retrieval time.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

SEP = "#"


@dataclass
class Passage:
    pid: str
    doc_id: str
    text: str
    index: int


def chunk_document(doc_id: str, text: str, size: int = 200, overlap: int = 40,
                   max_passages: int = 64) -> Iterator[Passage]:
    words = text.split()
    if not words:
        yield Passage(f"{doc_id}{SEP}0", doc_id, "", 0)
        return
    stride = max(1, size - overlap)
    for k, start in enumerate(range(0, len(words), stride)):
        if k >= max_passages:
            break
        yield Passage(f"{doc_id}{SEP}{k}", doc_id, " ".join(words[start:start + size]), k)
        if start + size >= len(words):
            break


def doc_id_of(pid: str) -> str:
    return pid.rsplit(SEP, 1)[0]

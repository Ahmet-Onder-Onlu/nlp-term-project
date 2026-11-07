"""Common retriever interface and MaxP aggregation."""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..metadata.chunking import doc_id_of

Ranking = list[tuple[str, float]]  # [(doc_id, score)] sorted desc


def maxp(pids: list[str], scores: list[float], k: int) -> Ranking:
    """Aggregate passage hits to document scores (max over passages)."""
    best: dict[str, float] = {}
    for pid, s in zip(pids, scores):
        d = doc_id_of(pid)
        if s > best.get(d, float("-inf")):
            best[d] = float(s)
    return sorted(best.items(), key=lambda kv: -kv[1])[:k]


class Retriever(ABC):
    name: str = "retriever"

    @abstractmethod
    def search(self, queries: list[str], k: int = 100) -> list[Ranking]:
        """Return one document ranking per query."""

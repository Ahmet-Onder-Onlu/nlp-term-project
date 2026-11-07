"""Rank fusion: Reciprocal Rank Fusion and normalised weighted CombSUM."""
from __future__ import annotations

from ..indexing.base import Ranking


def rrf(rankings: dict[str, Ranking], weights: dict[str, float] | None = None, k: int = 60,
        top: int = 1000) -> Ranking:
    """Weighted RRF: score(d) = sum_c w_c / (k + rank_c(d))."""
    weights = weights or {}
    scores: dict[str, float] = {}
    for name, ranking in rankings.items():
        w = weights.get(name, 1.0)
        if w == 0:
            continue
        for r, (doc, _) in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + w / (k + r)
    return sorted(scores.items(), key=lambda kv: -kv[1])[:top]


def _minmax(ranking: Ranking) -> dict[str, float]:
    if not ranking:
        return {}
    vals = [s for _, s in ranking]
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    return {d: (s - lo) / span for d, s in ranking}


def combsum(rankings: dict[str, Ranking], weights: dict[str, float] | None = None,
            top: int = 1000) -> Ranking:
    weights = weights or {}
    scores: dict[str, float] = {}
    for name, ranking in rankings.items():
        w = weights.get(name, 1.0)
        for doc, s in _minmax(ranking).items():
            scores[doc] = scores.get(doc, 0.0) + w * s
    return sorted(scores.items(), key=lambda kv: -kv[1])[:top]

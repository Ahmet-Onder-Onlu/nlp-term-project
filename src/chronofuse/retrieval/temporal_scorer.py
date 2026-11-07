"""Temporal alignment prior.

Combines a candidate's relevance score with how well the document's temporal
profile matches the query's temporal need:

    s'(d) = (1 - λ) · norm(s(d)) + λ · T(q, d)

    T(q, d) = year_overlap(q, d)           if the query has an explicit/inferred window
            = has_time(d) · τ               if the query is temporal but un-anchored
            = 0                             otherwise

λ is tuned per configuration on the train split.
"""
from __future__ import annotations

from ..indexing.base import Ranking
from ..metadata.profile import DocProfile
from ..metadata.temporal import year_overlap
from ..query.analyzer import QueryAnalysis


def temporal_alignment(qa: QueryAnalysis, prof: DocProfile | None, tolerance: int = 1,
                       unanchored_bonus: float = 0.3) -> float:
    if prof is None or not qa.is_temporal:
        return 0.0
    if qa.years:
        return year_overlap(qa.years, prof.years, tolerance)
    return unanchored_bonus if (prof.years or prof.relative) else 0.0


def apply_temporal_prior(ranking: Ranking, qa: QueryAnalysis, profiles: dict[str, DocProfile],
                         lam: float = 0.2, **kw) -> Ranking:
    if lam <= 0 or not ranking:
        return ranking
    vals = [s for _, s in ranking]
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    rescored = [(d, (1 - lam) * (s - lo) / span + lam * temporal_alignment(qa, profiles.get(d), **kw))
                for d, s in ranking]
    return sorted(rescored, key=lambda kv: -kv[1])

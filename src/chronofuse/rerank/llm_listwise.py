"""Temporal-aware listwise LLM reranking (RankGPT-style sliding window).

The prompt includes the query's analysed temporal need (intent, window,
anchors) and each candidate's extracted metadata, and instructs the model to
prefer passages that are *temporally aligned*, not merely topical — mirroring
RETECO's definition of relevance.
"""

from __future__ import annotations

import re

from ..indexing.base import Ranking
from ..metadata.profile import DocProfile
from ..query.analyzer import QueryAnalysis
from .cross_encoder import render_candidate

_PROMPT = """Rank the passages by how well they help answer the question, considering BOTH topic \
and time. A passage about the right topic but the wrong time period is less useful than one that \
covers the period/events the question depends on.

Question: {query}
Temporal need: intent={intent}; window={window}; anchors={anchors}; events={events}

Passages:
{passages}

Output the passage identifiers in descending order of usefulness, e.g. [3] > [1] > [2]. \
Output only the ranking."""


class LLMListwiseReranker:
    def __init__(self, llm, window: int = 20, step: int = 10, max_words: int = 120):
        self.llm, self.window, self.step, self.max_words = llm, window, step, max_words

    def _rank_window(
        self,
        query: str,
        qa: QueryAnalysis,
        docs: list[str],
        texts: dict[str, str],
        profiles: dict[str, DocProfile],
    ) -> list[str]:
        passages = "\n".join(
            f"[{i + 1}] {render_candidate(texts.get(d, ''), profiles.get(d), self.max_words)}"
            for i, d in enumerate(docs)
        )
        prompt = _PROMPT.format(
            query=query[:1500],
            intent=qa.temporal_intent,
            window=qa.year_window,
            anchors=qa.key_time_anchors[:5],
            events=qa.temporal_events[:5],
            passages=passages,
        )
        out = self.llm.chat([{"role": "user", "content": prompt}], max_tokens=200)
        order, seen = [], set()
        for m in re.findall(r"\[(\d+)\]", out):
            i = int(m) - 1
            if 0 <= i < len(docs) and i not in seen:
                order.append(docs[i])
                seen.add(i)
        return order + [d for i, d in enumerate(docs) if i not in seen]

    def rerank(
        self,
        query: str,
        qa: QueryAnalysis,
        ranking: Ranking,
        texts: dict[str, str],
        profiles: dict[str, DocProfile],
        top_n: int = 40,
    ) -> Ranking:
        docs = [d for d, _ in ranking[:top_n]]
        # Back-to-front sliding window (RankGPT).
        end = len(docs)
        while True:
            start = max(0, end - self.window)
            docs[start:end] = self._rank_window(query, qa, docs[start:end], texts, profiles)
            if start == 0:
                break
            end -= self.step
        n = len(ranking)
        return [(d, float(n - i)) for i, d in enumerate(docs)] + [
            (d, float(n - top_n - i - 1)) for i, (d, _) in enumerate(ranking[top_n:])
        ]

"""Temporal query understanding.

Produces a :class:`QueryAnalysis` whose schema mirrors TEMPO's
``guidance.query_guidance`` (temporal intent, anchors, events, granularity,
retrieval plan) plus retrieval-oriented fields (keyword query, HyDE passage,
year window, entities).

Two implementations:

* :class:`RuleQueryAnalyzer`  — no LLM; regex temporal profile + title keywords.
* :class:`LLMQueryAnalyzer`   — LLM with *guidance-distilled* few-shot examples:
  train-split guidance records are used as demonstrations, so the model learns
  to emit the same annotation schema the benchmark authors used. Guidance is
  never read for the query being analysed.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from ..data.reteco import Query
from ..metadata.profile import extract_entities
from ..metadata.temporal import extract_temporal

TEMPORAL_INTENTS = ["point_in_time", "period_range", "ongoing_status", "before_after",
                    "change_over_time", "duration", "recency", "none"]


@dataclass
class QueryAnalysis:
    qid: str
    is_temporal: bool = False
    temporal_intent: str = "none"
    granularity: str = "none"
    year_window: tuple[int, int] | None = None
    key_time_anchors: list[str] = field(default_factory=list)
    temporal_events: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    keyword_query: str = ""
    sub_queries: list[str] = field(default_factory=list)   # retrieval plan steps
    hyde_passage: str = ""
    summary: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def years(self) -> list[int]:
        if not self.year_window:
            return []
        a, b = self.year_window
        return list(range(a, b + 1)) if b - a <= 50 else [a, b]


class RuleQueryAnalyzer:
    def analyze(self, q: Query) -> QueryAnalysis:
        tp = extract_temporal(q.text)
        return QueryAnalysis(
            qid=q.id, is_temporal=bool(tp.years or tp.relative),
            temporal_intent="point_in_time" if tp.years else ("recency" if tp.relative else "none"),
            granularity=tp.granularity, year_window=tp.span,
            key_time_anchors=tp.explicit + tp.relative, entities=extract_entities(q.text, 10),
            keyword_query=q.title, summary=q.title,
        )


_SYSTEM = """You are a temporal information-retrieval analyst. Given a user question from a \
StackExchange-style forum, produce a JSON object describing its temporal information need and \
a retrieval plan. Fields:
- is_temporal (bool)
- temporal_intent: one of %s
- granularity: day|week|month|year|decade|century|none
- year_window: [start_year, end_year] or null — the period the evidence must be about (infer \
implicit periods from events, e.g. "the Mt. Gox collapse" -> [2014, 2014])
- key_time_anchors: list of time expressions or anchor events
- temporal_events: list of events the answer depends on
- entities: key named entities
- keyword_query: a concise BM25 keyword query (8-20 terms, include implied dates/version names)
- sub_queries: 2-4 retrieval steps, each a self-contained search query
- hyde_passage: a ~80-word hypothetical passage that would answer the question, with dates
- summary: one sentence
Return ONLY JSON.""" % TEMPORAL_INTENTS


def _guidance_to_target(rec: dict) -> dict:
    g = rec.get("query_guidance", {})
    return {
        "is_temporal": g.get("is_temporal_query", False),
        "temporal_intent": g.get("temporal_intent", "none"),
        "granularity": g.get("expected_granularity", "none"),
        "key_time_anchors": g.get("key_time_anchors", []),
        "temporal_events": g.get("query_temporal_events", []),
        "sub_queries": [s.get("action", "") for s in g.get("retrieval_plan", [])],
        "summary": g.get("query_summary", ""),
    }


class LLMQueryAnalyzer:
    def __init__(self, llm, demos: list[tuple[Query, dict]] | None = None, max_chars: int = 2500):
        """``demos``: (train query, guidance record) pairs used as few-shot examples."""
        self.llm = llm
        self.demos = demos or []
        self.max_chars = max_chars
        self.fallback = RuleQueryAnalyzer()

    def _messages(self, q: Query) -> list[dict]:
        msgs = [{"role": "system", "content": _SYSTEM}]
        for dq, grec in self.demos:
            msgs.append({"role": "user", "content": dq.text[: self.max_chars]})
            msgs.append({"role": "assistant", "content": json.dumps(_guidance_to_target(grec))})
        msgs.append({"role": "user", "content": q.text[: self.max_chars]})
        return msgs

    def analyze(self, q: Query) -> QueryAnalysis:
        base = self.fallback.analyze(q)
        try:
            out = self.llm.chat_json(self._messages(q))
        except Exception:  # noqa: BLE001 - degrade gracefully to rules
            return base
        yw = out.get("year_window")
        year_window = tuple(int(x) for x in yw) if isinstance(yw, list) and len(yw) == 2 and all(
            isinstance(x, (int, float)) for x in yw) else base.year_window
        return QueryAnalysis(
            qid=q.id, is_temporal=bool(out.get("is_temporal", base.is_temporal)),
            temporal_intent=out.get("temporal_intent", base.temporal_intent),
            granularity=out.get("granularity", base.granularity), year_window=year_window,
            key_time_anchors=out.get("key_time_anchors", []) or base.key_time_anchors,
            temporal_events=out.get("temporal_events", []),
            entities=out.get("entities", []) or base.entities,
            keyword_query=out.get("keyword_query", "") or base.keyword_query,
            sub_queries=[s for s in out.get("sub_queries", []) if isinstance(s, str)][:4],
            hyde_passage=out.get("hyde_passage", ""), summary=out.get("summary", ""),
        )

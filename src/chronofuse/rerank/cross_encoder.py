"""Cross-encoder reranking with metadata-augmented document text.

Each candidate is presented as ``[years] title — best passage`` so the reranker
sees the document's temporal metadata explicitly.
"""

from __future__ import annotations

from ..indexing.base import Ranking
from ..metadata.profile import DocProfile


def render_candidate(doc_text: str, prof: DocProfile | None, max_words: int = 300) -> str:
    head = ""
    if prof is not None:
        yrs = f"[{prof.years[0]}–{prof.years[-1]}] " if prof.years else ""
        head = f"{yrs}{prof.title} — "
    return head + " ".join(doc_text.split()[:max_words])


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        max_length: int = 512,
        batch_size: int = 16,
        device: str | None = None,
    ):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, max_length=max_length, device=device)
        self.batch_size = batch_size

    def rerank(
        self,
        query: str,
        ranking: Ranking,
        texts: dict[str, str],
        profiles: dict[str, DocProfile] | None = None,
        top_n: int = 100,
    ) -> Ranking:
        head, tail = ranking[:top_n], ranking[top_n:]
        pairs = [
            (query, render_candidate(texts.get(d, ""), (profiles or {}).get(d))) for d, _ in head
        ]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        reranked = sorted(zip([d for d, _ in head], map(float, scores)), key=lambda kv: -kv[1])
        floor = min((s for _, s in reranked), default=0.0) - 1.0
        return reranked + [(d, floor - i * 1e-6) for i, (d, _) in enumerate(tail)]

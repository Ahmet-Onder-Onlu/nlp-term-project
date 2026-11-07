"""Passage-level BM25 via ``bm25s`` (Lucene scoring variant), with MaxP to documents."""

from __future__ import annotations

from pathlib import Path

import bm25s
import orjson
import Stemmer

from .base import Ranking, Retriever, maxp


class BM25Index(Retriever):
    name = "bm25"

    def __init__(self, k1: float = 0.9, b: float = 0.4, stemmer: str = "english"):
        self.k1, self.b = k1, b
        self.stemmer = Stemmer.Stemmer(stemmer) if stemmer else None
        self.model: bm25s.BM25 | None = None
        self.pids: list[str] = []

    # ------------------------------------------------------------------ build
    def build(self, pids: list[str], texts: list[str]) -> BM25Index:
        tokens = bm25s.tokenize(texts, stopwords="en", stemmer=self.stemmer, show_progress=False)
        self.model = bm25s.BM25(k1=self.k1, b=self.b, method="lucene")
        self.model.index(tokens, show_progress=False)
        self.pids = list(pids)
        return self

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        assert self.model is not None
        self.model.save(str(path))
        (path / "pids.json").write_bytes(orjson.dumps(self.pids))

    @classmethod
    def load(cls, path: str | Path, stemmer: str = "english") -> BM25Index:
        path = Path(path)
        obj = cls(stemmer=stemmer)
        obj.model = bm25s.BM25.load(str(path))
        obj.pids = orjson.loads((path / "pids.json").read_bytes())
        return obj

    # ----------------------------------------------------------------- search
    def search(
        self, queries: list[str], k: int = 100, passage_k: int | None = None
    ) -> list[Ranking]:
        assert self.model is not None, "index not built/loaded"
        passage_k = min(passage_k or k * 5, len(self.pids))
        q_tokens = bm25s.tokenize(
            queries, stopwords="en", stemmer=self.stemmer, show_progress=False
        )
        idx, scores = self.model.retrieve(q_tokens, k=passage_k, show_progress=False)
        out: list[Ranking] = []
        for row_i, row_s in zip(idx, scores):
            pids = [self.pids[i] for i in row_i]
            out.append([(d, s) for d, s in maxp(pids, list(row_s), k) if s > 0])
        return out

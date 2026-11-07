"""Dense bi-encoder retrieval (sentence-transformers + FAISS inner-product index).

Default encoder is ``BAAI/bge-small-en-v1.5`` (fast on CPU). For the final
system, swap to a stronger encoder (e.g. ``BAAI/bge-m3``, ``intfloat/e5-large-v2``
or a reasoning-oriented retriever) on GPU via the config.
"""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np
import orjson

from .base import Ranking, Retriever, maxp


class DenseIndex(Retriever):
    name = "dense"

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        query_prefix: str = "",
        doc_prefix: str = "",
        batch_size: int = 64,
        max_seq_length: int = 256,
        device: str | None = None,
    ):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.query_prefix, self.doc_prefix = query_prefix, doc_prefix
        self.batch_size = batch_size
        self.encoder = SentenceTransformer(model_name, device=device)
        self.encoder.max_seq_length = max_seq_length
        self.index: faiss.Index | None = None
        self.pids: list[str] = []

    def encode(self, texts: list[str], is_query: bool) -> np.ndarray:
        prefix = self.query_prefix if is_query else self.doc_prefix
        emb = self.encoder.encode(
            [prefix + t for t in texts],
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=not is_query,
            convert_to_numpy=True,
        )
        return emb.astype(np.float32)

    def build(self, pids: list[str], texts: list[str]) -> DenseIndex:
        emb = self.encode(texts, is_query=False)
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb)
        self.pids = list(pids)
        return self

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "faiss.index"))
        (path / "pids.json").write_bytes(orjson.dumps(self.pids))
        (path / "meta.json").write_bytes(orjson.dumps({"model": self.model_name}))

    @classmethod
    def load(cls, path: str | Path, **kwargs) -> DenseIndex:
        path = Path(path)
        meta = orjson.loads((path / "meta.json").read_bytes())
        obj = cls(model_name=kwargs.pop("model_name", meta["model"]), **kwargs)
        obj.index = faiss.read_index(str(path / "faiss.index"))
        obj.pids = orjson.loads((path / "pids.json").read_bytes())
        return obj

    def search(
        self, queries: list[str], k: int = 100, passage_k: int | None = None
    ) -> list[Ranking]:
        assert self.index is not None, "index not built/loaded"
        passage_k = min(passage_k or k * 5, len(self.pids))
        q = self.encode(queries, is_query=True)
        scores, idx = self.index.search(q, passage_k)
        return [
            maxp([self.pids[i] for i in row_i if i >= 0], list(row_s), k)
            for row_i, row_s in zip(idx, scores)
        ]

"""Qdrant vector database backend with temporal/metadata payload filtering.

Runs embedded (``path=...``, no server) by default, or against a server when
``QDRANT_URL`` is set. Each point stores the passage vector plus payload
``{doc_id, years, granularity, entities}`` so retrieval can be restricted to a
query's time window (``year_range``) — the "metadata-filtered dense" channel.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np
from qdrant_client import QdrantClient, models

from .base import Ranking, maxp


class QdrantStore:
    def __init__(self, collection: str, dim: int, path: str | None = None, url: str | None = None):
        url = url or os.getenv("QDRANT_URL") or None
        self.client = QdrantClient(url=url) if url else QdrantClient(path=path or ":memory:")
        self.collection = collection
        if not self.client.collection_exists(collection):
            self.client.create_collection(
                collection,
                vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
            )
            self.client.create_payload_index(collection, "years", models.PayloadSchemaType.INTEGER)

    def upsert(
        self, pids: Sequence[str], vectors: np.ndarray, payloads: Sequence[dict], batch: int = 1024
    ) -> None:
        for s in range(0, len(pids), batch):
            self.client.upsert(
                self.collection,
                points=[
                    models.PointStruct(
                        id=s + i,
                        vector=vectors[s + i].tolist(),
                        payload={"pid": pids[s + i], **payloads[s + i]},
                    )
                    for i in range(min(batch, len(pids) - s))
                ],
            )

    def search(
        self,
        qvec: np.ndarray,
        k: int = 100,
        year_range: tuple[int, int] | None = None,
        passage_k: int | None = None,
    ) -> Ranking:
        flt = None
        if year_range:
            flt = models.Filter(
                must=[
                    models.FieldCondition(
                        key="years", range=models.Range(gte=year_range[0], lte=year_range[1])
                    )
                ]
            )
        hits = self.client.query_points(
            self.collection,
            query=qvec.tolist(),
            query_filter=flt,
            limit=passage_k or k * 5,
            with_payload=["pid"],
        ).points
        return maxp([h.payload["pid"] for h in hits], [h.score for h in hits], k)

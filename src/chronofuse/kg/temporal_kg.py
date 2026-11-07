"""Temporal knowledge graph over a domain corpus.

Nodes:  ``doc:<id>``, ``ent:<entity>``, ``year:<yyyy>``
Edges:  doc—MENTIONS→ent, doc—REFERS_TO→year, ent—ACTIVE_IN→year (via co-mention)

Retrieval channel: seed the graph with the query's entities and years and run
personalised PageRank; documents receive the stationary mass as a score. This
surfaces passages that share the query's *(entity, period)* context even when
they share few query terms — the core failure mode of BM25 on TEMPO.

The graph lives in NetworkX (in-process). ``export_neo4j`` writes the same
graph to a Neo4j database for inspection / Cypher queries.
"""
from __future__ import annotations

import pickle
from collections.abc import Iterable
from pathlib import Path

import networkx as nx

from ..metadata.profile import DocProfile


def _norm_ent(e: str) -> str:
    return e.strip().lower()


class TemporalKG:
    def __init__(self, max_entity_df: float = 0.05):
        self.g = nx.Graph()
        self.max_entity_df = max_entity_df

    def build(self, profiles: Iterable[DocProfile]) -> TemporalKG:
        profiles = list(profiles)
        df: dict[str, int] = {}
        for p in profiles:
            for e in {_norm_ent(x) for x in p.entities}:
                df[e] = df.get(e, 0) + 1
        cap = max(5, int(self.max_entity_df * len(profiles)))
        for p in profiles:
            dn = f"doc:{p.id}"
            self.g.add_node(dn, kind="doc")
            ents = [e for e in {_norm_ent(x) for x in p.entities} if df.get(e, 0) <= cap]
            years = p.years if len(p.years) <= 30 else p.years[:: max(1, len(p.years) // 30)]
            for e in ents:
                self.g.add_edge(dn, f"ent:{e}", rel="MENTIONS")
            for y in years:
                self.g.add_edge(dn, f"year:{y}", rel="REFERS_TO")
            for e in ents[:10]:
                for y in years[:10]:
                    u, v = f"ent:{e}", f"year:{y}"
                    w = self.g.get_edge_data(u, v, {}).get("weight", 0) + 1
                    self.g.add_edge(u, v, rel="ACTIVE_IN", weight=w)
        return self

    def search(self, entities: list[str], years: list[int], k: int = 100, alpha: float = 0.85) -> list[tuple[str, float]]:
        seeds = [f"ent:{_norm_ent(e)}" for e in entities] + [f"year:{y}" for y in years]
        seeds = [s for s in seeds if s in self.g]
        if not seeds:
            return []
        pers = {s: 1.0 / len(seeds) for s in seeds}
        # Restrict PPR to the 2-hop neighbourhood of the seeds for speed.
        nodes = set(seeds)
        for s in seeds:
            for n in self.g.neighbors(s):
                nodes.add(n)
                if len(nodes) > 50000:
                    break
        sub = self.g.subgraph(nodes)
        pr = nx.pagerank(sub, alpha=alpha, personalization=pers, max_iter=50, tol=1e-6)
        docs = [(n[4:], s) for n, s in pr.items() if n.startswith("doc:")]
        return sorted(docs, key=lambda kv: -kv[1])[:k]

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.g, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> TemporalKG:
        obj = cls()
        with open(path, "rb") as f:
            obj.g = pickle.load(f)
        return obj

    def export_neo4j(self, uri: str, user: str, password: str, domain: str, batch: int = 5000) -> None:
        from neo4j import GraphDatabase

        edges = [(u, v, d.get("rel", "REL")) for u, v, d in self.g.edges(data=True)]
        with GraphDatabase.driver(uri, auth=(user, password)) as drv, drv.session() as s:
            for i in range(0, len(edges), batch):
                s.run(
                    "UNWIND $rows AS r MERGE (a:Node {key:r[0], domain:$d}) "
                    "MERGE (b:Node {key:r[1], domain:$d}) MERGE (a)-[:LINK {rel:r[2]}]->(b)",
                    rows=edges[i:i + batch], d=domain)

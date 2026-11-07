"""SQLite document store: doc_id -> (truncated) content, for rerankers and inspection."""
from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from pathlib import Path


class DocStore:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.execute("CREATE TABLE IF NOT EXISTS docs (id TEXT PRIMARY KEY, content TEXT)")

    def add_many(self, rows: Iterable[tuple[str, str]], max_chars: int = 20000) -> None:
        self.conn.executemany("INSERT OR REPLACE INTO docs VALUES (?, ?)",
                              ((i, c[:max_chars]) for i, c in rows))
        self.conn.commit()

    def get_many(self, ids: list[str]) -> dict[str, str]:
        out: dict[str, str] = {}
        for s in range(0, len(ids), 500):
            chunk = ids[s:s + 500]
            q = f"SELECT id, content FROM docs WHERE id IN ({','.join('?' * len(chunk))})"
            out.update(dict(self.conn.execute(q, chunk).fetchall()))
        return out

    def __len__(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM docs").fetchone()[0]

# src/isotope/stores/sqlite_chunk.py
"""SQLite chunk store implementation."""

import json
import sqlite3
from pathlib import Path

from isotope.models import Chunk
from isotope.stores.base import ChunkStore


class SQLiteChunkStore(ChunkStore):
    """SQLite-based chunk store."""

    def __init__(self, db_path: str) -> None:
        """Initialize the SQLite store."""
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Create tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    source TEXT NOT NULL,
                    metadata TEXT NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_source ON chunks(source)")
            conn.commit()

    def put(self, chunk: Chunk) -> None:
        """Store a chunk, overwriting if exists."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO chunks (id, content, source, metadata)
                VALUES (?, ?, ?, ?)
                """,
                (chunk.id, chunk.content, chunk.source, json.dumps(chunk.metadata)),
            )
            conn.commit()

    def get(self, chunk_id: str) -> Chunk | None:
        """Retrieve a chunk by ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT id, content, source, metadata FROM chunks WHERE id = ?",
                (chunk_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return Chunk(
                id=row[0],
                content=row[1],
                source=row[2],
                metadata=json.loads(row[3]),
            )

    def get_many(self, chunk_ids: list[str]) -> list[Chunk]:
        """Retrieve multiple chunks by ID."""
        if not chunk_ids:
            return []
        placeholders = ",".join("?" * len(chunk_ids))
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                f"SELECT id, content, source, metadata FROM chunks WHERE id IN ({placeholders})",
                chunk_ids,
            )
            return [
                Chunk(id=row[0], content=row[1], source=row[2], metadata=json.loads(row[3]))
                for row in cursor.fetchall()
            ]

    def delete(self, chunk_id: str) -> None:
        """Delete a chunk by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM chunks WHERE id = ?", (chunk_id,))
            conn.commit()

    def put_many(self, chunks: list[Chunk]) -> None:
        """Store multiple chunks, overwriting if they exist."""
        if not chunks:
            return
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO chunks (id, content, source, metadata)
                VALUES (?, ?, ?, ?)
                """,
                [(c.id, c.content, c.source, json.dumps(c.metadata)) for c in chunks],
            )
            conn.commit()

    def delete_many(self, chunk_ids: list[str]) -> None:
        """Delete multiple chunks by ID."""
        if not chunk_ids:
            return
        placeholders = ",".join("?" * len(chunk_ids))
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(f"DELETE FROM chunks WHERE id IN ({placeholders})", chunk_ids)
            conn.commit()

    def count_chunks(self) -> int:
        """Count the total number of chunks in the store."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(id) FROM chunks")
            count = cursor.fetchone()
            return count[0] if count else 0

    def list_sources(self) -> list[str]:
        """List all unique sources."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT DISTINCT source FROM chunks")
            return [row[0] for row in cursor.fetchall()]

    def get_by_source(self, source: str) -> list[Chunk]:
        """Get all chunks from a specific source."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT id, content, source, metadata FROM chunks WHERE source = ?",
                (source,),
            )
            return [
                Chunk(id=row[0], content=row[1], source=row[2], metadata=json.loads(row[3]))
                for row in cursor.fetchall()
            ]

    def get_chunk_ids_by_source(self, source: str) -> list[str]:
        """Get all chunk IDs for a specific source."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT id FROM chunks WHERE source = ?",
                (source,),
            )
            return [row[0] for row in cursor.fetchall()]

    def delete_by_source(self, source: str) -> None:
        """Delete all chunks from a specific source."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM chunks WHERE source = ?", (source,))
            conn.commit()

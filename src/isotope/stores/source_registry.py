# src/isotope/stores/source_registry.py
"""SQLite implementation of source registry."""

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from isotope.stores.base import SourceRegistry


class SQLiteSourceRegistry(SourceRegistry):
    """SQLite-backed source registry."""

    def __init__(self, db_path: str) -> None:
        """Initialize the registry.

        Args:
            db_path: Path to SQLite database file
        """
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the database schema."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sources (
                    source TEXT PRIMARY KEY,
                    content_hash TEXT NOT NULL,
                    ingested_at TEXT NOT NULL
                )
            """)

    def get_hash(self, source: str) -> str | None:
        """Get content hash for a source, or None if not tracked."""
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute(
                "SELECT content_hash FROM sources WHERE source = ?",
                (source,),
            )
            row = cursor.fetchone()
            return row[0] if row else None

    def set_hash(self, source: str, content_hash: str) -> None:
        """Store content hash after successful ingestion."""
        now = datetime.now(UTC).isoformat()
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO sources (source, content_hash, ingested_at)
                VALUES (?, ?, ?)
                """,
                (source, content_hash, now),
            )

    def delete(self, source: str) -> None:
        """Remove tracking for a source."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("DELETE FROM sources WHERE source = ?", (source,))

    def list_sources(self) -> list[str]:
        """List all tracked sources."""
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute("SELECT source FROM sources")
            return [row[0] for row in cursor.fetchall()]

# src/isotope/stores/sqlite_atom.py
"""SQLite atom store implementation."""

import sqlite3
from pathlib import Path

from isotope.models import Atom
from isotope.stores.base import AtomStore


class SQLiteAtomStore(AtomStore):
    """SQLite-based atom store."""

    def __init__(self, db_path: str) -> None:
        """Initialize the SQLite atom store."""
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Create tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS atoms (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    atom_index INTEGER DEFAULT 0
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_atom_chunk ON atoms(chunk_id)")
            conn.commit()

    def put(self, atom: Atom) -> None:
        """Store an atom, overwriting if exists."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO atoms (id, content, chunk_id, atom_index)
                VALUES (?, ?, ?, ?)
                """,
                (atom.id, atom.content, atom.chunk_id, atom.index),
            )
            conn.commit()

    def put_many(self, atoms: list[Atom]) -> None:
        """Store multiple atoms."""
        if not atoms:
            return
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO atoms (id, content, chunk_id, atom_index)
                VALUES (?, ?, ?, ?)
                """,
                [(a.id, a.content, a.chunk_id, a.index) for a in atoms],
            )
            conn.commit()

    def get(self, atom_id: str) -> Atom | None:
        """Retrieve an atom by ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT id, content, chunk_id, atom_index FROM atoms WHERE id = ?",
                (atom_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return Atom(
                id=row[0],
                content=row[1],
                chunk_id=row[2],
                index=row[3],
            )

    def get_by_chunk(self, chunk_id: str) -> list[Atom]:
        """Get all atoms from a chunk, ordered by index."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT id, content, chunk_id, atom_index FROM atoms "
                "WHERE chunk_id = ? ORDER BY atom_index",
                (chunk_id,),
            )
            return [
                Atom(id=row[0], content=row[1], chunk_id=row[2], index=row[3])
                for row in cursor.fetchall()
            ]

    def delete_by_chunk_ids(self, chunk_ids: list[str]) -> None:
        """Delete all atoms for given chunks."""
        if not chunk_ids:
            return
        placeholders = ",".join("?" * len(chunk_ids))
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(f"DELETE FROM atoms WHERE chunk_id IN ({placeholders})", chunk_ids)
            conn.commit()

    def list_chunk_ids(self) -> set[str]:
        """List all unique chunk IDs with atoms."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT DISTINCT chunk_id FROM atoms")
            return {row[0] for row in cursor.fetchall()}

    def count_atoms(self) -> int:
        """Count the total number of atoms in the store."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(id) FROM atoms")
            count = cursor.fetchone()
            return count[0] if count else 0

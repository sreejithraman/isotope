# tests/stores/test_sqlite_atom.py
"""Tests for SQLite atom store."""

import os
import tempfile

import pytest

from isotope.models import Atom
from isotope.stores.base import AtomStore
from isotope.stores.sqlite_atom import SQLiteAtomStore


@pytest.fixture
def temp_db():
    """Create a temporary database file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield os.path.join(tmpdir, "test.db")


@pytest.fixture
def atom_store(temp_db):
    """Create a SQLiteAtomStore instance."""
    return SQLiteAtomStore(temp_db)


class TestSQLiteAtomStore:
    def test_is_atomstore(self, atom_store):
        assert isinstance(atom_store, AtomStore)

    def test_put_and_get_atom(self, atom_store):
        atom = Atom(content="Python is interpreted.", chunk_id="c1", index=0)
        atom_store.put(atom)

        retrieved = atom_store.get(atom.id)
        assert retrieved is not None
        assert retrieved.id == atom.id
        assert retrieved.content == "Python is interpreted."
        assert retrieved.chunk_id == "c1"
        assert retrieved.index == 0

    def test_put_many_atoms(self, atom_store):
        atoms = [
            Atom(content="First fact.", chunk_id="c1", index=0),
            Atom(content="Second fact.", chunk_id="c1", index=1),
            Atom(content="Third fact.", chunk_id="c2", index=0),
        ]
        atom_store.put_many(atoms)

        for atom in atoms:
            retrieved = atom_store.get(atom.id)
            assert retrieved is not None
            assert retrieved.content == atom.content

    def test_get_by_chunk_ordered_by_index(self, atom_store):
        # Insert out of order
        atoms = [
            Atom(content="Third.", chunk_id="c1", index=2),
            Atom(content="First.", chunk_id="c1", index=0),
            Atom(content="Second.", chunk_id="c1", index=1),
        ]
        atom_store.put_many(atoms)

        retrieved = atom_store.get_by_chunk("c1")
        assert len(retrieved) == 3
        assert retrieved[0].content == "First."
        assert retrieved[1].content == "Second."
        assert retrieved[2].content == "Third."

    def test_get_nonexistent_atom(self, atom_store):
        result = atom_store.get("nonexistent-id")
        assert result is None

    def test_delete_by_chunk_ids(self, atom_store):
        atoms = [
            Atom(content="A", chunk_id="c1", index=0),
            Atom(content="B", chunk_id="c1", index=1),
            Atom(content="C", chunk_id="c2", index=0),
        ]
        atom_store.put_many(atoms)

        atom_store.delete_by_chunk_ids(["c1"])

        # c1 atoms should be gone
        assert atom_store.get_by_chunk("c1") == []
        # c2 atoms should remain
        remaining = atom_store.get_by_chunk("c2")
        assert len(remaining) == 1
        assert remaining[0].content == "C"

    def test_list_chunk_ids(self, atom_store):
        atoms = [
            Atom(content="A", chunk_id="c1", index=0),
            Atom(content="B", chunk_id="c2", index=0),
            Atom(content="C", chunk_id="c1", index=1),
        ]
        atom_store.put_many(atoms)

        chunk_ids = atom_store.list_chunk_ids()
        assert chunk_ids == {"c1", "c2"}

    def test_list_chunk_ids_empty(self, atom_store):
        chunk_ids = atom_store.list_chunk_ids()
        assert chunk_ids == set()

    def test_put_overwrites_existing(self, atom_store):
        atom = Atom(id="atom-1", content="Original", chunk_id="c1", index=0)
        atom_store.put(atom)

        updated = Atom(id="atom-1", content="Updated", chunk_id="c1", index=0)
        atom_store.put(updated)

        retrieved = atom_store.get("atom-1")
        assert retrieved.content == "Updated"

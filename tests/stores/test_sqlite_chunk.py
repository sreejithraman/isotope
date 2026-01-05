# tests/stores/test_sqlite.py
"""Tests for SQLite chunk store."""

import os
import tempfile

import pytest

from isotope.models import Chunk
from isotope.stores.base import ChunkStore
from isotope.stores.sqlite_chunk import SQLiteChunkStore


@pytest.fixture
def temp_db():
    """Create a temporary database file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield os.path.join(tmpdir, "test.db")


@pytest.fixture
def chunk_store(temp_db):
    """Create a SQLiteChunkStore instance."""
    return SQLiteChunkStore(temp_db)


class TestSQLiteChunkStore:
    def test_is_chunkstore(self, chunk_store):
        assert isinstance(chunk_store, ChunkStore)

    def test_put_and_get(self, chunk_store):
        chunk = Chunk(content="Hello world", source="test.md")
        chunk_store.put(chunk)

        retrieved = chunk_store.get(chunk.id)
        assert retrieved is not None
        assert retrieved.id == chunk.id
        assert retrieved.content == "Hello world"
        assert retrieved.source == "test.md"

    def test_get_nonexistent(self, chunk_store):
        result = chunk_store.get("nonexistent-id")
        assert result is None

    def test_get_many(self, chunk_store):
        chunks = [
            Chunk(content="One", source="a.md"),
            Chunk(content="Two", source="b.md"),
            Chunk(content="Three", source="c.md"),
        ]
        for c in chunks:
            chunk_store.put(c)

        retrieved = chunk_store.get_many([chunks[0].id, chunks[2].id])
        assert len(retrieved) == 2
        contents = {c.content for c in retrieved}
        assert contents == {"One", "Three"}

    def test_get_many_skips_missing(self, chunk_store):
        chunk = Chunk(content="Exists", source="x.md")
        chunk_store.put(chunk)

        retrieved = chunk_store.get_many([chunk.id, "missing-id"])
        assert len(retrieved) == 1
        assert retrieved[0].id == chunk.id

    def test_delete(self, chunk_store):
        chunk = Chunk(content="To delete", source="d.md")
        chunk_store.put(chunk)
        assert chunk_store.get(chunk.id) is not None

        chunk_store.delete(chunk.id)
        assert chunk_store.get(chunk.id) is None

    def test_list_sources(self, chunk_store):
        chunk_store.put(Chunk(content="A", source="file1.md"))
        chunk_store.put(Chunk(content="B", source="file2.md"))
        chunk_store.put(Chunk(content="C", source="file1.md"))

        sources = chunk_store.list_sources()
        assert set(sources) == {"file1.md", "file2.md"}

    def test_get_by_source(self, chunk_store):
        chunk_store.put(Chunk(content="A", source="target.md"))
        chunk_store.put(Chunk(content="B", source="other.md"))
        chunk_store.put(Chunk(content="C", source="target.md"))

        results = chunk_store.get_by_source("target.md")
        assert len(results) == 2
        assert all(c.source == "target.md" for c in results)

    def test_metadata_preserved(self, chunk_store):
        chunk = Chunk(
            content="PDF content",
            source="doc.pdf",
            metadata={"page": 5, "type": "pdf", "author": "Test"},
        )
        chunk_store.put(chunk)

        retrieved = chunk_store.get(chunk.id)
        assert retrieved.metadata["page"] == 5
        assert retrieved.metadata["type"] == "pdf"
        assert retrieved.metadata["author"] == "Test"

    def test_put_overwrites_existing(self, chunk_store):
        chunk = Chunk(id="same-id", content="Original", source="x.md")
        chunk_store.put(chunk)

        updated = Chunk(id="same-id", content="Updated", source="x.md")
        chunk_store.put(updated)

        retrieved = chunk_store.get("same-id")
        assert retrieved.content == "Updated"

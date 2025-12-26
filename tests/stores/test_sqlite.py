# tests/stores/test_sqlite.py
"""Tests for SQLite document store."""

import pytest
import tempfile
import os
from pathlib import Path

from isotopedb.stores.sqlite import SQLiteDocStore
from isotopedb.stores.base import DocStore
from isotopedb.models import Chunk


@pytest.fixture
def temp_db():
    """Create a temporary database file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield os.path.join(tmpdir, "test.db")


@pytest.fixture
def doc_store(temp_db):
    """Create a SQLiteDocStore instance."""
    return SQLiteDocStore(temp_db)


class TestSQLiteDocStore:
    def test_is_docstore(self, doc_store):
        assert isinstance(doc_store, DocStore)

    def test_put_and_get(self, doc_store):
        chunk = Chunk(content="Hello world", source="test.md")
        doc_store.put(chunk)

        retrieved = doc_store.get(chunk.id)
        assert retrieved is not None
        assert retrieved.id == chunk.id
        assert retrieved.content == "Hello world"
        assert retrieved.source == "test.md"

    def test_get_nonexistent(self, doc_store):
        result = doc_store.get("nonexistent-id")
        assert result is None

    def test_get_many(self, doc_store):
        chunks = [
            Chunk(content="One", source="a.md"),
            Chunk(content="Two", source="b.md"),
            Chunk(content="Three", source="c.md"),
        ]
        for c in chunks:
            doc_store.put(c)

        retrieved = doc_store.get_many([chunks[0].id, chunks[2].id])
        assert len(retrieved) == 2
        contents = {c.content for c in retrieved}
        assert contents == {"One", "Three"}

    def test_get_many_skips_missing(self, doc_store):
        chunk = Chunk(content="Exists", source="x.md")
        doc_store.put(chunk)

        retrieved = doc_store.get_many([chunk.id, "missing-id"])
        assert len(retrieved) == 1
        assert retrieved[0].id == chunk.id

    def test_delete(self, doc_store):
        chunk = Chunk(content="To delete", source="d.md")
        doc_store.put(chunk)
        assert doc_store.get(chunk.id) is not None

        doc_store.delete(chunk.id)
        assert doc_store.get(chunk.id) is None

    def test_list_sources(self, doc_store):
        doc_store.put(Chunk(content="A", source="file1.md"))
        doc_store.put(Chunk(content="B", source="file2.md"))
        doc_store.put(Chunk(content="C", source="file1.md"))

        sources = doc_store.list_sources()
        assert set(sources) == {"file1.md", "file2.md"}

    def test_get_by_source(self, doc_store):
        doc_store.put(Chunk(content="A", source="target.md"))
        doc_store.put(Chunk(content="B", source="other.md"))
        doc_store.put(Chunk(content="C", source="target.md"))

        results = doc_store.get_by_source("target.md")
        assert len(results) == 2
        assert all(c.source == "target.md" for c in results)

    def test_metadata_preserved(self, doc_store):
        chunk = Chunk(
            content="PDF content",
            source="doc.pdf",
            metadata={"page": 5, "type": "pdf", "author": "Test"},
        )
        doc_store.put(chunk)

        retrieved = doc_store.get(chunk.id)
        assert retrieved.metadata["page"] == 5
        assert retrieved.metadata["type"] == "pdf"
        assert retrieved.metadata["author"] == "Test"

    def test_put_overwrites_existing(self, doc_store):
        chunk = Chunk(id="same-id", content="Original", source="x.md")
        doc_store.put(chunk)

        updated = Chunk(id="same-id", content="Updated", source="x.md")
        doc_store.put(updated)

        retrieved = doc_store.get("same-id")
        assert retrieved.content == "Updated"

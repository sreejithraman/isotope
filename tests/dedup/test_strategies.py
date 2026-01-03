# tests/dedup/test_strategies.py
"""Tests for deduplication strategies."""

import os
import tempfile

import pytest

from isotopedb.dedup.base import Deduplicator
from isotopedb.dedup.strategies import NoDedup, SourceAwareDedup
from isotopedb.models import Chunk
from isotopedb.stores.sqlite import SQLiteDocStore


@pytest.fixture
def temp_db():
    """Create a temporary database file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield os.path.join(tmpdir, "test.db")


@pytest.fixture
def doc_store(temp_db):
    """Create a SQLiteDocStore instance."""
    return SQLiteDocStore(temp_db)


class TestNoDedup:
    def test_is_deduplicator(self):
        assert isinstance(NoDedup(), Deduplicator)

    def test_returns_empty_list(self, doc_store):
        # Add some existing chunks
        existing = Chunk(content="Existing content", source="doc.md")
        doc_store.put(existing)

        # New chunks from same source
        new_chunks = [Chunk(content="New content", source="doc.md")]

        dedup = NoDedup()
        to_remove = dedup.get_chunks_to_remove(new_chunks, doc_store)

        assert to_remove == []

    def test_empty_new_chunks(self, doc_store):
        dedup = NoDedup()
        to_remove = dedup.get_chunks_to_remove([], doc_store)
        assert to_remove == []


class TestSourceAwareDedup:
    def test_is_deduplicator(self):
        assert isinstance(SourceAwareDedup(), Deduplicator)

    def test_removes_chunks_from_same_source(self, doc_store):
        # Add existing chunks from doc.md
        existing1 = Chunk(content="Old content 1", source="doc.md")
        existing2 = Chunk(content="Old content 2", source="doc.md")
        doc_store.put(existing1)
        doc_store.put(existing2)

        # New chunks from same source
        new_chunks = [Chunk(content="New content", source="doc.md")]

        dedup = SourceAwareDedup()
        to_remove = dedup.get_chunks_to_remove(new_chunks, doc_store)

        assert set(to_remove) == {existing1.id, existing2.id}

    def test_does_not_remove_chunks_from_other_sources(self, doc_store):
        # Add chunks from different sources
        chunk_a = Chunk(content="Content A", source="a.md")
        chunk_b = Chunk(content="Content B", source="b.md")
        doc_store.put(chunk_a)
        doc_store.put(chunk_b)

        # New chunks only from a.md
        new_chunks = [Chunk(content="New A", source="a.md")]

        dedup = SourceAwareDedup()
        to_remove = dedup.get_chunks_to_remove(new_chunks, doc_store)

        # Should only remove a.md chunks
        assert to_remove == [chunk_a.id]

    def test_handles_multiple_sources_in_new_chunks(self, doc_store):
        # Add chunks from multiple sources
        chunk_a = Chunk(content="Content A", source="a.md")
        chunk_b = Chunk(content="Content B", source="b.md")
        chunk_c = Chunk(content="Content C", source="c.md")
        doc_store.put(chunk_a)
        doc_store.put(chunk_b)
        doc_store.put(chunk_c)

        # New chunks from a.md and b.md
        new_chunks = [
            Chunk(content="New A", source="a.md"),
            Chunk(content="New B", source="b.md"),
        ]

        dedup = SourceAwareDedup()
        to_remove = dedup.get_chunks_to_remove(new_chunks, doc_store)

        # Should remove a.md and b.md chunks, not c.md
        assert set(to_remove) == {chunk_a.id, chunk_b.id}

    def test_empty_new_chunks(self, doc_store):
        # Add some existing chunks
        chunk = Chunk(content="Content", source="doc.md")
        doc_store.put(chunk)

        dedup = SourceAwareDedup()
        to_remove = dedup.get_chunks_to_remove([], doc_store)

        assert to_remove == []

    def test_no_existing_chunks_for_source(self, doc_store):
        # New chunks from a source with no existing data
        new_chunks = [Chunk(content="New content", source="new.md")]

        dedup = SourceAwareDedup()
        to_remove = dedup.get_chunks_to_remove(new_chunks, doc_store)

        assert to_remove == []

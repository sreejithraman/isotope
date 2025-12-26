# tests/stores/test_chroma.py
"""Tests for ChromaDB vector store."""

import pytest
import tempfile

from isotopedb.stores.chroma import ChromaVectorStore
from isotopedb.stores.base import VectorStore
from isotopedb.models import Question


@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def vector_store(temp_dir):
    """Create a ChromaVectorStore instance."""
    return ChromaVectorStore(temp_dir)


class TestChromaVectorStore:
    def test_is_vectorstore(self, vector_store):
        assert isinstance(vector_store, VectorStore)

    def test_add_and_search(self, vector_store):
        # Create questions with embeddings
        q1 = Question(
            text="What is Python?",
            chunk_id="chunk-1",
            embedding=[1.0, 0.0, 0.0],
        )
        q2 = Question(
            text="What is JavaScript?",
            chunk_id="chunk-2",
            embedding=[0.0, 1.0, 0.0],
        )
        vector_store.add([q1, q2])

        # Search with embedding similar to q1
        results = vector_store.search([0.9, 0.1, 0.0], k=2)
        assert len(results) == 2
        # First result should be closest to search vector
        assert results[0][0].text == "What is Python?"
        assert results[0][1] > results[1][1]  # Higher score = closer match

    def test_search_returns_question_objects(self, vector_store):
        q = Question(
            text="Test question?",
            chunk_id="c1",
            atom_id="a1",
            embedding=[1.0, 0.0, 0.0],
        )
        vector_store.add([q])

        results = vector_store.search([1.0, 0.0, 0.0], k=1)
        assert len(results) == 1
        question, score = results[0]
        assert question.text == "Test question?"
        assert question.chunk_id == "c1"
        assert question.atom_id == "a1"

    def test_delete_by_chunk_ids(self, vector_store):
        questions = [
            Question(text="Q1", chunk_id="c1", embedding=[1.0, 0.0, 0.0]),
            Question(text="Q2", chunk_id="c1", embedding=[0.0, 1.0, 0.0]),
            Question(text="Q3", chunk_id="c2", embedding=[0.0, 0.0, 1.0]),
        ]
        vector_store.add(questions)

        vector_store.delete_by_chunk_ids(["c1"])

        # Only c2 should remain
        results = vector_store.search([0.0, 0.0, 1.0], k=10)
        assert len(results) == 1
        assert results[0][0].chunk_id == "c2"

    def test_list_chunk_ids(self, vector_store):
        questions = [
            Question(text="Q1", chunk_id="c1", embedding=[1.0, 0.0, 0.0]),
            Question(text="Q2", chunk_id="c2", embedding=[0.0, 1.0, 0.0]),
            Question(text="Q3", chunk_id="c1", embedding=[0.0, 0.0, 1.0]),
        ]
        vector_store.add(questions)

        chunk_ids = vector_store.list_chunk_ids()
        assert chunk_ids == {"c1", "c2"}

    def test_search_empty_store(self, vector_store):
        results = vector_store.search([1.0, 0.0, 0.0], k=5)
        assert results == []

    def test_add_requires_embeddings(self, vector_store):
        q = Question(text="No embedding", chunk_id="c1")
        with pytest.raises(ValueError, match="embedding"):
            vector_store.add([q])

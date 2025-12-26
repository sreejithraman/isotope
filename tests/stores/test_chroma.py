# tests/stores/test_chroma.py
"""Tests for ChromaDB vector store."""

import pytest
import tempfile

from isotopedb.stores.chroma import ChromaVectorStore
from isotopedb.stores.base import VectorStore
from isotopedb.models import EmbeddedQuestion, Question


@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def vector_store(temp_dir):
    """Create a ChromaVectorStore instance."""
    return ChromaVectorStore(temp_dir)


def make_embedded(text: str, chunk_id: str, embedding: list[float], atom_id: str | None = None) -> EmbeddedQuestion:
    """Helper to create EmbeddedQuestion."""
    return EmbeddedQuestion(
        question=Question(text=text, chunk_id=chunk_id, atom_id=atom_id),
        embedding=embedding,
    )


class TestChromaVectorStore:
    def test_is_vectorstore(self, vector_store):
        assert isinstance(vector_store, VectorStore)

    def test_add_and_search(self, vector_store):
        # Create questions with embeddings
        eq1 = make_embedded("What is Python?", "chunk-1", [1.0, 0.0, 0.0])
        eq2 = make_embedded("What is JavaScript?", "chunk-2", [0.0, 1.0, 0.0])
        vector_store.add([eq1, eq2])

        # Search with embedding similar to q1
        results = vector_store.search([0.9, 0.1, 0.0], k=2)
        assert len(results) == 2
        # First result should be closest to search vector
        assert results[0][0].text == "What is Python?"
        assert results[0][1] > results[1][1]  # Higher score = closer match

    def test_search_returns_question_objects(self, vector_store):
        eq = make_embedded("Test question?", "c1", [1.0, 0.0, 0.0], atom_id="a1")
        vector_store.add([eq])

        results = vector_store.search([1.0, 0.0, 0.0], k=1)
        assert len(results) == 1
        question, score = results[0]
        assert question.text == "Test question?"
        assert question.chunk_id == "c1"
        assert question.atom_id == "a1"

    def test_delete_by_chunk_ids(self, vector_store):
        questions = [
            make_embedded("Q1", "c1", [1.0, 0.0, 0.0]),
            make_embedded("Q2", "c1", [0.0, 1.0, 0.0]),
            make_embedded("Q3", "c2", [0.0, 0.0, 1.0]),
        ]
        vector_store.add(questions)

        vector_store.delete_by_chunk_ids(["c1"])

        # Only c2 should remain
        results = vector_store.search([0.0, 0.0, 1.0], k=10)
        assert len(results) == 1
        assert results[0][0].chunk_id == "c2"

    def test_list_chunk_ids(self, vector_store):
        questions = [
            make_embedded("Q1", "c1", [1.0, 0.0, 0.0]),
            make_embedded("Q2", "c2", [0.0, 1.0, 0.0]),
            make_embedded("Q3", "c1", [0.0, 0.0, 1.0]),
        ]
        vector_store.add(questions)

        chunk_ids = vector_store.list_chunk_ids()
        assert chunk_ids == {"c1", "c2"}

    def test_search_empty_store(self, vector_store):
        results = vector_store.search([1.0, 0.0, 0.0], k=5)
        assert results == []

# tests/stores/test_chroma.py
"""Tests for ChromaDB embedded question store."""

import tempfile

import pytest

from isotope.models import EmbeddedQuestion, Question
from isotope.stores.base import ChunkEmbeddingStore, EmbeddedQuestionStore
from isotope.stores.chroma import ChromaChunkEmbeddingStore, ChromaEmbeddedQuestionStore


@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def embedded_question_store(temp_dir):
    """Create a ChromaEmbeddedQuestionStore instance."""
    return ChromaEmbeddedQuestionStore(temp_dir)


@pytest.fixture
def chunk_embedding_store(temp_dir):
    """Create a ChromaChunkEmbeddingStore instance."""
    return ChromaChunkEmbeddingStore(temp_dir)


def make_embedded(
    text: str, chunk_id: str, embedding: list[float], atom_id: str = "a1"
) -> EmbeddedQuestion:
    """Helper to create EmbeddedQuestion."""
    return EmbeddedQuestion(
        question=Question(text=text, chunk_id=chunk_id, atom_id=atom_id),
        embedding=embedding,
    )


class TestChromaEmbeddedQuestionStore:
    def test_is_embedded_question_store(self, embedded_question_store):
        assert isinstance(embedded_question_store, EmbeddedQuestionStore)

    def test_add_and_search(self, embedded_question_store):
        # Create questions with embeddings
        eq1 = make_embedded("What is Python?", "chunk-1", [1.0, 0.0, 0.0])
        eq2 = make_embedded("What is JavaScript?", "chunk-2", [0.0, 1.0, 0.0])
        embedded_question_store.add([eq1, eq2])

        # Search with embedding similar to q1
        results = embedded_question_store.search([0.9, 0.1, 0.0], k=2)
        assert len(results) == 2
        # First result should be closest to search vector
        assert results[0][0].text == "What is Python?"
        assert results[0][1] > results[1][1]  # Higher score = closer match

    def test_search_returns_question_objects(self, embedded_question_store):
        eq = make_embedded("Test question?", "c1", [1.0, 0.0, 0.0], atom_id="a1")
        embedded_question_store.add([eq])

        results = embedded_question_store.search([1.0, 0.0, 0.0], k=1)
        assert len(results) == 1
        question, score = results[0]
        assert question.text == "Test question?"
        assert question.chunk_id == "c1"
        assert question.atom_id == "a1"

    def test_delete_by_chunk_ids(self, embedded_question_store):
        questions = [
            make_embedded("Q1", "c1", [1.0, 0.0, 0.0]),
            make_embedded("Q2", "c1", [0.0, 1.0, 0.0]),
            make_embedded("Q3", "c2", [0.0, 0.0, 1.0]),
        ]
        embedded_question_store.add(questions)

        embedded_question_store.delete_by_chunk_ids(["c1"])

        # Only c2 should remain
        results = embedded_question_store.search([0.0, 0.0, 1.0], k=10)
        assert len(results) == 1
        assert results[0][0].chunk_id == "c2"

    def test_list_chunk_ids(self, embedded_question_store):
        questions = [
            make_embedded("Q1", "c1", [1.0, 0.0, 0.0]),
            make_embedded("Q2", "c2", [0.0, 1.0, 0.0]),
            make_embedded("Q3", "c1", [0.0, 0.0, 1.0]),
        ]
        embedded_question_store.add(questions)

        chunk_ids = embedded_question_store.list_chunk_ids()
        assert chunk_ids == {"c1", "c2"}

    def test_search_empty_store(self, embedded_question_store):
        results = embedded_question_store.search([1.0, 0.0, 0.0], k=5)
        assert results == []


class TestChromaChunkEmbeddingStore:
    def test_is_chunk_embedding_store(self, chunk_embedding_store):
        assert isinstance(chunk_embedding_store, ChunkEmbeddingStore)

    def test_add_and_search(self, chunk_embedding_store):
        chunk_embedding_store.add(
            chunk_ids=["c1", "c2"],
            embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        )
        results = chunk_embedding_store.search([0.9, 0.1, 0.0], k=2)
        assert len(results) == 2
        # First result should be closest
        assert results[0][0] == "c1"
        assert results[0][1] > results[1][1]

    def test_search_returns_chunk_id_score_tuples(self, chunk_embedding_store):
        chunk_embedding_store.add(chunk_ids=["c1"], embeddings=[[1.0, 0.0, 0.0]])
        results = chunk_embedding_store.search([1.0, 0.0, 0.0], k=1)
        assert len(results) == 1
        chunk_id, score = results[0]
        assert chunk_id == "c1"
        assert isinstance(score, float)

    def test_delete_by_chunk_ids(self, chunk_embedding_store):
        chunk_embedding_store.add(
            chunk_ids=["c1", "c2", "c3"],
            embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        )
        chunk_embedding_store.delete_by_chunk_ids(["c1", "c2"])
        results = chunk_embedding_store.search([0.0, 0.0, 1.0], k=10)
        assert len(results) == 1
        assert results[0][0] == "c3"

    def test_count(self, chunk_embedding_store):
        assert chunk_embedding_store.count() == 0
        chunk_embedding_store.add(
            chunk_ids=["c1", "c2"],
            embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        )
        assert chunk_embedding_store.count() == 2

    def test_search_empty_store(self, chunk_embedding_store):
        results = chunk_embedding_store.search([1.0, 0.0, 0.0], k=5)
        assert results == []

    def test_delete_empty_list(self, chunk_embedding_store):
        """delete_by_chunk_ids with empty list should not error."""
        chunk_embedding_store.delete_by_chunk_ids([])

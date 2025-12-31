"""Tests for the Retriever pipeline."""

import pytest
import tempfile
import os
from unittest.mock import MagicMock, patch

from isotopedb.retriever import Retriever
from isotopedb.stores import ChromaVectorStore, SQLiteDocStore
from isotopedb.embedder import Embedder
from isotopedb.models import Chunk, Question, EmbeddedQuestion, SearchResult


@pytest.fixture
def temp_dir():
    """Create a temporary directory for stores."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def stores(temp_dir):
    """Create store instances."""
    return {
        "vector_store": ChromaVectorStore(os.path.join(temp_dir, "chroma")),
        "doc_store": SQLiteDocStore(os.path.join(temp_dir, "docs.db")),
    }


class TestRetrieverInit:
    def test_init_with_required_components(self, stores):
        retriever = Retriever(
            vector_store=stores["vector_store"],
            doc_store=stores["doc_store"],
            embedder=Embedder(model="gemini/text-embedding-004"),
        )
        assert retriever is not None

    def test_init_with_default_k(self, stores):
        retriever = Retriever(
            vector_store=stores["vector_store"],
            doc_store=stores["doc_store"],
            embedder=Embedder(model="gemini/text-embedding-004"),
        )
        assert retriever.default_k == 5

    def test_init_with_custom_k(self, stores):
        retriever = Retriever(
            vector_store=stores["vector_store"],
            doc_store=stores["doc_store"],
            embedder=Embedder(model="gemini/text-embedding-004"),
            default_k=10,
        )
        assert retriever.default_k == 10


class TestRetrieverSearch:
    @patch("isotopedb.embedder.litellm_embedder.litellm.embedding")
    def test_search_returns_results(self, mock_embedding, stores):
        mock_embedding.return_value = MagicMock(
            data=[{"embedding": [1.0, 0.0, 0.0], "index": 0}]
        )

        # Setup: add a chunk and question
        chunk = Chunk(content="Python is a programming language.", source="test.md")
        stores["doc_store"].put(chunk)

        question = Question(text="What is Python?", chunk_id=chunk.id)
        embedded_q = EmbeddedQuestion(question=question, embedding=[1.0, 0.0, 0.0])
        stores["vector_store"].add([embedded_q])

        retriever = Retriever(
            vector_store=stores["vector_store"],
            doc_store=stores["doc_store"],
            embedder=Embedder(model="test-model"),
        )

        results = retriever.search("What is Python?")

        assert len(results) == 1
        assert isinstance(results[0], SearchResult)
        assert results[0].chunk.id == chunk.id
        assert results[0].question.text == "What is Python?"
        assert results[0].score > 0

    @patch("isotopedb.embedder.litellm_embedder.litellm.embedding")
    def test_search_respects_k(self, mock_embedding, stores):
        mock_embedding.return_value = MagicMock(
            data=[{"embedding": [1.0, 0.0, 0.0], "index": 0}]
        )

        # Add multiple questions
        chunk = Chunk(content="Content", source="test.md")
        stores["doc_store"].put(chunk)

        for i in range(10):
            q = Question(text=f"Q{i}?", chunk_id=chunk.id)
            eq = EmbeddedQuestion(question=q, embedding=[1.0, 0.0, 0.0])
            stores["vector_store"].add([eq])

        retriever = Retriever(
            vector_store=stores["vector_store"],
            doc_store=stores["doc_store"],
            embedder=Embedder(model="test-model"),
            default_k=3,
        )

        results = retriever.search("query")
        assert len(results) == 3

    @patch("isotopedb.embedder.litellm_embedder.litellm.embedding")
    def test_search_empty_store(self, mock_embedding, stores):
        mock_embedding.return_value = MagicMock(
            data=[{"embedding": [1.0, 0.0, 0.0], "index": 0}]
        )

        retriever = Retriever(
            vector_store=stores["vector_store"],
            doc_store=stores["doc_store"],
            embedder=Embedder(model="test-model"),
        )

        results = retriever.search("any query")
        assert results == []

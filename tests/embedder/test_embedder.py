# tests/embedder/test_embedder.py
"""Tests for the ClientEmbedder wrapper."""

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("litellm", reason="Tests require litellm package")

from isotope.embedder import ClientEmbedder, Embedder
from isotope.models import EmbeddedQuestion, Question
from isotope.providers.litellm import LiteLLMEmbeddingClient


def mock_embedding_response(embeddings: list[list[float]]):
    """Create a mock LiteLLM embedding response."""
    mock_response = MagicMock()
    mock_response.data = [{"index": i, "embedding": emb} for i, emb in enumerate(embeddings)]
    return mock_response


@pytest.fixture
def embedding_client():
    """Create a LiteLLMEmbeddingClient instance."""
    return LiteLLMEmbeddingClient()


@pytest.fixture
def embedder(embedding_client):
    """Create an ClientEmbedder instance."""
    return ClientEmbedder(embedding_client=embedding_client)


class TestClientEmbedder:
    def test_is_embedder(self, embedder):
        assert isinstance(embedder, Embedder)

    @patch("isotope.providers.litellm.client.litellm.embedding")
    def test_embed_text(self, mock_embedding, embedder):
        mock_embedding.return_value = mock_embedding_response([[0.1, 0.2, 0.3]])

        result = embedder.embed_text("Hello world")

        assert result == [0.1, 0.2, 0.3]
        mock_embedding.assert_called_once()

    @patch("isotope.providers.litellm.client.litellm.embedding")
    def test_embed_texts_batch(self, mock_embedding, embedder):
        mock_embedding.return_value = mock_embedding_response(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )

        result = embedder.embed_texts(["Text 1", "Text 2"])

        assert len(result) == 2
        assert result[0] == [1.0, 0.0]
        assert result[1] == [0.0, 1.0]

    @patch("isotope.providers.litellm.client.litellm.embedding")
    def test_embed_texts_empty(self, mock_embedding, embedder):
        result = embedder.embed_texts([])

        assert result == []
        mock_embedding.assert_not_called()

    @patch("isotope.providers.litellm.client.litellm.embedding")
    def test_embed_texts_preserves_order(self, mock_embedding, embedder):
        # Simulate out-of-order response (can happen with some APIs)
        mock_response = MagicMock()
        mock_response.data = [
            {"index": 1, "embedding": [0.0, 1.0]},
            {"index": 0, "embedding": [1.0, 0.0]},
        ]
        mock_embedding.return_value = mock_response

        result = embedder.embed_texts(["First", "Second"])

        # Should be sorted by index
        assert result[0] == [1.0, 0.0]  # index 0
        assert result[1] == [0.0, 1.0]  # index 1

    @patch("isotope.providers.litellm.client.litellm.embedding")
    def test_embed_question(self, mock_embedding, embedder):
        mock_embedding.return_value = mock_embedding_response([[0.5, 0.5]])

        question = Question(text="What is Python?", chunk_id="c1", atom_id="a1")
        result = embedder.embed_question(question)

        assert isinstance(result, EmbeddedQuestion)
        assert result.question == question
        assert result.embedding == [0.5, 0.5]

    @patch("isotope.providers.litellm.client.litellm.embedding")
    def test_embed_questions_batch(self, mock_embedding, embedder):
        mock_embedding.return_value = mock_embedding_response(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )

        questions = [
            Question(text="Q1?", chunk_id="c1", atom_id="a1"),
            Question(text="Q2?", chunk_id="c2", atom_id="a2"),
        ]
        result = embedder.embed_questions(questions)

        assert len(result) == 2
        assert all(isinstance(r, EmbeddedQuestion) for r in result)
        assert result[0].question == questions[0]
        assert result[1].question == questions[1]
        assert result[0].embedding == [1.0, 0.0]
        assert result[1].embedding == [0.0, 1.0]

    @patch("isotope.providers.litellm.client.litellm.embedding")
    def test_embed_questions_empty(self, mock_embedding, embedder):
        result = embedder.embed_questions([])

        assert result == []
        mock_embedding.assert_not_called()

    @patch("isotope.providers.litellm.client.litellm.embedding")
    def test_embed_questions_preserves_question_data(self, mock_embedding, embedder):
        mock_embedding.return_value = mock_embedding_response([[0.1, 0.2]])

        question = Question(
            text="What is X?",
            chunk_id="chunk-123",
            atom_id="atom-456",
        )
        result = embedder.embed_questions([question])

        assert result[0].question.text == "What is X?"
        assert result[0].question.chunk_id == "chunk-123"
        assert result[0].question.atom_id == "atom-456"
        assert result[0].question.id == question.id

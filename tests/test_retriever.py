"""Tests for the Retriever pipeline."""

from unittest.mock import MagicMock, patch

import pytest

from isotope.embedder import ClientEmbedder
from isotope.models import Atom, Chunk, EmbeddedQuestion, Question, SearchResult
from isotope.providers import LLMClient
from isotope.providers.litellm import LiteLLMEmbeddingClient
from isotope.retriever import Retriever
from isotope.stores.chroma import ChromaChunkEmbeddingStore


@pytest.fixture
def chunk_embedding_store(tmp_path):
    store = ChromaChunkEmbeddingStore(str(tmp_path / "chunk_embeddings"))
    yield store
    store.close()


def create_test_data(stores):
    """Helper to create chunk, atom, and question for testing."""
    chunk = Chunk(content="Python is a programming language.", source="test.md")
    stores["chunk_store"].put(chunk)

    atom = Atom(content="Python is a programming language.", chunk_id=chunk.id)
    stores["atom_store"].put(atom)

    question = Question(text="What is Python?", chunk_id=chunk.id, atom_id=atom.id)
    embedded_q = EmbeddedQuestion(question=question, embedding=[1.0, 0.0, 0.0])
    stores["embedded_question_store"].add([embedded_q])

    return chunk, atom, question


class TestRetrieverInit:
    def test_init_with_required_components(self, stores):
        retriever = Retriever(
            embedded_question_store=stores["embedded_question_store"],
            chunk_store=stores["chunk_store"],
            atom_store=stores["atom_store"],
            embedder=ClientEmbedder(embedding_client=LiteLLMEmbeddingClient()),
        )
        assert retriever is not None

    def test_init_with_default_k(self, stores):
        retriever = Retriever(
            embedded_question_store=stores["embedded_question_store"],
            chunk_store=stores["chunk_store"],
            atom_store=stores["atom_store"],
            embedder=ClientEmbedder(embedding_client=LiteLLMEmbeddingClient()),
        )
        assert retriever.default_k == 5

    def test_init_with_custom_k(self, stores):
        retriever = Retriever(
            embedded_question_store=stores["embedded_question_store"],
            chunk_store=stores["chunk_store"],
            atom_store=stores["atom_store"],
            embedder=ClientEmbedder(embedding_client=LiteLLMEmbeddingClient()),
            default_k=10,
        )
        assert retriever.default_k == 10


class TestRetrieverGetContext:
    @pytest.mark.mock_integration
    @patch("isotope.providers.litellm.client.litellm.embedding")
    def test_get_context_returns_results(self, mock_embedding, stores):
        mock_embedding.return_value = MagicMock(data=[{"embedding": [1.0, 0.0, 0.0], "index": 0}])

        chunk, atom, question = create_test_data(stores)

        retriever = Retriever(
            embedded_question_store=stores["embedded_question_store"],
            chunk_store=stores["chunk_store"],
            atom_store=stores["atom_store"],
            embedder=ClientEmbedder(embedding_client=LiteLLMEmbeddingClient()),
        )

        results = retriever.get_context("What is Python?")

        assert len(results) == 1
        assert isinstance(results[0], SearchResult)
        assert results[0].chunk.id == chunk.id
        assert results[0].atom.id == atom.id
        assert results[0].question.text == "What is Python?"
        assert results[0].score > 0

    @pytest.mark.mock_integration
    @patch("isotope.providers.litellm.client.litellm.embedding")
    def test_get_context_respects_k(self, mock_embedding, stores):
        mock_embedding.return_value = MagicMock(data=[{"embedding": [1.0, 0.0, 0.0], "index": 0}])

        # Add multiple questions
        chunk = Chunk(content="Content", source="test.md")
        stores["chunk_store"].put(chunk)

        atom = Atom(content="Content", chunk_id=chunk.id)
        stores["atom_store"].put(atom)

        for i in range(10):
            q = Question(text=f"Q{i}?", chunk_id=chunk.id, atom_id=atom.id)
            eq = EmbeddedQuestion(question=q, embedding=[1.0, 0.0, 0.0])
            stores["embedded_question_store"].add([eq])

        retriever = Retriever(
            embedded_question_store=stores["embedded_question_store"],
            chunk_store=stores["chunk_store"],
            atom_store=stores["atom_store"],
            embedder=ClientEmbedder(embedding_client=LiteLLMEmbeddingClient()),
            default_k=3,
        )

        results = retriever.get_context("query")
        assert len(results) == 3

    @pytest.mark.mock_integration
    @patch("isotope.providers.litellm.client.litellm.embedding")
    def test_get_context_empty_store(self, mock_embedding, stores):
        mock_embedding.return_value = MagicMock(data=[{"embedding": [1.0, 0.0, 0.0], "index": 0}])

        retriever = Retriever(
            embedded_question_store=stores["embedded_question_store"],
            chunk_store=stores["chunk_store"],
            atom_store=stores["atom_store"],
            embedder=ClientEmbedder(embedding_client=LiteLLMEmbeddingClient()),
        )

        results = retriever.get_context("any query")
        assert results == []

    @pytest.mark.mock_integration
    @patch("isotope.providers.litellm.client.litellm.embedding")
    def test_get_context_includes_atom(self, mock_embedding, stores):
        """Test that get_context includes atoms in results."""
        mock_embedding.return_value = MagicMock(data=[{"embedding": [1.0, 0.0, 0.0], "index": 0}])

        chunk, atom, question = create_test_data(stores)

        retriever = Retriever(
            embedded_question_store=stores["embedded_question_store"],
            chunk_store=stores["chunk_store"],
            atom_store=stores["atom_store"],
            embedder=ClientEmbedder(embedding_client=LiteLLMEmbeddingClient()),
        )

        results = retriever.get_context("What is Python?")

        assert len(results) == 1
        assert results[0].chunk.id == chunk.id
        assert results[0].atom is not None
        assert results[0].atom.id == atom.id
        assert results[0].atom.content == "Python is a programming language."


class TestRetrieverGetAnswer:
    @pytest.mark.mock_integration
    @patch("isotope.providers.litellm.client.litellm.embedding")
    def test_get_answer_returns_response_with_synthesis(self, mock_embedding, stores):
        mock_embedding.return_value = MagicMock(data=[{"embedding": [1.0, 0.0, 0.0], "index": 0}])

        chunk, atom, question = create_test_data(stores)

        # Create mock LLM client
        mock_llm_client = MagicMock(spec=LLMClient)
        mock_llm_client.complete.return_value = "Python is a programming language."

        retriever = Retriever(
            embedded_question_store=stores["embedded_question_store"],
            chunk_store=stores["chunk_store"],
            atom_store=stores["atom_store"],
            embedder=ClientEmbedder(embedding_client=LiteLLMEmbeddingClient()),
            llm_client=mock_llm_client,
        )

        from isotope.models import QueryResponse

        response = retriever.get_answer("What is Python?")

        assert isinstance(response, QueryResponse)
        assert response.query == "What is Python?"
        assert response.answer is not None  # Synthesizes when llm_client is set
        assert response.answer == "Python is a programming language."
        assert len(response.results) > 0
        mock_llm_client.complete.assert_called_once()

    @pytest.mark.mock_integration
    @patch("isotope.providers.litellm.client.litellm.embedding")
    def test_get_answer_no_synthesis_without_llm_client(self, mock_embedding, stores):
        """Test that get_answer() returns no answer when llm_client is not set."""
        mock_embedding.return_value = MagicMock(data=[{"embedding": [1.0, 0.0, 0.0], "index": 0}])

        chunk, atom, question = create_test_data(stores)

        retriever = Retriever(
            embedded_question_store=stores["embedded_question_store"],
            chunk_store=stores["chunk_store"],
            atom_store=stores["atom_store"],
            embedder=ClientEmbedder(embedding_client=LiteLLMEmbeddingClient()),
            # No llm_client - so no synthesis
        )

        from isotope.models import QueryResponse

        response = retriever.get_answer("What is Python?")

        assert isinstance(response, QueryResponse)
        assert response.answer is None  # No synthesis without llm_client
        assert len(response.results) > 0

    @pytest.mark.mock_integration
    @patch("isotope.providers.litellm.client.litellm.embedding")
    def test_get_answer_passes_temperature_to_llm_client(self, mock_embedding, stores):
        """Test that synthesis_temperature is passed to the LLM client."""
        mock_embedding.return_value = MagicMock(data=[{"embedding": [1.0, 0.0, 0.0], "index": 0}])

        chunk, atom, question = create_test_data(stores)

        mock_llm_client = MagicMock(spec=LLMClient)
        mock_llm_client.complete.return_value = "Answer"

        retriever = Retriever(
            embedded_question_store=stores["embedded_question_store"],
            chunk_store=stores["chunk_store"],
            atom_store=stores["atom_store"],
            embedder=ClientEmbedder(embedding_client=LiteLLMEmbeddingClient()),
            llm_client=mock_llm_client,
            synthesis_temperature=0.7,
        )

        retriever.get_answer("What is Python?")

        # Verify temperature was passed
        call_kwargs = mock_llm_client.complete.call_args.kwargs
        assert call_kwargs["temperature"] == 0.7

    @pytest.mark.mock_integration
    @patch("isotope.providers.litellm.client.litellm.embedding")
    def test_get_answer_no_results(self, mock_embedding, stores):
        mock_embedding.return_value = MagicMock(data=[{"embedding": [1.0, 0.0, 0.0], "index": 0}])

        retriever = Retriever(
            embedded_question_store=stores["embedded_question_store"],
            chunk_store=stores["chunk_store"],
            atom_store=stores["atom_store"],
            embedder=ClientEmbedder(embedding_client=LiteLLMEmbeddingClient()),
        )

        response = retriever.get_answer("anything")

        assert response.answer is None
        assert response.results == []


class TestRetrieverHybridFallback:
    @pytest.mark.mock_integration
    @patch("isotope.providers.litellm.client.litellm.embedding")
    def test_no_fallback_when_disabled(self, mock_embedding, stores, chunk_embedding_store):
        """threshold=0 disables hybrid fallback entirely."""
        mock_embedding.return_value = MagicMock(data=[{"embedding": [1.0, 0.0, 0.0], "index": 0}])
        chunk, atom, question = create_test_data(stores)

        chunk_embedding_store.add([chunk.id], [[1.0, 0.0, 0.0]])

        retriever = Retriever(
            embedded_question_store=stores["embedded_question_store"],
            chunk_store=stores["chunk_store"],
            atom_store=stores["atom_store"],
            embedder=ClientEmbedder(embedding_client=LiteLLMEmbeddingClient()),
            chunk_embedding_store=chunk_embedding_store,
            hybrid_confidence_threshold=0,
        )
        results = retriever.get_context("What is Python?")
        assert len(results) == 1
        assert results[0].question is not None

    @pytest.mark.mock_integration
    @patch("isotope.providers.litellm.client.litellm.embedding")
    def test_no_fallback_when_scores_above_threshold(
        self, mock_embedding, stores, chunk_embedding_store
    ):
        """No fallback when best question score >= threshold."""
        mock_embedding.return_value = MagicMock(data=[{"embedding": [1.0, 0.0, 0.0], "index": 0}])
        chunk, atom, question = create_test_data(stores)
        chunk_embedding_store.add([chunk.id], [[1.0, 0.0, 0.0]])

        retriever = Retriever(
            embedded_question_store=stores["embedded_question_store"],
            chunk_store=stores["chunk_store"],
            atom_store=stores["atom_store"],
            embedder=ClientEmbedder(embedding_client=LiteLLMEmbeddingClient()),
            chunk_embedding_store=chunk_embedding_store,
            hybrid_confidence_threshold=0.5,
        )
        results = retriever.get_context("What is Python?")
        assert all(r.question is not None for r in results)

    @pytest.mark.mock_integration
    @patch("isotope.providers.litellm.client.litellm.embedding")
    def test_fallback_when_scores_below_threshold(
        self, mock_embedding, stores, chunk_embedding_store
    ):
        """Fallback triggers when best question score < threshold."""
        mock_embedding.return_value = MagicMock(data=[{"embedding": [1.0, 0.0, 0.0], "index": 0}])

        chunk = Chunk(content="Python info.", source="test.md")
        stores["chunk_store"].put(chunk)
        atom = Atom(content="Python info.", chunk_id=chunk.id)
        stores["atom_store"].put(atom)
        q = Question(text="What is Python?", chunk_id=chunk.id, atom_id=atom.id)
        eq = EmbeddedQuestion(question=q, embedding=[0.0, 1.0, 0.0])
        stores["embedded_question_store"].add([eq])

        chunk2 = Chunk(content="Extra Python info.", source="test2.md")
        stores["chunk_store"].put(chunk2)
        chunk_embedding_store.add([chunk2.id], [[0.95, 0.05, 0.0]])

        retriever = Retriever(
            embedded_question_store=stores["embedded_question_store"],
            chunk_store=stores["chunk_store"],
            atom_store=stores["atom_store"],
            embedder=ClientEmbedder(embedding_client=LiteLLMEmbeddingClient()),
            chunk_embedding_store=chunk_embedding_store,
            hybrid_confidence_threshold=0.9,
        )
        results = retriever.get_context("What is Python?", k=5)
        has_question_result = any(r.question is not None for r in results)
        has_chunk_result = any(r.question is None for r in results)
        assert has_question_result
        assert has_chunk_result

    @pytest.mark.mock_integration
    @patch("isotope.providers.litellm.client.litellm.embedding")
    def test_fallback_fills_all_slots_when_no_question_results(
        self, mock_embedding, stores, chunk_embedding_store
    ):
        """When question store is empty, fallback fills all k slots."""
        mock_embedding.return_value = MagicMock(data=[{"embedding": [1.0, 0.0, 0.0], "index": 0}])

        chunk = Chunk(content="Python info.", source="test.md")
        stores["chunk_store"].put(chunk)
        chunk_embedding_store.add([chunk.id], [[1.0, 0.0, 0.0]])

        retriever = Retriever(
            embedded_question_store=stores["embedded_question_store"],
            chunk_store=stores["chunk_store"],
            atom_store=stores["atom_store"],
            embedder=ClientEmbedder(embedding_client=LiteLLMEmbeddingClient()),
            chunk_embedding_store=chunk_embedding_store,
            hybrid_confidence_threshold=0.7,
        )
        results = retriever.get_context("What is Python?", k=5)
        assert len(results) == 1
        assert results[0].question is None
        assert results[0].chunk.id == chunk.id

    @pytest.mark.mock_integration
    @patch("isotope.providers.litellm.client.litellm.embedding")
    def test_fallback_deduplicates_by_chunk_id(self, mock_embedding, stores, chunk_embedding_store):
        """Same chunk from both paths appears only once."""
        mock_embedding.return_value = MagicMock(data=[{"embedding": [1.0, 0.0, 0.0], "index": 0}])

        chunk = Chunk(content="Python info.", source="test.md")
        stores["chunk_store"].put(chunk)
        atom = Atom(content="Python info.", chunk_id=chunk.id)
        stores["atom_store"].put(atom)

        q = Question(text="What is Python?", chunk_id=chunk.id, atom_id=atom.id)
        eq = EmbeddedQuestion(question=q, embedding=[0.0, 1.0, 0.0])
        stores["embedded_question_store"].add([eq])

        chunk_embedding_store.add([chunk.id], [[1.0, 0.0, 0.0]])

        retriever = Retriever(
            embedded_question_store=stores["embedded_question_store"],
            chunk_store=stores["chunk_store"],
            atom_store=stores["atom_store"],
            embedder=ClientEmbedder(embedding_client=LiteLLMEmbeddingClient()),
            chunk_embedding_store=chunk_embedding_store,
            hybrid_confidence_threshold=0.9,
        )
        results = retriever.get_context("What is Python?", k=5)
        chunk_ids = [r.chunk.id for r in results]
        assert len(chunk_ids) == len(set(chunk_ids))

    @pytest.mark.mock_integration
    @patch("isotope.providers.litellm.client.litellm.embedding")
    def test_results_capped_at_k(self, mock_embedding, stores, chunk_embedding_store):
        """Combined results don't exceed k."""
        mock_embedding.return_value = MagicMock(data=[{"embedding": [1.0, 0.0, 0.0], "index": 0}])

        for i in range(10):
            chunk = Chunk(content=f"Chunk {i}", source=f"test{i}.md")
            stores["chunk_store"].put(chunk)
            atom = Atom(content=f"Chunk {i}", chunk_id=chunk.id)
            stores["atom_store"].put(atom)
            q = Question(text=f"Q{i}?", chunk_id=chunk.id, atom_id=atom.id)
            eq = EmbeddedQuestion(question=q, embedding=[0.0, 1.0, 0.0])
            stores["embedded_question_store"].add([eq])
            chunk_embedding_store.add([chunk.id], [[1.0, 0.0, 0.0]])

        retriever = Retriever(
            embedded_question_store=stores["embedded_question_store"],
            chunk_store=stores["chunk_store"],
            atom_store=stores["atom_store"],
            embedder=ClientEmbedder(embedding_client=LiteLLMEmbeddingClient()),
            chunk_embedding_store=chunk_embedding_store,
            hybrid_confidence_threshold=0.9,
            default_k=3,
        )
        results = retriever.get_context("query")
        assert len(results) <= 3

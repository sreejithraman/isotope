"""Tests for the Ingestor pipeline."""

from unittest.mock import MagicMock, patch

import pytest

from isotope.atomizer import SentenceAtomizer
from isotope.embedder import ClientEmbedder
from isotope.ingestor import Ingestor
from isotope.models import Chunk
from isotope.providers.litellm import LiteLLMClient, LiteLLMEmbeddingClient
from isotope.question_generator import ClientQuestionGenerator, DiversityFilter


class TestIngestorInit:
    def test_init_with_all_components(self, stores):
        ingestor = Ingestor(
            embedded_question_store=stores["embedded_question_store"],
            chunk_store=stores["chunk_store"],
            atom_store=stores["atom_store"],
            atomizer=SentenceAtomizer(),
            embedder=ClientEmbedder(embedding_client=LiteLLMEmbeddingClient()),
            question_generator=ClientQuestionGenerator(llm_client=LiteLLMClient()),
        )
        assert ingestor is not None

    def test_init_with_optional_diversity_filter(self, stores):
        ingestor = Ingestor(
            embedded_question_store=stores["embedded_question_store"],
            chunk_store=stores["chunk_store"],
            atom_store=stores["atom_store"],
            atomizer=SentenceAtomizer(),
            embedder=ClientEmbedder(embedding_client=LiteLLMEmbeddingClient()),
            question_generator=ClientQuestionGenerator(llm_client=LiteLLMClient()),
            diversity_filter=DiversityFilter(threshold=0.85),
        )
        assert ingestor.diversity_filter is not None


class TestIngestChunks:
    @pytest.mark.mock_integration
    @patch("isotope.providers.litellm.client.litellm.embedding")
    @patch("isotope.providers.litellm.client.litellm.completion")
    def test_ingest_single_chunk(self, mock_completion, mock_embedding, stores):
        # Setup mocks
        mock_completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='["Q1?", "Q2?"]'))]
        )
        # embed_texts accesses response.data as a list of dicts with "embedding" and "index" keys
        mock_embedding.return_value = MagicMock(
            data=[
                {"embedding": [0.1, 0.2, 0.3], "index": 0},
                {"embedding": [0.4, 0.5, 0.6], "index": 1},
            ]
        )

        ingestor = Ingestor(
            embedded_question_store=stores["embedded_question_store"],
            chunk_store=stores["chunk_store"],
            atom_store=stores["atom_store"],
            atomizer=SentenceAtomizer(),
            embedder=ClientEmbedder(embedding_client=LiteLLMEmbeddingClient()),
            question_generator=ClientQuestionGenerator(llm_client=LiteLLMClient()),
        )

        chunk = Chunk(content="Python is great.", source="test.md")
        result = ingestor.ingest_chunks([chunk])

        # Verify chunk was stored
        assert stores["chunk_store"].get(chunk.id) is not None

        # Verify atoms were created and stored
        atoms = stores["atom_store"].get_by_chunk(chunk.id)
        assert len(atoms) >= 1

        # Verify result contains stats
        assert "chunks" in result
        assert "atoms" in result
        assert "questions" in result
        assert result["chunks"] == 1

    @pytest.mark.mock_integration
    @patch("isotope.providers.litellm.client.litellm.embedding")
    @patch("isotope.providers.litellm.client.litellm.completion")
    def test_ingest_empty_list(self, mock_completion, mock_embedding, stores):
        ingestor = Ingestor(
            embedded_question_store=stores["embedded_question_store"],
            chunk_store=stores["chunk_store"],
            atom_store=stores["atom_store"],
            atomizer=SentenceAtomizer(),
            embedder=ClientEmbedder(embedding_client=LiteLLMEmbeddingClient()),
            question_generator=ClientQuestionGenerator(llm_client=LiteLLMClient()),
        )

        result = ingestor.ingest_chunks([])
        assert result["chunks"] == 0
        assert result["atoms"] == 0
        assert result["questions"] == 0


class TestIngestorProgress:
    @pytest.mark.mock_integration
    @patch("isotope.providers.litellm.client.litellm.embedding")
    @patch("isotope.providers.litellm.client.litellm.completion")
    def test_progress_callback_called(self, mock_completion, mock_embedding, stores):
        mock_completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='["Q1?"]'))]
        )

        # Return one embedding per input text
        def make_embeddings(*args, **kwargs):
            input_texts = kwargs.get("input", args[1] if len(args) > 1 else [])
            if isinstance(input_texts, str):
                input_texts = [input_texts]
            return MagicMock(
                data=[{"embedding": [0.1, 0.2, 0.3], "index": i} for i in range(len(input_texts))]
            )

        mock_embedding.side_effect = make_embeddings

        ingestor = Ingestor(
            embedded_question_store=stores["embedded_question_store"],
            chunk_store=stores["chunk_store"],
            atom_store=stores["atom_store"],
            atomizer=SentenceAtomizer(),
            embedder=ClientEmbedder(embedding_client=LiteLLMEmbeddingClient()),
            question_generator=ClientQuestionGenerator(llm_client=LiteLLMClient()),
        )

        progress_events = []

        def on_progress(event: str, current: int, total: int, message: str):
            progress_events.append((event, current, total, message))

        chunks = [
            Chunk(content="First sentence.", source="a.md"),
            Chunk(content="Second sentence.", source="b.md"),
        ]
        ingestor.ingest_chunks(chunks, on_progress=on_progress)

        # Should have progress events
        assert len(progress_events) > 0
        # Should include different phases
        phases = {e[0] for e in progress_events}
        assert "atomizing" in phases or "generating" in phases or "embedding" in phases

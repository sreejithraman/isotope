"""Tests for the Ingestor pipeline."""

from unittest.mock import MagicMock, patch

import pytest
from isotopedb.atomizer import SentenceAtomizer
from isotopedb.dedup import NoDedup
from isotopedb.generator import DiversityFilter
from isotopedb.ingestor import Ingestor
from isotopedb.litellm import LiteLLMEmbedder, LiteLLMQuestionGenerator
from isotopedb.models import Chunk


class TestIngestorInit:
    def test_init_with_all_components(self, stores):
        ingestor = Ingestor(
            vector_store=stores["vector_store"],
            doc_store=stores["doc_store"],
            atom_store=stores["atom_store"],
            atomizer=SentenceAtomizer(),
            embedder=LiteLLMEmbedder(model="gemini/text-embedding-004"),
            generator=LiteLLMQuestionGenerator(),
            deduplicator=NoDedup(),
        )
        assert ingestor is not None

    def test_init_with_optional_diversity_filter(self, stores):
        ingestor = Ingestor(
            vector_store=stores["vector_store"],
            doc_store=stores["doc_store"],
            atom_store=stores["atom_store"],
            atomizer=SentenceAtomizer(),
            embedder=LiteLLMEmbedder(model="gemini/text-embedding-004"),
            generator=LiteLLMQuestionGenerator(),
            deduplicator=NoDedup(),
            diversity_filter=DiversityFilter(threshold=0.85),
        )
        assert ingestor.diversity_filter is not None


class TestIngestChunks:
    @pytest.mark.mock_integration
    @patch("isotopedb.litellm.embedder.litellm.embedding")
    @patch("isotopedb.litellm.generator.litellm.completion")
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
            vector_store=stores["vector_store"],
            doc_store=stores["doc_store"],
            atom_store=stores["atom_store"],
            atomizer=SentenceAtomizer(),
            embedder=LiteLLMEmbedder(model="test-model"),
            generator=LiteLLMQuestionGenerator(),
            deduplicator=NoDedup(),
        )

        chunk = Chunk(content="Python is great.", source="test.md")
        result = ingestor.ingest_chunks([chunk])

        # Verify chunk was stored
        assert stores["doc_store"].get(chunk.id) is not None

        # Verify atoms were created and stored
        atoms = stores["atom_store"].get_by_chunk(chunk.id)
        assert len(atoms) >= 1

        # Verify result contains stats
        assert "chunks" in result
        assert "atoms" in result
        assert "questions" in result
        assert result["chunks"] == 1

    @pytest.mark.mock_integration
    @patch("isotopedb.litellm.embedder.litellm.embedding")
    @patch("isotopedb.litellm.generator.litellm.completion")
    def test_ingest_with_deduplication(self, mock_completion, mock_embedding, stores):
        mock_completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='["Q1?"]'))]
        )
        mock_embedding.return_value = MagicMock(data=[{"embedding": [0.1, 0.2, 0.3], "index": 0}])

        from isotopedb.dedup import SourceAwareDedup

        ingestor = Ingestor(
            vector_store=stores["vector_store"],
            doc_store=stores["doc_store"],
            atom_store=stores["atom_store"],
            atomizer=SentenceAtomizer(),
            embedder=LiteLLMEmbedder(model="test-model"),
            generator=LiteLLMQuestionGenerator(),
            deduplicator=SourceAwareDedup(),
        )

        # First ingestion
        chunk1 = Chunk(content="Original content.", source="test.md")
        ingestor.ingest_chunks([chunk1])

        # Re-ingest same source with new content
        chunk2 = Chunk(content="Updated content.", source="test.md")
        result = ingestor.ingest_chunks([chunk2])

        # Old chunk should be removed
        assert stores["doc_store"].get(chunk1.id) is None
        # New chunk should exist
        assert stores["doc_store"].get(chunk2.id) is not None
        assert result["chunks_removed"] == 1

    @pytest.mark.mock_integration
    @patch("isotopedb.litellm.embedder.litellm.embedding")
    @patch("isotopedb.litellm.generator.litellm.completion")
    def test_ingest_empty_list(self, mock_completion, mock_embedding, stores):
        ingestor = Ingestor(
            vector_store=stores["vector_store"],
            doc_store=stores["doc_store"],
            atom_store=stores["atom_store"],
            atomizer=SentenceAtomizer(),
            embedder=LiteLLMEmbedder(model="test-model"),
            generator=LiteLLMQuestionGenerator(),
            deduplicator=NoDedup(),
        )

        result = ingestor.ingest_chunks([])
        assert result["chunks"] == 0
        assert result["atoms"] == 0
        assert result["questions"] == 0


class TestIngestorProgress:
    @pytest.mark.mock_integration
    @patch("isotopedb.litellm.embedder.litellm.embedding")
    @patch("isotopedb.litellm.generator.litellm.completion")
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
            vector_store=stores["vector_store"],
            doc_store=stores["doc_store"],
            atom_store=stores["atom_store"],
            atomizer=SentenceAtomizer(),
            embedder=LiteLLMEmbedder(model="test-model"),
            generator=LiteLLMQuestionGenerator(),
            deduplicator=NoDedup(),
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

"""Tests for the Ingestor pipeline."""

import pytest
import tempfile
import os
from unittest.mock import MagicMock, patch

from isotopedb.ingestor import Ingestor
from isotopedb.stores import ChromaVectorStore, SQLiteDocStore, SQLiteAtomStore
from isotopedb.atomizer import SentenceAtomizer
from isotopedb.dedup import NoDedup
from isotopedb.embedder import Embedder
from isotopedb.generator import QuestionGenerator, DiversityFilter
from isotopedb.models import Chunk


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
        "atom_store": SQLiteAtomStore(os.path.join(temp_dir, "atoms.db")),
    }


class TestIngestorInit:
    def test_init_with_all_components(self, stores):
        ingestor = Ingestor(
            vector_store=stores["vector_store"],
            doc_store=stores["doc_store"],
            atom_store=stores["atom_store"],
            atomizer=SentenceAtomizer(),
            embedder=Embedder(model="gemini/text-embedding-004"),
            generator=QuestionGenerator(),
            deduplicator=NoDedup(),
        )
        assert ingestor is not None

    def test_init_with_optional_diversity_filter(self, stores):
        ingestor = Ingestor(
            vector_store=stores["vector_store"],
            doc_store=stores["doc_store"],
            atom_store=stores["atom_store"],
            atomizer=SentenceAtomizer(),
            embedder=Embedder(model="gemini/text-embedding-004"),
            generator=QuestionGenerator(),
            deduplicator=NoDedup(),
            diversity_filter=DiversityFilter(threshold=0.85),
        )
        assert ingestor.diversity_filter is not None


class TestIngestChunks:
    @patch("isotopedb.embedder.litellm_embedder.litellm.embedding")
    @patch("isotopedb.generator.question_generator.litellm.completion")
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
            embedder=Embedder(model="test-model"),
            generator=QuestionGenerator(),
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

    @patch("isotopedb.embedder.litellm_embedder.litellm.embedding")
    @patch("isotopedb.generator.question_generator.litellm.completion")
    def test_ingest_with_deduplication(self, mock_completion, mock_embedding, stores):
        mock_completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='["Q1?"]'))]
        )
        mock_embedding.return_value = MagicMock(
            data=[{"embedding": [0.1, 0.2, 0.3], "index": 0}]
        )

        from isotopedb.dedup import SourceAwareDedup

        ingestor = Ingestor(
            vector_store=stores["vector_store"],
            doc_store=stores["doc_store"],
            atom_store=stores["atom_store"],
            atomizer=SentenceAtomizer(),
            embedder=Embedder(model="test-model"),
            generator=QuestionGenerator(),
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

    @patch("isotopedb.embedder.litellm_embedder.litellm.embedding")
    @patch("isotopedb.generator.question_generator.litellm.completion")
    def test_ingest_empty_list(self, mock_completion, mock_embedding, stores):
        ingestor = Ingestor(
            vector_store=stores["vector_store"],
            doc_store=stores["doc_store"],
            atom_store=stores["atom_store"],
            atomizer=SentenceAtomizer(),
            embedder=Embedder(model="test-model"),
            generator=QuestionGenerator(),
            deduplicator=NoDedup(),
        )

        result = ingestor.ingest_chunks([])
        assert result["chunks"] == 0
        assert result["atoms"] == 0
        assert result["questions"] == 0

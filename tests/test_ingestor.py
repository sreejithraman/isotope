"""Tests for the Ingestor pipeline."""

import pytest
import tempfile
import os

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

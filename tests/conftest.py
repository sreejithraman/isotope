"""Shared pytest fixtures."""

import os
import tempfile

import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for stores."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def stores(temp_dir):
    """Create store instances for testing (requires chromadb)."""
    pytest.importorskip("chromadb", reason="This fixture requires chromadb")
    from isotopedb.stores import ChromaVectorStore, SQLiteAtomStore, SQLiteDocStore

    return {
        "vector_store": ChromaVectorStore(os.path.join(temp_dir, "chroma")),
        "doc_store": SQLiteDocStore(os.path.join(temp_dir, "docs.db")),
        "atom_store": SQLiteAtomStore(os.path.join(temp_dir, "atoms.db")),
    }

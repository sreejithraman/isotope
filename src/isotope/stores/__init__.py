# src/isotope/stores/__init__.py
"""Storage abstractions for Isotope."""

from isotope.stores.base import (
    AtomStore,
    ChunkEmbeddingStore,
    ChunkStore,
    EmbeddedQuestionStore,
    SourceRegistry,
)
from isotope.stores.source_registry import SQLiteSourceRegistry
from isotope.stores.sqlite_atom import SQLiteAtomStore
from isotope.stores.sqlite_chunk import SQLiteChunkStore

try:
    from isotope.stores.chroma import ChromaChunkEmbeddingStore, ChromaEmbeddedQuestionStore
except ImportError:
    from isotope._optional import _create_missing_dependency_class

    ChromaChunkEmbeddingStore = _create_missing_dependency_class(  # type: ignore[misc,assignment]
        "ChromaChunkEmbeddingStore", "chroma"
    )
    ChromaEmbeddedQuestionStore = _create_missing_dependency_class(  # type: ignore[misc,assignment]
        "ChromaEmbeddedQuestionStore", "chroma"
    )

__all__ = [
    "EmbeddedQuestionStore",
    "ChunkEmbeddingStore",
    "ChunkStore",
    "AtomStore",
    "SourceRegistry",
    "SQLiteChunkStore",
    "SQLiteAtomStore",
    "SQLiteSourceRegistry",
    "ChromaChunkEmbeddingStore",
    "ChromaEmbeddedQuestionStore",
]

# src/isotopedb/stores/__init__.py
"""Storage abstractions for Isotope."""

from isotopedb.stores.base import AtomStore, ChunkStore, VectorStore
from isotopedb.stores.source_registry import SourceRegistry, SQLiteSourceRegistry
from isotopedb.stores.sqlite import SQLiteChunkStore
from isotopedb.stores.sqlite_atom import SQLiteAtomStore

try:
    from isotopedb.stores.chroma import ChromaVectorStore
except ImportError:
    from isotopedb._optional import _create_missing_dependency_class

    ChromaVectorStore = _create_missing_dependency_class(  # type: ignore[misc,assignment]
        "ChromaVectorStore", "chromadb"
    )

__all__ = [
    "VectorStore",
    "ChunkStore",
    "AtomStore",
    "SourceRegistry",
    "SQLiteChunkStore",
    "SQLiteAtomStore",
    "SQLiteSourceRegistry",
    "ChromaVectorStore",
]

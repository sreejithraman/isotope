# src/isotopedb/stores/__init__.py
"""Storage abstractions for Isotope."""

from isotopedb.stores.base import AtomStore, DocStore, VectorStore
from isotopedb.stores.sqlite import SQLiteDocStore
from isotopedb.stores.sqlite_atom import SQLiteAtomStore

try:
    from isotopedb.stores.chroma import ChromaVectorStore
except ImportError:
    from isotopedb._optional import _create_missing_dependency_class

    ChromaVectorStore = _create_missing_dependency_class("ChromaVectorStore", "chromadb")

__all__ = [
    "VectorStore",
    "DocStore",
    "AtomStore",
    "SQLiteDocStore",
    "SQLiteAtomStore",
    "ChromaVectorStore",
]

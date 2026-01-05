# src/isotope/configuration/storage/local.py
"""Local filesystem storage configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isotope.stores import AtomStore, ChunkStore, EmbeddedQuestionStore, SourceRegistry


@dataclass(frozen=True)
class LocalStorage:
    """Local filesystem storage using Chroma and SQLite.

    All data is persisted to the specified directory:
    - chroma/: Embedded questions (ChromaDB)
    - chunks.db: Original text chunks (SQLite)
    - atoms.db: Atomic statements (SQLite)
    - sources.db: Source content hashes (SQLite)

    Requires the chromadb package: pip install isotope-rag[chroma]

    Args:
        data_dir: Base directory for all storage files.
                  Created if it doesn't exist.

    Example:
        storage = LocalStorage("./my_data")

        # In combination with a provider:
        iso = Isotope(
            provider=LiteLLMProvider(llm="openai/gpt-4o", embedding="text-embedding-3-small"),
            storage=LocalStorage("./my_data"),
        )
    """

    data_dir: str

    def build_stores(self) -> tuple[EmbeddedQuestionStore, ChunkStore, AtomStore, SourceRegistry]:
        """Build all four storage components.

        Creates the data directory if it doesn't exist.

        Returns:
            Tuple of (embedded_question_store, chunk_store, atom_store, source_registry)
        """
        from isotope.stores import (
            ChromaEmbeddedQuestionStore,
            SQLiteAtomStore,
            SQLiteChunkStore,
            SQLiteSourceRegistry,
        )

        # Ensure directory exists
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)

        embedded_question_store = ChromaEmbeddedQuestionStore(os.path.join(self.data_dir, "chroma"))
        chunk_store = SQLiteChunkStore(os.path.join(self.data_dir, "chunks.db"))
        atom_store = SQLiteAtomStore(os.path.join(self.data_dir, "atoms.db"))
        source_registry = SQLiteSourceRegistry(os.path.join(self.data_dir, "sources.db"))

        return embedded_question_store, chunk_store, atom_store, source_registry

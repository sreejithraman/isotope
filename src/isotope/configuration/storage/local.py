# src/isotope/configuration/storage/local.py
"""Local filesystem storage configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from isotope.configuration.base import StorageBundle


@dataclass(frozen=True)
class LocalStorage:
    """Local filesystem storage using Chroma and SQLite.

    All data is persisted to the specified directory:
    - chroma/: Embedded questions and chunk embeddings (ChromaDB, separate collections)
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
            provider=LiteLLMProvider(
                llm="openai/gpt-5-mini-2025-08-07",
                embedding="openai/text-embedding-3-small",
            ),
            storage=LocalStorage("./my_data"),
        )
    """

    data_dir: str

    def build_stores(self) -> StorageBundle:
        """Build all five storage components.

        Creates the data directory if it doesn't exist.

        Returns:
            StorageBundle with named store fields.
        """
        from isotope.stores import (
            ChromaChunkEmbeddingStore,
            ChromaEmbeddedQuestionStore,
            SQLiteAtomStore,
            SQLiteChunkStore,
            SQLiteSourceRegistry,
        )

        # Ensure directory exists
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)

        chroma_dir = os.path.join(self.data_dir, "chroma")
        embedded_question_store = ChromaEmbeddedQuestionStore(chroma_dir)
        chunk_embedding_store = ChromaChunkEmbeddingStore(chroma_dir)
        chunk_store = SQLiteChunkStore(os.path.join(self.data_dir, "chunks.db"))
        atom_store = SQLiteAtomStore(os.path.join(self.data_dir, "atoms.db"))
        source_registry = SQLiteSourceRegistry(os.path.join(self.data_dir, "sources.db"))

        return StorageBundle(
            embedded_question_store=embedded_question_store,
            chunk_embedding_store=chunk_embedding_store,
            chunk_store=chunk_store,
            atom_store=atom_store,
            source_registry=source_registry,
        )

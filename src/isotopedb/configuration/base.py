# src/isotopedb/configuration/base.py
"""Protocol definitions for configuration objects.

These protocols define the interfaces for provider and storage configurations.
Implementations can use @dataclass(frozen=True) for immutability.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from isotopedb.atomizer import Atomizer
    from isotopedb.config import Settings
    from isotopedb.embedder import Embedder
    from isotopedb.question_generator import QuestionGenerator
    from isotopedb.stores import AtomStore, ChunkStore, SourceRegistry, VectorStore


@runtime_checkable
class ProviderConfig(Protocol):
    """Protocol for provider configurations.

    Provider configurations build the AI/ML components:
    - Embedder: Creates vector embeddings for semantic search
    - Atomizer: Breaks chunks into atomic statements
    - QuestionGenerator: Generates synthetic questions for atoms

    Example implementation:
        @dataclass(frozen=True)
        class LiteLLMProvider:
            llm: str
            embedding: str

            def build_embedder(self) -> Embedder: ...
            def build_atomizer(self, settings: Settings) -> Atomizer: ...
            def build_question_generator(self, settings: Settings) -> QuestionGenerator: ...
    """

    def build_embedder(self) -> Embedder:
        """Build an embedder for creating vector embeddings."""
        ...

    def build_atomizer(self, settings: Settings) -> Atomizer:
        """Build an atomizer for breaking chunks into atoms.

        Args:
            settings: Settings containing atomizer_prompt if customized.
        """
        ...

    def build_question_generator(self, settings: Settings) -> QuestionGenerator:
        """Build a question generator for creating synthetic questions.

        Args:
            settings: Settings containing questions_per_atom and
                      question_generator_prompt if customized.
        """
        ...


@runtime_checkable
class StorageConfig(Protocol):
    """Protocol for storage configurations.

    Storage configurations build the data stores:
    - VectorStore: Stores and retrieves question embeddings
    - ChunkStore: Stores original text chunks
    - AtomStore: Stores atomic statements
    - SourceRegistry: Tracks source content hashes for change detection

    Example implementation:
        @dataclass(frozen=True)
        class LocalStorage:
            data_dir: str

            def build_stores(self) -> tuple[VectorStore, ChunkStore, AtomStore, SourceRegistry]: ...
    """

    def build_stores(self) -> tuple[VectorStore, ChunkStore, AtomStore, SourceRegistry]:
        """Build all four storage components.

        Returns:
            Tuple of (vector_store, chunk_store, atom_store, source_registry)
        """
        ...

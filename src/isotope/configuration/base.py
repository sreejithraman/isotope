# src/isotope/configuration/base.py
"""Protocol definitions for configuration objects.

These protocols define the interfaces for provider and storage configurations.
Implementations can use @dataclass(frozen=True) for immutability.

Design Note: Why Protocols here vs ABCs in stores/base.py?

- **Protocols (here):** Structural typing for configuration factories. Any frozen
  dataclass with the right methods satisfies the interface without inheritance.
  Good for provider/storage configs where implementations vary by vendor.

- **ABCs (stores):** Nominal typing requiring explicit inheritance. Good for
  storage implementations that need polymorphic behavior and may share
  implementation via inheritance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from isotope.atomizer import Atomizer
    from isotope.embedder import Embedder
    from isotope.providers import LLMClient
    from isotope.question_generator import QuestionGenerator
    from isotope.settings import Settings
    from isotope.stores import AtomStore, ChunkStore, EmbeddedQuestionStore, SourceRegistry


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

    def build_llm_client(self) -> LLMClient:
        """Build an LLM client for general-purpose completions.

        This can be used for answer synthesis or any other LLM task
        not covered by the specialized component builders.
        """
        ...


@runtime_checkable
class StorageConfig(Protocol):
    """Protocol for storage configurations.

    Storage configurations build the data stores:
    - EmbeddedQuestionStore: Stores and retrieves embedded questions
    - ChunkStore: Stores original text chunks
    - AtomStore: Stores atomic statements
    - SourceRegistry: Tracks source content hashes for change detection

    Example implementation:
        @dataclass(frozen=True)
        class LocalStorage:
            data_dir: str

            def build_stores(
                self,
            ) -> tuple[EmbeddedQuestionStore, ChunkStore, AtomStore, SourceRegistry]: ...
    """

    def build_stores(self) -> tuple[EmbeddedQuestionStore, ChunkStore, AtomStore, SourceRegistry]:
        """Build all four storage components.

        Returns:
            Tuple of (embedded_question_store, chunk_store, atom_store, source_registry)
        """
        ...

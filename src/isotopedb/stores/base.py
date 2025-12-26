# src/isotopedb/stores/base.py
"""Abstract base classes for storage."""

from abc import ABC, abstractmethod

from isotopedb.models import Chunk, EmbeddedQuestion, Question


class VectorStore(ABC):
    """Abstract base class for vector storage."""

    @abstractmethod
    def add(self, questions: list[EmbeddedQuestion]) -> None:
        """Add questions with embeddings to the store."""
        ...

    @abstractmethod
    def search(self, embedding: list[float], k: int = 5) -> list[tuple[Question, float]]:
        """Search for similar questions. Returns (Question, score) pairs ordered by relevance."""
        ...

    @abstractmethod
    def delete_by_chunk_ids(self, chunk_ids: list[str]) -> None:
        """Delete all questions associated with the given chunk IDs."""
        ...

    @abstractmethod
    def list_chunk_ids(self) -> set[str]:
        """List all unique chunk IDs in the store."""
        ...


class DocStore(ABC):
    """Abstract base class for document storage."""

    @abstractmethod
    def put(self, chunk: Chunk) -> None:
        """Store a chunk."""
        ...

    @abstractmethod
    def get(self, chunk_id: str) -> Chunk | None:
        """Retrieve a chunk by ID. Returns None if not found."""
        ...

    @abstractmethod
    def get_many(self, chunk_ids: list[str]) -> list[Chunk]:
        """Retrieve multiple chunks by ID. Skips missing chunks."""
        ...

    @abstractmethod
    def delete(self, chunk_id: str) -> None:
        """Delete a chunk by ID."""
        ...

    @abstractmethod
    def list_sources(self) -> list[str]:
        """List all unique sources in the store."""
        ...

    @abstractmethod
    def get_by_source(self, source: str) -> list[Chunk]:
        """Get all chunks from a specific source."""
        ...

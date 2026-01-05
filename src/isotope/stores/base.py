# src/isotope/stores/base.py
"""Abstract base classes for storage."""

from abc import ABC, abstractmethod

from isotope.models import Atom, Chunk, EmbeddedQuestion, Question


class EmbeddedQuestionStore(ABC):
    """Abstract base class for embedded question storage."""

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

    @abstractmethod
    def count_questions(self) -> int:
        """Count the total number of questions in the store."""
        ...


class ChunkStore(ABC):
    """Abstract base class for chunk storage."""

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
    def put_many(self, chunks: list[Chunk]) -> None:
        """Store multiple chunks, overwriting if they exist."""
        ...

    @abstractmethod
    def delete_many(self, chunk_ids: list[str]) -> None:
        """Delete multiple chunks by ID."""
        ...

    @abstractmethod
    def count_chunks(self) -> int:
        """Count the total number of chunks in the store."""
        ...

    @abstractmethod
    def list_sources(self) -> list[str]:
        """List all unique sources in the store."""
        ...

    @abstractmethod
    def get_by_source(self, source: str) -> list[Chunk]:
        """Get all chunks from a specific source."""
        ...

    @abstractmethod
    def get_chunk_ids_by_source(self, source: str) -> list[str]:
        """Get all chunk IDs for a specific source."""
        ...

    @abstractmethod
    def delete_by_source(self, source: str) -> None:
        """Delete all chunks from a specific source."""
        ...


class AtomStore(ABC):
    """Abstract base class for atom storage."""

    @abstractmethod
    def put(self, atom: Atom) -> None:
        """Store an atom."""
        ...

    @abstractmethod
    def put_many(self, atoms: list[Atom]) -> None:
        """Store multiple atoms."""
        ...

    @abstractmethod
    def get(self, atom_id: str) -> Atom | None:
        """Retrieve an atom by ID."""
        ...

    @abstractmethod
    def get_by_chunk(self, chunk_id: str) -> list[Atom]:
        """Get all atoms from a chunk, ordered by index."""
        ...

    @abstractmethod
    def delete_by_chunk_ids(self, chunk_ids: list[str]) -> None:
        """Delete all atoms for given chunks."""
        ...

    @abstractmethod
    def list_chunk_ids(self) -> set[str]:
        """List all unique chunk IDs with atoms."""
        ...

    @abstractmethod
    def count_atoms(self) -> int:
        """Count the total number of atoms in the store."""
        ...


class SourceRegistry(ABC):
    """Tracks source metadata for change detection.

    Similar to LangChain's RecordManager - keeps tracking separate from
    document storage. Stores content hashes to detect when files change.
    """

    @abstractmethod
    def get_hash(self, source: str) -> str | None:
        """Get content hash for a source, or None if not tracked."""

    @abstractmethod
    def set_hash(self, source: str, content_hash: str) -> None:
        """Store content hash after successful ingestion."""

    @abstractmethod
    def delete(self, source: str) -> None:
        """Remove tracking for a source."""

    @abstractmethod
    def list_sources(self) -> list[str]:
        """List all tracked sources."""

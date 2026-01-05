# src/isotope/loaders/base.py
"""Loader abstract base class."""

from abc import ABC, abstractmethod

from isotope.models import Chunk


class Loader(ABC):
    """Abstract base class for file loading."""

    @abstractmethod
    def load(self, path: str, source_id: str | None = None) -> list[Chunk]:
        """Load a file and return chunks.

        Args:
            path: Path to the file to load
            source_id: Optional custom source identifier. If not provided,
                      the absolute path will be used as the source.

        Returns:
            List of Chunk objects with source set to source_id or absolute path
        """
        ...

    @abstractmethod
    def supports(self, path: str) -> bool:
        """Check if this loader supports the given path."""
        ...

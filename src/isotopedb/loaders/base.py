# src/isotopedb/loaders/base.py
"""Loader abstract base class."""

from abc import ABC, abstractmethod

from isotopedb.models import Chunk


class Loader(ABC):
    """Abstract base class for file loading."""

    @abstractmethod
    def load(self, path: str) -> list[Chunk]:
        """Load a file and return chunks."""
        ...

    @abstractmethod
    def supports(self, path: str) -> bool:
        """Check if this loader supports the given path."""
        ...

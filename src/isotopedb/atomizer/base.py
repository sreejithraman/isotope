# src/isotopedb/atomizer/base.py
"""Atomizer abstract base class."""

from abc import ABC, abstractmethod

from isotopedb.models import Atom, Chunk


class Atomizer(ABC):
    """Abstract base class for atomization."""

    @abstractmethod
    def atomize(self, chunk: Chunk) -> list[Atom]:
        """Break a chunk into atomic statements."""
        ...

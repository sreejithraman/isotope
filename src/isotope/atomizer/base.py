# src/isotope/atomizer/base.py
"""Atomizer abstract base class."""

from abc import ABC, abstractmethod

from isotope.models import Atom, Chunk


class Atomizer(ABC):
    """Abstract base class for atomization."""

    @abstractmethod
    def atomize(self, chunk: Chunk) -> list[Atom]:
        """Break a chunk into atomic statements."""
        ...

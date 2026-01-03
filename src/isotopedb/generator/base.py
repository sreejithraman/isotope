# src/isotopedb/generator/base.py
"""QuestionGenerator abstract base class."""

from abc import ABC, abstractmethod

from isotopedb.models import Atom, Question


class QuestionGenerator(ABC):
    """Abstract base class for question generation."""

    @abstractmethod
    def generate(self, atom: Atom, chunk_content: str = "") -> list[Question]:
        """Generate questions for a single atom."""
        ...

    @abstractmethod
    def generate_batch(self, atoms: list[Atom], chunk_content: str = "") -> list[Question]:
        """Generate questions for multiple atoms."""
        ...

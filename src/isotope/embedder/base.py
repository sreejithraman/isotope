# src/isotope/embedder/base.py
"""Embedder abstract base class."""

from abc import ABC, abstractmethod

from isotope.models import EmbeddedQuestion, Question


class Embedder(ABC):
    """Abstract base class for embedding generation."""

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Generate an embedding vector for a single text."""
        ...

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embedding vectors for multiple texts (batched)."""
        ...

    @abstractmethod
    def embed_question(self, question: Question) -> EmbeddedQuestion:
        """Embed a single question."""
        ...

    @abstractmethod
    def embed_questions(self, questions: list[Question]) -> list[EmbeddedQuestion]:
        """Embed multiple questions (batched for efficiency)."""
        ...

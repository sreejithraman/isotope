# src/isotope/embedder/base.py
"""Embedder abstract base class."""

from abc import ABC, abstractmethod

from isotope.models import EmbeddedQuestion, Question


class Embedder(ABC):
    """Abstract base class for embedding generation.

    Subclasses must implement embed_text and embed_texts.
    """

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Generate an embedding vector for a single text."""
        ...

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embedding vectors for multiple texts (batched)."""
        ...

    def embed_question(self, question: Question) -> EmbeddedQuestion:
        """Embed a single question."""
        embedding = self.embed_text(question.text)
        return EmbeddedQuestion(question=question, embedding=embedding)

    def embed_questions(self, questions: list[Question]) -> list[EmbeddedQuestion]:
        """Embed multiple questions (batched for efficiency)."""
        if not questions:
            return []
        texts = [q.text for q in questions]
        embeddings = self.embed_texts(texts)
        return [
            EmbeddedQuestion(question=q, embedding=emb)
            for q, emb in zip(questions, embeddings, strict=True)
        ]

# src/isotopedb/embedder/base.py
"""Embedder abstract base class."""

from abc import ABC, abstractmethod

from isotopedb.models import EmbeddedQuestion, Question


class Embedder(ABC):
    """Abstract base class for embedding generation.

    Implementations of this class generate vector embeddings for text.
    The default implementation uses LiteLLM, but users can implement
    their own using any embedding provider.

    Example:
        class MyEmbedder(Embedder):
            def embed_text(self, text: str) -> list[float]:
                return my_embedding_api.embed(text)

            def embed_texts(self, texts: list[str]) -> list[list[float]]:
                return [self.embed_text(t) for t in texts]

            def embed_question(self, question: Question) -> EmbeddedQuestion:
                embedding = self.embed_text(question.text)
                return EmbeddedQuestion(question=question, embedding=embedding)

            def embed_questions(self, questions: list[Question]) -> list[EmbeddedQuestion]:
                return [self.embed_question(q) for q in questions]
    """

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Generate an embedding vector for a single text.

        Args:
            text: The text to embed

        Returns:
            A list of floats representing the embedding vector
        """
        ...

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embedding vectors for multiple texts (batched).

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors, one per input text
        """
        ...

    @abstractmethod
    def embed_question(self, question: Question) -> EmbeddedQuestion:
        """Embed a single question.

        Args:
            question: The Question to embed

        Returns:
            An EmbeddedQuestion with the embedding attached
        """
        ...

    @abstractmethod
    def embed_questions(self, questions: list[Question]) -> list[EmbeddedQuestion]:
        """Embed multiple questions (batched for efficiency).

        Args:
            questions: List of Questions to embed

        Returns:
            List of EmbeddedQuestions with embeddings attached
        """
        ...

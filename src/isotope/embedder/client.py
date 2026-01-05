# src/isotope/embedder/client.py
"""Client-based embedder implementation."""

from isotope.embedder.base import Embedder
from isotope.models import EmbeddedQuestion, Question
from isotope.providers.base import EmbeddingClient


class ClientEmbedder(Embedder):
    """Embedder that uses an EmbeddingClient for generating embeddings.

    Example:
        from isotope.providers.litellm import LiteLLMEmbeddingClient
        from isotope.embedder import ClientEmbedder

        client = LiteLLMEmbeddingClient(model="openai/text-embedding-3-small")
        embedder = ClientEmbedder(embedding_client=client)
    """

    def __init__(self, embedding_client: EmbeddingClient) -> None:
        """Initialize the embedder.

        Args:
            embedding_client: Any EmbeddingClient implementation
        """
        self._client = embedding_client

    def embed_text(self, text: str) -> list[float]:
        """Generate an embedding vector for a single text."""
        result = self._client.embed([text])
        return result[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embedding vectors for multiple texts (batched)."""
        return self._client.embed(texts)

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

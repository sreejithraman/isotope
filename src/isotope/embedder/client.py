# src/isotope/embedder/client.py
"""Client-based embedder implementation."""

from isotope.embedder.base import Embedder
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

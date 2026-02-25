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

    def __init__(
        self,
        embedding_client: EmbeddingClient,
        batch_size: int | None = None,
    ) -> None:
        """Initialize the embedder.

        Args:
            embedding_client: Any EmbeddingClient implementation
            batch_size: Max texts per embedding API call. None means no limit.
        """
        self._client = embedding_client
        self._batch_size = batch_size

    def embed_text(self, text: str) -> list[float]:
        """Generate an embedding vector for a single text."""
        result = self._client.embed([text])
        return result[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embedding vectors for multiple texts (batched)."""
        if self._batch_size is None or len(texts) <= self._batch_size:
            return self._client.embed(texts)

        results: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            results.extend(self._client.embed(batch))
        return results

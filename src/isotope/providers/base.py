# src/isotope/providers/base.py
"""Abstract base classes for LLM and embedding providers."""

from abc import ABC, abstractmethod


class LLMClient(ABC):
    """Abstract base class for LLM completion providers.

    Implementations of this class provide text generation/completion
    capabilities. The interface is intentionally minimal to support
    the widest range of providers.

    Example:
        class MyLLMClient(LLMClient):
            def complete(self, messages, temperature=None):
                return my_api.chat(messages, temp=temperature)
    """

    @abstractmethod
    def complete(
        self,
        messages: list[dict],
        temperature: float | None = None,
    ) -> str:
        """Generate a completion for the given messages.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
                      Example: [{"role": "user", "content": "Hello"}]
            temperature: Optional temperature for generation (0.0-1.0).
                         If None, use provider default.

        Returns:
            The generated text response.
        """
        ...

    async def acomplete(
        self,
        messages: list[dict],
        temperature: float | None = None,
    ) -> str:
        """Generate a completion for the given messages (async).

        Default implementation calls sync complete() for backwards compatibility.
        Override in subclasses for true async behavior.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            temperature: Optional temperature for generation (0.0-1.0).

        Returns:
            The generated text response.
        """
        return self.complete(messages, temperature)


class EmbeddingClient(ABC):
    """Abstract base class for embedding providers.

    Implementations of this class generate vector embeddings for text.
    The interface supports batched embedding for efficiency.

    Example:
        class MyEmbeddingClient(EmbeddingClient):
            def embed(self, texts):
                return my_api.embed_batch(texts)
    """

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embedding vectors for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors, one per input text.
            Order is preserved (result[i] corresponds to texts[i]).
        """
        ...

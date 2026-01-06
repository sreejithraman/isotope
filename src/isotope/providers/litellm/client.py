# src/isotope/providers/litellm/client.py
"""LiteLLM client implementations for LLM and embedding APIs."""

import litellm

from isotope.providers.base import EmbeddingClient, LLMClient
from isotope.providers.litellm.models import ChatModels, EmbeddingModels


class LiteLLMClient(LLMClient):
    """LiteLLM-based LLM client for text generation.

    Supports any model available through LiteLLM (OpenAI, Anthropic, Gemini,
    Bedrock, etc.).

    Example:
        from isotope.providers.litellm import LiteLLMClient, ChatModels

        client = LiteLLMClient(model=ChatModels.GEMINI_3_FLASH)
        response = client.complete([{"role": "user", "content": "Hello"}])

        # With retry for rate-limited APIs
        client = LiteLLMClient(model=ChatModels.GEMINI_3_FLASH, num_retries=5)
    """

    def __init__(
        self,
        model: str = ChatModels.GEMINI_3_FLASH,
        num_retries: int = 3,
    ) -> None:
        """Initialize the LiteLLM client.

        Args:
            model: LiteLLM model identifier.
                   Examples: "openai/gpt-5-mini-2025-08-07", "anthropic/claude-sonnet-4-5-20250929"
            num_retries: Number of retries on rate limit errors. LiteLLM handles
                        exponential backoff automatically. Default: 3.
        """
        self.model = model
        self.num_retries = num_retries

    def complete(
        self,
        messages: list[dict],
        temperature: float | None = None,
    ) -> str:
        """Generate a completion using LiteLLM."""
        completion_kwargs = {
            "model": self.model,
            "messages": messages,
            "drop_params": True,
            "num_retries": self.num_retries,
        }
        if temperature is not None:
            completion_kwargs["temperature"] = temperature

        response = litellm.completion(**completion_kwargs)

        if not response.choices:
            raise ValueError(f"LLM returned no choices for model {self.model}")
        content = response.choices[0].message.content
        if content is None:
            raise ValueError(f"LLM returned None content for model {self.model}")
        return str(content)

    async def acomplete(
        self,
        messages: list[dict],
        temperature: float | None = None,
    ) -> str:
        """Generate a completion using LiteLLM (async)."""
        completion_kwargs = {
            "model": self.model,
            "messages": messages,
            "drop_params": True,
            "num_retries": self.num_retries,
        }
        if temperature is not None:
            completion_kwargs["temperature"] = temperature

        response = await litellm.acompletion(**completion_kwargs)

        if not response.choices:
            raise ValueError(f"LLM returned no choices for model {self.model}")
        content = response.choices[0].message.content
        if content is None:
            raise ValueError(f"LLM returned None content for model {self.model}")
        return str(content)


class LiteLLMEmbeddingClient(EmbeddingClient):
    """LiteLLM-based embedding client.

    Supports any embedding model available through LiteLLM.

    Example:
        from isotope.providers.litellm import LiteLLMEmbeddingClient, EmbeddingModels

        client = LiteLLMEmbeddingClient(model=EmbeddingModels.TEXT_3_SMALL)
        embeddings = client.embed(["Hello world", "How are you?"])

        # With retry for rate-limited APIs
        client = LiteLLMEmbeddingClient(model=EmbeddingModels.TEXT_3_SMALL, num_retries=5)
    """

    def __init__(
        self,
        model: str = EmbeddingModels.GEMINI_EMBEDDING_001,
        num_retries: int = 3,
    ) -> None:
        """Initialize the LiteLLM embedding client.

        Args:
            model: LiteLLM embedding model identifier.
                   Examples: "openai/text-embedding-3-small", "gemini/gemini-embedding-001"
            num_retries: Number of retries on rate limit errors. LiteLLM handles
                        exponential backoff automatically. Default: 3.
        """
        self.model = model
        self.num_retries = num_retries

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using LiteLLM."""
        if not texts:
            return []

        response = litellm.embedding(
            model=self.model,
            input=texts,
            num_retries=self.num_retries,
        )
        # Sort by index to maintain order
        sorted_data = sorted(response.data, key=lambda x: x["index"])
        return [item["embedding"] for item in sorted_data]

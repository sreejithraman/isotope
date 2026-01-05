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
    """

    def __init__(self, model: str = ChatModels.GEMINI_3_FLASH) -> None:
        """Initialize the LiteLLM client.

        Args:
            model: LiteLLM model identifier.
                   Examples: "openai/gpt-4o", "anthropic/claude-sonnet-4-5-20250929"
        """
        self.model = model

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
    """

    def __init__(self, model: str = EmbeddingModels.GEMINI_004) -> None:
        """Initialize the LiteLLM embedding client.

        Args:
            model: LiteLLM embedding model identifier.
                   Examples: "openai/text-embedding-3-small", "gemini/text-embedding-004"
        """
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using LiteLLM."""
        if not texts:
            return []

        response = litellm.embedding(
            model=self.model,
            input=texts,
        )
        # Sort by index to maintain order
        sorted_data = sorted(response.data, key=lambda x: x["index"])
        return [item["embedding"] for item in sorted_data]

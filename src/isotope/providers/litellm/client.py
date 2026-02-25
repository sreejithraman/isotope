# src/isotope/providers/litellm/client.py
"""LiteLLM client implementations for LLM and embedding APIs."""

import re
import traceback

import litellm
from litellm.types.utils import Choices

from isotope.providers.base import EmbeddingClient, LLMClient
from isotope.providers.litellm.models import ChatModels, EmbeddingModels

litellm.suppress_debug_info = True


class LLMError(Exception):
    """Exception for LLM API errors with clean user-facing messages.

    Attributes:
        message: Clean, user-friendly error message
        details: Full error details for verbose/debug mode
    """

    def __init__(self, message: str, details: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details


def _extract_clean_message(e: Exception) -> tuple[str, str]:
    """Extract a clean user-facing message from a LiteLLM exception.

    Returns:
        Tuple of (clean_message, full_details)
    """
    full_details = "".join(traceback.format_exception(type(e), e, e.__traceback__))
    error_str = str(e)

    # Strip LiteLLM boilerplate
    boilerplate_pattern = (
        r"Give Feedback / Get Help:.*?LiteLLM\.Info:.*?`litellm\._turn_on_debug\(\)'\.?"
    )
    cleaned = re.sub(boilerplate_pattern, "", error_str, flags=re.DOTALL).strip()

    # Handle specific error types
    error_type = type(e).__name__

    if "AuthenticationError" in error_type or "401" in error_str:
        clean = "Authentication failed: Check your API key"
        if cleaned and cleaned != clean:
            clean = f"Authentication failed: {cleaned}"
    elif "RateLimitError" in error_type or "429" in error_str:
        clean = f"Rate limited: {cleaned}" if cleaned else "Rate limited by API"
    elif "APIConnectionError" in error_type or "connection" in error_str.lower():
        clean = f"Connection failed: {cleaned}" if cleaned else "Failed to connect to API"
    elif "NotFoundError" in error_type or "404" in error_str:
        clean = f"Model not found: {cleaned}" if cleaned else "Model not found"
    elif cleaned:
        clean = cleaned
    else:
        clean = f"API error: {error_type}"

    # Ensure clean message doesn't still have boilerplate
    clean = re.sub(boilerplate_pattern, "", clean, flags=re.DOTALL).strip()

    return clean, full_details


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

        # With explicit API key (bypasses env var lookup)
        client = LiteLLMClient(model=ChatModels.GPT_5_MINI, api_key="sk-...")
    """

    def __init__(
        self,
        model: str = ChatModels.GEMINI_3_FLASH,
        num_retries: int = 3,
        api_key: str | None = None,
    ) -> None:
        """Initialize the LiteLLM client.

        Args:
            model: LiteLLM model identifier.
                   Examples: "openai/gpt-5-mini-2025-08-07", "anthropic/claude-sonnet-4-5-20250929"
            num_retries: Number of retries on rate limit errors. LiteLLM handles
                        exponential backoff automatically. Default: 3.
            api_key: Optional API key. If provided, bypasses environment variable lookup.
        """
        self.model = model
        self.num_retries = num_retries
        self.api_key = api_key

    def _build_completion_kwargs(self, messages: list[dict], temperature: float | None) -> dict:
        """Build kwargs for litellm completion calls."""
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "drop_params": True,
            "num_retries": self.num_retries,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if self.api_key is not None:
            kwargs["api_key"] = self.api_key
        return kwargs

    def _extract_content(self, response: litellm.ModelResponse) -> str:
        """Extract content from litellm response."""
        if not response.choices:
            raise ValueError(f"LLM returned no choices for model {self.model}")

        choice = response.choices[0]

        # We use non-streaming completion, so expect Choices (not StreamingChoices)
        if not isinstance(choice, Choices):
            raise ValueError(
                f"Expected non-streaming response (Choices), got {type(choice).__name__}. "
                f"This client does not support streaming."
            )

        content = choice.message.content
        if content is None:
            raise ValueError(f"LLM returned None content for model {self.model}")
        return str(content)

    def complete(
        self,
        messages: list[dict],
        temperature: float | None = None,
    ) -> str:
        """Generate a completion using LiteLLM."""
        kwargs = self._build_completion_kwargs(messages, temperature)
        try:
            response = litellm.completion(**kwargs)
        except Exception as e:
            clean, details = _extract_clean_message(e)
            raise LLMError(clean, details) from e
        return self._extract_content(response)

    async def acomplete(
        self,
        messages: list[dict],
        temperature: float | None = None,
    ) -> str:
        """Generate a completion using LiteLLM (async)."""
        kwargs = self._build_completion_kwargs(messages, temperature)
        try:
            response = await litellm.acompletion(**kwargs)
        except Exception as e:
            clean, details = _extract_clean_message(e)
            raise LLMError(clean, details) from e
        return self._extract_content(response)


class LiteLLMEmbeddingClient(EmbeddingClient):
    """LiteLLM-based embedding client.

    Supports any embedding model available through LiteLLM.

    Example:
        from isotope.providers.litellm import LiteLLMEmbeddingClient, EmbeddingModels

        client = LiteLLMEmbeddingClient(model=EmbeddingModels.TEXT_3_SMALL)
        embeddings = client.embed(["Hello world", "How are you?"])

        # With retry for rate-limited APIs
        client = LiteLLMEmbeddingClient(model=EmbeddingModels.TEXT_3_SMALL, num_retries=5)

        # With explicit API key (bypasses env var lookup)
        client = LiteLLMEmbeddingClient(model=EmbeddingModels.TEXT_3_SMALL, api_key="sk-...")
    """

    def __init__(
        self,
        model: str = EmbeddingModels.GEMINI_EMBEDDING_001,
        num_retries: int = 3,
        api_key: str | None = None,
    ) -> None:
        """Initialize the LiteLLM embedding client.

        Args:
            model: LiteLLM embedding model identifier.
                   Examples: "openai/text-embedding-3-small", "gemini/gemini-embedding-001"
            num_retries: Number of retries on rate limit errors. LiteLLM handles
                        exponential backoff automatically. Default: 3.
            api_key: Optional API key. If provided, bypasses environment variable lookup.
        """
        self.model = model
        self.num_retries = num_retries
        self.api_key = api_key

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using LiteLLM."""
        if not texts:
            return []

        embedding_kwargs: dict = {
            "model": self.model,
            "input": texts,
            "num_retries": self.num_retries,
        }
        if self.api_key is not None:
            embedding_kwargs["api_key"] = self.api_key

        try:
            response = litellm.embedding(**embedding_kwargs)
        except Exception as e:
            clean, details = _extract_clean_message(e)
            raise LLMError(clean, details) from e
        # Sort by index to maintain order
        sorted_data = sorted(response.data, key=lambda x: x["index"])
        return [item["embedding"] for item in sorted_data]

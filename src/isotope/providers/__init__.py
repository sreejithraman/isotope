# src/isotope/providers/__init__.py
"""Provider implementations for Isotope.

This module contains LLM and embedding provider abstractions:
- LLMClient: Abstract base class for LLM completion providers
- EmbeddingClient: Abstract base class for embedding providers
- LiteLLM implementations (requires: pip install isotope-rag[litellm])

Usage:
    from isotope.providers import LLMClient, EmbeddingClient
    from isotope.providers.litellm import LiteLLMClient, ChatModels
"""

from isotope.providers.base import EmbeddingClient, LLMClient

try:
    from isotope.providers.litellm import (
        ChatModels,
        EmbeddingModels,
        LiteLLMClient,
        LiteLLMEmbeddingClient,
    )
except ImportError:
    from isotope._optional import _create_missing_dependency_class

    class ChatModels:  # type: ignore[no-redef]
        """Placeholder - requires litellm package."""

        pass

    class EmbeddingModels:  # type: ignore[no-redef]
        """Placeholder - requires litellm package."""

        pass

    LiteLLMClient = _create_missing_dependency_class(  # type: ignore[misc,assignment]
        "LiteLLMClient", "litellm"
    )
    LiteLLMEmbeddingClient = _create_missing_dependency_class(  # type: ignore[misc,assignment]
        "LiteLLMEmbeddingClient", "litellm"
    )

__all__ = [
    # ABCs
    "LLMClient",
    "EmbeddingClient",
    # Model constants
    "ChatModels",
    "EmbeddingModels",
    # LiteLLM clients
    "LiteLLMClient",
    "LiteLLMEmbeddingClient",
]

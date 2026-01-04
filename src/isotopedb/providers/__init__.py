# src/isotopedb/providers/__init__.py
"""Provider implementations for IsotopeDB.

This module contains LLM and embedding provider abstractions:
- LLMClient: Abstract base class for LLM completion providers
- EmbeddingClient: Abstract base class for embedding providers
- LiteLLM implementations (requires: pip install isotopedb[litellm])

Usage:
    from isotopedb.providers import LLMClient, EmbeddingClient
    from isotopedb.providers.litellm import LiteLLMClient, ChatModels
"""

from isotopedb.providers.base import EmbeddingClient, LLMClient

try:
    from isotopedb.providers.litellm import (
        ChatModels,
        EmbeddingModels,
        LiteLLMClient,
        LiteLLMEmbeddingClient,
    )
except ImportError:
    from isotopedb._optional import _create_missing_dependency_class

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

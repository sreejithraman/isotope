# src/isotopedb/providers/__init__.py
"""Provider implementations for IsotopeDB.

This module contains LLM and embedding provider abstractions:
- LLMClient: Abstract base class for LLM completion providers
- EmbeddingClient: Abstract base class for embedding providers
- LiteLLM implementations

Usage:
    from isotopedb.providers import LLMClient, EmbeddingClient
    from isotopedb.providers.litellm import LiteLLMClient, ChatModels
"""

from isotopedb.providers.base import EmbeddingClient, LLMClient
from isotopedb.providers.litellm import (
    ChatModels,
    EmbeddingModels,
    LiteLLMClient,
    LiteLLMEmbeddingClient,
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

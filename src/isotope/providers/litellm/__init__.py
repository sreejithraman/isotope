# src/isotope/providers/litellm/__init__.py
"""LiteLLM provider clients for Isotope.

This module contains LiteLLM-based client implementations:
- LiteLLMClient: LLM completion using LiteLLM
- LiteLLMEmbeddingClient: Embeddings using LiteLLM
- ChatModels: Curated chat model constants
- EmbeddingModels: Curated embedding model constants

Usage:
    from isotope.providers.litellm import LiteLLMClient, ChatModels
    from isotope.atomizer import LLMAtomizer

    client = LiteLLMClient(model=ChatModels.GEMINI_3_FLASH)
    atomizer = LLMAtomizer(llm_client=client)
"""

from isotope.providers.litellm.client import LiteLLMClient, LiteLLMEmbeddingClient
from isotope.providers.litellm.models import ChatModels, EmbeddingModels

__all__ = [
    # Model constants
    "ChatModels",
    "EmbeddingModels",
    # Clients
    "LiteLLMClient",
    "LiteLLMEmbeddingClient",
]

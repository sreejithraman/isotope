# src/isotopedb/litellm/__init__.py
"""LiteLLM provider implementation for IsotopeDB.

This module contains all LiteLLM-based implementations:
- LiteLLMEmbedder: Embedding using LiteLLM
- LiteLLMGenerator: Question generation using LiteLLM
- LiteLLMAtomizer: Atomization using LiteLLM

Usage:
    from isotopedb.litellm import LiteLLMEmbedder, LiteLLMGenerator, LiteLLMAtomizer
    from isotopedb.litellm import ChatModels, EmbeddingModels
"""

from isotopedb.litellm.atomizer import LiteLLMAtomizer
from isotopedb.litellm.embedder import LiteLLMEmbedder
from isotopedb.litellm.generator import LiteLLMGenerator
from isotopedb.litellm.models import ChatModels, EmbeddingModels

# Backwards compatibility alias
LiteLLMQuestionGenerator = LiteLLMGenerator

__all__ = [
    "LiteLLMEmbedder",
    "LiteLLMGenerator",
    "LiteLLMQuestionGenerator",  # Alias for backwards compat
    "LiteLLMAtomizer",
    "ChatModels",
    "EmbeddingModels",
]

# src/isotopedb/embedder/__init__.py
"""Embedding functionality for IsotopeDB.

This module exports the Embedder abstract base class.
For the LiteLLM implementation, use:
    from isotopedb.litellm import LiteLLMEmbedder
"""

from isotopedb.embedder.base import Embedder

__all__ = ["Embedder"]

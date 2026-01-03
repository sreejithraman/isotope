# src/isotopedb/embedder/__init__.py
"""Embedding functionality for Isotope."""

from isotopedb.embedder.base import Embedder

try:
    from isotopedb.embedder.litellm_embedder import LiteLLMEmbedder
except ImportError:
    from isotopedb._optional import _create_missing_dependency_class

    LiteLLMEmbedder = _create_missing_dependency_class(  # type: ignore[misc,assignment]
        "LiteLLMEmbedder", "litellm"
    )

__all__ = ["Embedder", "LiteLLMEmbedder"]

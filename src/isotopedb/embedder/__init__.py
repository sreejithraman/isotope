# src/isotopedb/embedder/__init__.py
"""Embedding functionality for IsotopeDB."""

from isotopedb.embedder.base import Embedder
from isotopedb.embedder.client import ClientEmbedder

__all__ = ["Embedder", "ClientEmbedder"]

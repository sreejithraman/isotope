# src/isotopedb/dedup/__init__.py
"""Deduplication for Isotope."""

from isotopedb.dedup.base import Deduplicator
from isotopedb.dedup.strategies import NoDedup, SourceAwareDedup

__all__ = ["Deduplicator", "NoDedup", "SourceAwareDedup"]

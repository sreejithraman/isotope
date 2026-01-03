# src/isotopedb/atomizer/__init__.py
"""Atomization functionality for IsotopeDB.

This module exports:
- Atomizer: Abstract base class for atomizers
- SentenceAtomizer: Simple sentence-based atomizer (no LLM required)

For the LiteLLM-based atomizer, use:
    from isotopedb.litellm import LiteLLMAtomizer
"""

from isotopedb.atomizer.base import Atomizer
from isotopedb.atomizer.sentence import SentenceAtomizer

__all__ = ["Atomizer", "SentenceAtomizer"]

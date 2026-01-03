# src/isotopedb/atomizer/__init__.py
"""Atomization functionality for IsotopeDB.

This module exports:
- Atomizer: Abstract base class for atomizers
- SentenceAtomizer: Simple sentence-based atomizer (no LLM required)
- LLMAtomizer: LLM-based atomizer (requires LLMClient)

Example:
    from isotopedb.providers.litellm import LiteLLMClient
    from isotopedb.atomizer import LLMAtomizer

    client = LiteLLMClient(model="openai/gpt-4o")
    atomizer = LLMAtomizer(llm_client=client)
"""

from isotopedb.atomizer.base import Atomizer
from isotopedb.atomizer.llm import LLMAtomizer
from isotopedb.atomizer.sentence import SentenceAtomizer

__all__ = ["Atomizer", "SentenceAtomizer", "LLMAtomizer"]

# src/isotope/atomizer/__init__.py
"""Atomization functionality for Isotope.

This module exports:
- Atomizer: Abstract base class for atomizers
- SentenceAtomizer: Simple sentence-based atomizer (no LLM required)
- LLMAtomizer: LLM-based atomizer (requires LLMClient)

Example:
    from isotope.providers.litellm import LiteLLMClient
    from isotope.atomizer import LLMAtomizer

    client = LiteLLMClient(model="openai/gpt-5-mini-2025-08-07")
    atomizer = LLMAtomizer(llm_client=client)
"""

from isotope.atomizer.base import Atomizer
from isotope.atomizer.llm import LLMAtomizer
from isotope.atomizer.sentence import SentenceAtomizer

__all__ = ["Atomizer", "SentenceAtomizer", "LLMAtomizer"]

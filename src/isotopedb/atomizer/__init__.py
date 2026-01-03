# src/isotopedb/atomizer/__init__.py
"""Atomization for Isotope."""

from isotopedb.atomizer.base import Atomizer
from isotopedb.atomizer.sentence import SentenceAtomizer

try:
    from isotopedb.atomizer.llm import LiteLLMAtomizer
except ImportError:
    from isotopedb._optional import _create_missing_dependency_class

    LiteLLMAtomizer = _create_missing_dependency_class(  # type: ignore[misc,assignment]
        "LiteLLMAtomizer", "litellm"
    )

# Backwards compatibility alias (deprecated)
LLMAtomizer = LiteLLMAtomizer

__all__ = ["Atomizer", "SentenceAtomizer", "LiteLLMAtomizer", "LLMAtomizer"]

# src/isotopedb/generator/__init__.py
"""Question generation functionality for IsotopeDB.

This module exports:
- QuestionGenerator: Abstract base class for question generators
- DiversityFilter: Filter for removing similar questions
- FilterScope: Type for diversity filter scope options

For the LiteLLM implementation, use:
    from isotopedb.litellm import LiteLLMGenerator
"""

from isotopedb.generator.base import QuestionGenerator
from isotopedb.generator.diversity_filter import DiversityFilter, FilterScope

__all__ = [
    "QuestionGenerator",
    "DiversityFilter",
    "FilterScope",
]

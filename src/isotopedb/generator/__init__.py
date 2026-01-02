# src/isotopedb/generator/__init__.py
"""Question generation for Isotope."""

from isotopedb.generator.base import QuestionGenerator
from isotopedb.generator.diversity_filter import DiversityFilter

try:
    from isotopedb.generator.question_generator import LiteLLMQuestionGenerator
except ImportError:
    from isotopedb._optional import _create_missing_dependency_class

    LiteLLMQuestionGenerator = _create_missing_dependency_class(
        "LiteLLMQuestionGenerator", "litellm"
    )

__all__ = ["QuestionGenerator", "LiteLLMQuestionGenerator", "DiversityFilter"]

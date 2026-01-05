# src/isotope/question_generator/exceptions.py
"""Exceptions for question generation."""

from isotope.models import Question


class BatchGenerationError(Exception):
    """Raised when batch question generation has too many failures.

    Attributes:
        partial_results: Questions that were successfully generated.
        errors: List of (index, exception) tuples for failed atoms.
    """

    def __init__(
        self,
        message: str,
        partial_results: list[Question],
        errors: list[tuple[int, BaseException]],
    ) -> None:
        super().__init__(message)
        self.partial_results = partial_results
        self.errors = errors

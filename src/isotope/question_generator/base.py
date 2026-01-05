# src/isotope/question_generator/base.py
"""QuestionGenerator abstract base class."""

import asyncio
from abc import ABC, abstractmethod

from isotope.models import Atom, Question


class QuestionGenerator(ABC):
    """Abstract base class for question generation."""

    @abstractmethod
    def generate(self, atom: Atom, chunk_content: str = "") -> list[Question]:
        """Generate questions for a single atom."""
        ...

    async def agenerate(self, atom: Atom, chunk_content: str = "") -> list[Question]:
        """Generate questions for a single atom (async).

        Default implementation calls sync generate() for backwards compatibility.
        Override in subclasses for true async behavior.
        """
        return self.generate(atom, chunk_content)

    async def agenerate_batch(
        self,
        atoms: list[Atom],
        chunk_contents: list[str] | None = None,
        max_concurrent: int = 10,
    ) -> list[Question]:
        """Generate questions for multiple atoms concurrently (async).

        Args:
            atoms: List of atoms to generate questions for.
            chunk_contents: Optional list of chunk contexts (one per atom).
                           If None, uses empty string for all.
            max_concurrent: Maximum concurrent requests. Default 10.

        Returns:
            List of all generated questions.
        """
        if not atoms:
            return []

        chunk_contents = chunk_contents or [""] * len(atoms)
        if len(chunk_contents) != len(atoms):
            raise ValueError(
                f"chunk_contents length ({len(chunk_contents)}) must match "
                f"atoms length ({len(atoms)})"
            )

        semaphore = asyncio.Semaphore(max_concurrent)

        async def generate_with_limit(atom: Atom, chunk_content: str) -> list[Question]:
            async with semaphore:
                return await self.agenerate(atom, chunk_content)

        tasks = [
            generate_with_limit(atom, chunk_content)
            for atom, chunk_content in zip(atoms, chunk_contents, strict=True)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_questions: list[Question] = []
        errors: list[tuple[int, BaseException]] = []
        for i, result in enumerate(results):
            if isinstance(result, BaseException):
                errors.append((i, result))
                continue
            all_questions.extend(result)

        # Raise if more than 50% failed
        if errors and len(errors) / len(atoms) > 0.5:
            from isotope.question_generator.exceptions import BatchGenerationError

            raise BatchGenerationError(
                f"Too many failures: {len(errors)}/{len(atoms)} atoms failed",
                partial_results=all_questions,
                errors=errors,
            )

        return all_questions

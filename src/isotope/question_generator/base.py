# src/isotope/question_generator/base.py
"""QuestionGenerator abstract base class with batch-first architecture."""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass

from isotope.models import Atom, Question


@dataclass
class BatchConfig:
    """Configuration for batch question generation.

    Attributes:
        batch_size: Number of atoms to process per LLM prompt.
            Higher values reduce API calls but increase prompt size.
            Default 1 (single-atom prompts) for cloud provider compatibility.
        max_concurrent: Maximum concurrent batch operations.
            Controls parallelism at the batch level.
            Default 10 for cloud APIs with high rate limits.
    """

    batch_size: int = 1
    max_concurrent: int = 10


class QuestionGenerator(ABC):
    """Abstract base class for question generation.

    This is a batch-first API: implementations must implement the batch methods,
    and single-atom methods are provided as convenience wrappers.

    Implementations should use one of the provided mixins:
    - AsyncOnlyGeneratorMixin: For async-first implementations (most common)
    - SyncOnlyGeneratorMixin: For sync-only implementations

    Example:
        class MyGenerator(AsyncOnlyGeneratorMixin, QuestionGenerator):
            async def agenerate_batch(self, atoms, chunk_contents=None, config=None):
                # Implementation here
                ...
    """

    @abstractmethod
    def generate_batch(
        self,
        atoms: list[Atom],
        chunk_contents: list[str] | None = None,
        config: BatchConfig | None = None,
    ) -> list[Question]:
        """Generate questions for multiple atoms.

        This is the primary sync method. Implementations should override this
        or use AsyncOnlyGeneratorMixin to delegate to agenerate_batch().

        Args:
            atoms: List of atoms to generate questions for.
            chunk_contents: Optional chunk context for each atom (same length as atoms).
                If None, empty string is used for all.
            config: Optional batch configuration. If None, uses default BatchConfig.

        Returns:
            List of all generated questions.

        Raises:
            BatchGenerationError: If too many atoms fail (>50% by default).
        """
        ...

    @abstractmethod
    async def agenerate_batch(
        self,
        atoms: list[Atom],
        chunk_contents: list[str] | None = None,
        config: BatchConfig | None = None,
    ) -> list[Question]:
        """Generate questions for multiple atoms (async).

        This is the primary async method. Implementations should override this
        or use SyncOnlyGeneratorMixin to delegate to generate_batch().

        Args:
            atoms: List of atoms to generate questions for.
            chunk_contents: Optional chunk context for each atom.
            config: Optional batch configuration.

        Returns:
            List of all generated questions.

        Raises:
            BatchGenerationError: If too many atoms fail (>50% by default).
        """
        ...

    # Convenience methods - single atom delegates to batch

    def generate(self, atom: Atom, chunk_content: str = "") -> list[Question]:
        """Generate questions for a single atom.

        Convenience method that delegates to generate_batch().
        """
        return self.generate_batch([atom], [chunk_content])

    async def agenerate(self, atom: Atom, chunk_content: str = "") -> list[Question]:
        """Generate questions for a single atom (async).

        Convenience method that delegates to agenerate_batch().
        """
        return await self.agenerate_batch([atom], [chunk_content])


class SyncOnlyGeneratorMixin:
    """Mixin for generators that only have sync implementation.

    Provides agenerate_batch() that calls generate_batch() synchronously.
    Use this when your implementation doesn't benefit from async (e.g., CPU-bound).
    """

    async def agenerate_batch(
        self,
        atoms: list[Atom],
        chunk_contents: list[str] | None = None,
        config: BatchConfig | None = None,
    ) -> list[Question]:
        """Run sync generate_batch() for async callers."""
        return self.generate_batch(atoms, chunk_contents, config)  # type: ignore[attr-defined]


class AsyncOnlyGeneratorMixin:
    """Mixin for generators that only have async implementation.

    Provides generate_batch() that runs agenerate_batch() with asyncio.run().
    Use this for I/O-bound implementations like API calls.
    """

    def generate_batch(
        self,
        atoms: list[Atom],
        chunk_contents: list[str] | None = None,
        config: BatchConfig | None = None,
    ) -> list[Question]:
        """Run async agenerate_batch() for sync callers."""
        return asyncio.run(
            self.agenerate_batch(atoms, chunk_contents, config)  # type: ignore[attr-defined]
        )

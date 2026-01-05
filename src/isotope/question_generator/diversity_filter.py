# src/isotope/question_generator/diversity_filter.py
"""Question diversity filtering to remove near-duplicates."""

from collections import defaultdict
from typing import Literal

import numpy as np

from isotope.models import EmbeddedQuestion

# Scope for diversity filtering.
#
# Trade-offs:
#   - "global": Best retrieval quality (paper-validated), O(N²) complexity.
#       Catches all duplicates across entire corpus.
#   - "per_chunk": ~100x faster, O(C × Q²) where C=chunks.
#       May return similar questions from different chunks.
#   - "per_atom": ~1000x faster, only deduplicates within each atom's questions.
#       Most duplicates retained across atoms.
FilterScope = Literal["global", "per_chunk", "per_atom"]


class DiversityFilter:
    """Filters out questions that are too similar to each other.

    Uses cosine similarity between embeddings to identify and remove
    near-duplicate questions, ensuring diverse question coverage.
    """

    def __init__(self, threshold: float = 0.85) -> None:
        """Initialize the diversity filter.

        Args:
            threshold: Cosine similarity threshold (0.0 to 1.0).
                      Questions with similarity >= threshold are considered duplicates.
                      Default 0.85 removes very similar questions while keeping
                      semantically distinct ones.
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        self.threshold = threshold

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors.

        Cosine similarity measures the angle between two vectors:
        - 1.0 = identical direction (parallel)
        - 0.0 = perpendicular (orthogonal)
        - -1.0 = opposite direction (anti-parallel)

        For text embeddings, values are typically 0 to 1 (modern embedding
        models like OpenAI, Cohere, Gemini rarely produce negative similarities):
        - 0.9+ = near-duplicates / paraphrases
        - 0.7-0.9 = same topic, different angle
        - 0.5-0.7 = loosely related
        - <0.5 = unrelated

        Formula: cos(θ) = (a · b) / (||a|| * ||b||)
        where · is dot product and ||x|| is the L2 norm (magnitude)
        """
        a_arr, b_arr = np.asarray(a), np.asarray(b)

        # Compute L2 norms (magnitude/length of each vector)
        norm_a, norm_b = np.linalg.norm(a_arr), np.linalg.norm(b_arr)

        # Zero vectors have no direction, so similarity is undefined - return 0
        if norm_a == 0 or norm_b == 0:
            return 0.0

        # dot(a,b) projects vectors onto each other; dividing by norms normalizes to [-1, 1]
        return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))

    def filter(self, questions: list[EmbeddedQuestion]) -> list[EmbeddedQuestion]:
        """Filter out near-duplicate questions.

        Uses a greedy approach: iterate through questions in order,
        keeping each one only if it's sufficiently different from
        all previously kept questions.

        Args:
            questions: List of embedded questions to filter

        Returns:
            Filtered list with near-duplicates removed
        """
        if not questions:
            return []

        kept: list[EmbeddedQuestion] = []

        for eq in questions:
            is_diverse = True
            for kept_eq in kept:
                similarity = self._cosine_similarity(eq.embedding, kept_eq.embedding)
                if similarity >= self.threshold:
                    is_diverse = False
                    break

            if is_diverse:
                kept.append(eq)

        return kept

    def filter_by_scope(
        self,
        questions: list[EmbeddedQuestion],
        scope: FilterScope = "global",
    ) -> list[EmbeddedQuestion]:
        """Filter questions with configurable scope.

        Args:
            questions: List of embedded questions to filter
            scope: Filtering scope (see FilterScope for trade-offs)

        Returns:
            Filtered list with near-duplicates removed within the specified scope
        """
        if not questions or scope == "global":
            return self.filter(questions)

        # Group questions by the appropriate key
        def get_chunk_id(eq: EmbeddedQuestion) -> str:
            return eq.question.chunk_id

        def get_atom_id(eq: EmbeddedQuestion) -> str:
            return eq.question.atom_id

        key_fn = get_chunk_id if scope == "per_chunk" else get_atom_id

        groups: dict[str, list[EmbeddedQuestion]] = defaultdict(list)
        for eq in questions:
            groups[key_fn(eq)].append(eq)

        # Filter each group independently
        filtered: list[EmbeddedQuestion] = []
        for group_questions in groups.values():
            filtered.extend(self.filter(group_questions))

        return filtered

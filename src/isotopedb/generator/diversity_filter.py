# src/isotopedb/generator/diversity_filter.py
"""Question diversity filtering to remove near-duplicates."""

import numpy as np

from isotopedb.models import EmbeddedQuestion


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

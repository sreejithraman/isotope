# src/isotope/atomizer/sentence.py
"""Sentence-based atomizer implementation."""

import pysbd

from isotope.atomizer.base import Atomizer
from isotope.models import Atom, Chunk


class SentenceAtomizer(Atomizer):
    """Atomizer that splits chunks into sentences.

    This is the "structured" atomization approach from the paper.
    Each sentence becomes a separate atom.

    Uses pySBD (Python Sentence Boundary Disambiguation) for robust
    sentence splitting that handles abbreviations, decimals, and
    other edge cases.
    """

    def __init__(self, min_length: int = 10, language: str = "en") -> None:
        """Initialize the sentence atomizer.

        Args:
            min_length: Minimum character length for a sentence to be kept.
                        Filters out very short fragments.
            language: Language code for sentence segmentation (default: "en").
                      Supports 22 languages including en, de, fr, es, etc.
        """
        self.min_length = min_length
        self.segmenter = pysbd.Segmenter(language=language, clean=False)

    def atomize(self, chunk: Chunk) -> list[Atom]:
        """Split a chunk into sentence-based atoms."""
        content = chunk.content.strip()
        if not content:
            return []

        # Use pySBD for robust sentence boundary detection
        sentences = self.segmenter.segment(content)

        atoms: list[Atom] = []
        for sentence in sentences:
            sentence = sentence.strip()
            # Skip empty or too-short sentences
            if len(sentence) >= self.min_length:
                atoms.append(
                    Atom(
                        content=sentence,
                        chunk_id=chunk.id,
                        index=len(atoms),  # Contiguous index based on added atoms
                    )
                )

        return atoms

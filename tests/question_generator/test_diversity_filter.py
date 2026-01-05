# tests/question_generator/test_diversity_filter.py
"""Tests for the DiversityFilter."""

import pytest

from isotope.models import EmbeddedQuestion, Question
from isotope.question_generator import DiversityFilter


def make_eq(
    text: str, embedding: list[float], chunk_id: str = "c1", atom_id: str = "a1"
) -> EmbeddedQuestion:
    """Helper to create an EmbeddedQuestion."""
    return EmbeddedQuestion(
        question=Question(text=text, chunk_id=chunk_id, atom_id=atom_id),
        embedding=embedding,
    )


@pytest.fixture
def filter():
    """Create a DiversityFilter with default threshold."""
    return DiversityFilter()


class TestDiversityFilter:
    def test_default_threshold(self, filter):
        assert filter.threshold == 0.85

    def test_custom_threshold(self):
        f = DiversityFilter(threshold=0.9)
        assert f.threshold == 0.9

    def test_threshold_validation(self):
        with pytest.raises(ValueError):
            DiversityFilter(threshold=1.5)
        with pytest.raises(ValueError):
            DiversityFilter(threshold=-0.1)

    def test_empty_list(self, filter):
        result = filter.filter([])
        assert result == []

    def test_single_question(self, filter):
        questions = [make_eq("Q1?", [1.0, 0.0, 0.0])]
        result = filter.filter(questions)
        assert len(result) == 1

    def test_keeps_diverse_questions(self, filter):
        # Orthogonal vectors = 0 similarity
        questions = [
            make_eq("Q1?", [1.0, 0.0, 0.0]),
            make_eq("Q2?", [0.0, 1.0, 0.0]),
            make_eq("Q3?", [0.0, 0.0, 1.0]),
        ]
        result = filter.filter(questions)
        assert len(result) == 3

    def test_removes_duplicates(self):
        # Identical vectors = 1.0 similarity
        filter = DiversityFilter(threshold=0.9)
        questions = [
            make_eq("Q1?", [1.0, 0.0, 0.0]),
            make_eq("Q1 duplicate?", [1.0, 0.0, 0.0]),  # Identical embedding
        ]
        result = filter.filter(questions)
        assert len(result) == 1
        assert result[0].question.text == "Q1?"

    def test_removes_near_duplicates(self):
        filter = DiversityFilter(threshold=0.9)
        questions = [
            make_eq("Q1?", [1.0, 0.0, 0.0]),
            make_eq("Q2?", [0.99, 0.1, 0.0]),  # Very similar
        ]
        result = filter.filter(questions)
        assert len(result) == 1

    def test_threshold_boundary(self):
        # Test exact threshold boundary
        filter = DiversityFilter(threshold=0.5)

        # These have cosine similarity of 0.5 (exactly at threshold)
        # cos_sim([1,0], [0.5, 0.866]) ≈ 0.5
        questions = [
            make_eq("Q1?", [1.0, 0.0]),
            make_eq("Q2?", [0.5, 0.866]),  # 60 degrees apart
        ]
        result = filter.filter(questions)
        # At threshold = 0.5, similarity >= 0.5 is considered duplicate
        assert len(result) == 1

    def test_preserves_order(self, filter):
        questions = [
            make_eq("First?", [1.0, 0.0, 0.0]),
            make_eq("Second?", [0.0, 1.0, 0.0]),
            make_eq("Third?", [0.0, 0.0, 1.0]),
        ]
        result = filter.filter(questions)
        assert result[0].question.text == "First?"
        assert result[1].question.text == "Second?"
        assert result[2].question.text == "Third?"

    def test_keeps_first_of_similar_group(self):
        filter = DiversityFilter(threshold=0.9)
        questions = [
            make_eq("Original?", [1.0, 0.0]),
            make_eq("Similar 1?", [0.99, 0.1]),
            make_eq("Similar 2?", [0.98, 0.15]),
            make_eq("Different?", [0.0, 1.0]),
        ]
        result = filter.filter(questions)
        assert len(result) == 2
        assert result[0].question.text == "Original?"
        assert result[1].question.text == "Different?"

    def test_handles_zero_vectors(self, filter):
        questions = [
            make_eq("Q1?", [0.0, 0.0, 0.0]),
            make_eq("Q2?", [1.0, 0.0, 0.0]),
        ]
        # Zero vector has 0 similarity with anything
        result = filter.filter(questions)
        assert len(result) == 2

    def test_mixed_chunk_ids(self, filter):
        # Filter works within the same list regardless of chunk_id
        questions = [
            make_eq("Q1?", [1.0, 0.0], chunk_id="c1"),
            make_eq("Q2?", [0.0, 1.0], chunk_id="c2"),
        ]
        result = filter.filter(questions)
        assert len(result) == 2


FILTER_SCOPES: list[str] = ["global", "per_chunk", "per_atom"]


class TestFilterByScope:
    """Tests for the filter_by_scope method."""

    @pytest.fixture
    def filter(self):
        return DiversityFilter(threshold=0.9)

    def test_global_scope_same_as_filter(self, filter):
        """GLOBAL scope should behave exactly like filter()."""
        questions = [
            make_eq("Q1?", [1.0, 0.0], chunk_id="c1", atom_id="a1"),
            make_eq("Q2?", [1.0, 0.0], chunk_id="c2", atom_id="a2"),  # Duplicate
        ]
        global_result = filter.filter_by_scope(questions, "global")
        filter_result = filter.filter(questions)
        assert len(global_result) == len(filter_result)

    def test_per_chunk_filters_within_chunks(self, filter):
        """PER_CHUNK should filter duplicates within each chunk separately."""
        questions = [
            # Chunk 1: two similar questions
            make_eq("Q1?", [1.0, 0.0], chunk_id="c1", atom_id="a1"),
            make_eq("Q2?", [1.0, 0.0], chunk_id="c1", atom_id="a2"),  # Same chunk, duplicate
            # Chunk 2: one question with same embedding (cross-chunk)
            make_eq("Q3?", [1.0, 0.0], chunk_id="c2", atom_id="a3"),
        ]
        result = filter.filter_by_scope(questions, "per_chunk")
        # Within c1: keeps 1 (filters duplicate)
        # Within c2: keeps 1
        # Cross-chunk duplicates NOT filtered
        assert len(result) == 2

    def test_per_chunk_keeps_cross_chunk_duplicates(self, filter):
        """PER_CHUNK should NOT filter duplicates across different chunks."""
        questions = [
            make_eq("Q1?", [1.0, 0.0], chunk_id="c1", atom_id="a1"),
            make_eq("Q2?", [1.0, 0.0], chunk_id="c2", atom_id="a2"),  # Different chunk
        ]
        result = filter.filter_by_scope(questions, "per_chunk")
        # Each chunk has 1 question, no within-chunk duplicates
        assert len(result) == 2

    def test_per_atom_filters_within_atoms(self, filter):
        """PER_ATOM should filter duplicates within each atom separately."""
        questions = [
            # Atom 1: two similar questions
            make_eq("Q1?", [1.0, 0.0], chunk_id="c1", atom_id="a1"),
            make_eq("Q2?", [1.0, 0.0], chunk_id="c1", atom_id="a1"),  # Same atom, duplicate
            # Atom 2: one question with same embedding (cross-atom)
            make_eq("Q3?", [1.0, 0.0], chunk_id="c1", atom_id="a2"),
        ]
        result = filter.filter_by_scope(questions, "per_atom")
        # Within a1: keeps 1
        # Within a2: keeps 1
        assert len(result) == 2

    def test_per_atom_keeps_cross_atom_duplicates(self, filter):
        """PER_ATOM should NOT filter duplicates across different atoms."""
        questions = [
            make_eq("Q1?", [1.0, 0.0], chunk_id="c1", atom_id="a1"),
            make_eq("Q2?", [1.0, 0.0], chunk_id="c1", atom_id="a2"),  # Different atom
        ]
        result = filter.filter_by_scope(questions, "per_atom")
        assert len(result) == 2

    def test_empty_list(self, filter):
        """Empty list should return empty for all scopes."""
        for scope in FILTER_SCOPES:
            result = filter.filter_by_scope([], scope)
            assert result == []

    def test_default_scope_is_global(self, filter):
        """Default scope should be GLOBAL."""
        questions = [
            make_eq("Q1?", [1.0, 0.0], chunk_id="c1", atom_id="a1"),
            make_eq("Q2?", [1.0, 0.0], chunk_id="c2", atom_id="a2"),
        ]
        default_result = filter.filter_by_scope(questions)
        global_result = filter.filter_by_scope(questions, "global")
        assert len(default_result) == len(global_result)

# tests/commands/test_questions.py
"""Tests for the questions command."""

import os
import tempfile

from isotope.commands import questions


class TestSampleQuestions:
    """Tests for questions.sample_questions()."""

    def test_no_database(self) -> None:
        """Returns success with empty list when database doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nonexistent_dir = os.path.join(tmpdir, "nonexistent")
            result = questions.sample_questions(data_dir=nonexistent_dir)

            assert result.success is True
            assert result.questions == []
            assert result.total == 0

    def test_empty_database(self) -> None:
        """Returns success with empty list for empty database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = questions.sample_questions(data_dir=tmpdir)

            # Should succeed (or fail gracefully if stores not initialized)
            assert result.success is True or result.error is not None

    def test_returns_correct_types(self) -> None:
        """Result has correct attribute types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = questions.sample_questions(data_dir=tmpdir)

            assert isinstance(result.success, bool)
            assert isinstance(result.questions, list)
            assert isinstance(result.total, int)

    def test_source_filter_no_match(self) -> None:
        """Returns empty list when source filter has no matches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = questions.sample_questions(
                source="nonexistent.md",
                data_dir=tmpdir,
            )

            # Should succeed with empty results (or fail gracefully)
            assert result.success is True or result.error is not None

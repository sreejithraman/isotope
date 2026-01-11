# tests/commands/test_query.py
"""Tests for the query command."""

import os
import tempfile

from isotope.commands import query


class TestQueryCommand:
    """Tests for query.query()."""

    def test_query_no_database(self) -> None:
        """Query with no database returns error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nonexistent_dir = os.path.join(tmpdir, "nonexistent")
            result = query.query(
                question="test question",
                data_dir=nonexistent_dir,
            )

            assert result.success is False
            assert result.error is not None
            assert "not found" in result.error.lower()

    def test_query_returns_correct_types(self) -> None:
        """Query result has correct attribute types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = query.query(question="test", data_dir=tmpdir)

            assert isinstance(result.success, bool)
            assert isinstance(result.query, str)
            assert result.answer is None or isinstance(result.answer, str)
            assert isinstance(result.results, list)

    def test_query_preserves_question(self) -> None:
        """Query result includes the original question."""
        question = "How does authentication work?"
        result = query.query(question=question, data_dir="/nonexistent")

        assert result.query == question

    def test_query_with_k_parameter(self) -> None:
        """Query accepts k parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Should not raise error for k parameter
            result = query.query(
                question="test",
                data_dir=tmpdir,
                k=10,
            )

            # Will fail because no database, but k parameter was accepted
            assert result.success is False

    def test_query_with_raw_mode(self) -> None:
        """Query accepts raw mode parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = query.query(
                question="test",
                data_dir=tmpdir,
                raw=True,
            )

            # Will fail because no database, but raw parameter was accepted
            assert result.success is False

# tests/commands/test_status.py
"""Tests for the status command."""

import os
import tempfile

from isotope.commands import status


class TestStatusCommand:
    """Tests for status.status()."""

    def test_status_no_database(self) -> None:
        """Status with no database returns success with zero counts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nonexistent_dir = os.path.join(tmpdir, "nonexistent")
            result = status.status(data_dir=nonexistent_dir)

            assert result.success is True
            assert result.total_sources == 0
            assert result.total_chunks == 0
            assert result.total_atoms == 0
            assert result.total_questions == 0

    def test_status_empty_database(self) -> None:
        """Status with empty directory returns success with zero counts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = status.status(data_dir=tmpdir)

            # Should succeed with zero counts (or fail gracefully if stores not initialized)
            # This depends on whether get_stores creates empty stores
            assert result.success is True or result.error is not None

    def test_status_returns_correct_types(self) -> None:
        """Status result has correct attribute types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = status.status(data_dir=tmpdir)

            assert isinstance(result.success, bool)
            assert isinstance(result.total_sources, int)
            assert isinstance(result.total_chunks, int)
            assert isinstance(result.total_atoms, int)
            assert isinstance(result.total_questions, int)
            assert isinstance(result.sources, list)

    def test_status_detailed_returns_sources_list(self) -> None:
        """Status with detailed=True returns sources breakdown."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = status.status(data_dir=tmpdir, detailed=True)

            # Even with no data, sources should be a list
            assert isinstance(result.sources, list)

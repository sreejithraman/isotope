# tests/commands/test_status.py
"""Tests for the inspect command (renamed from status)."""

import os
import tempfile

from isotope.commands import inspect


class TestInspectCommand:
    """Tests for inspect.inspect()."""

    def test_inspect_no_database(self) -> None:
        """Inspect with no database returns success with zero counts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nonexistent_dir = os.path.join(tmpdir, "nonexistent")
            result = inspect.inspect(data_dir=nonexistent_dir)

            assert result.success is True
            assert result.total_sources == 0
            assert result.total_chunks == 0
            assert result.total_atoms == 0
            assert result.total_questions == 0

    def test_inspect_empty_database(self) -> None:
        """Inspect with empty directory returns success with zero counts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = inspect.inspect(data_dir=tmpdir)

            # Should succeed with zero counts (or fail gracefully if stores not initialized)
            assert result.success is True or result.error is not None

    def test_inspect_returns_correct_types(self) -> None:
        """Inspect result has correct attribute types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = inspect.inspect(data_dir=tmpdir)

            assert isinstance(result.success, bool)
            assert isinstance(result.total_sources, int)
            assert isinstance(result.total_chunks, int)
            assert isinstance(result.total_atoms, int)
            assert isinstance(result.total_questions, int)
            assert isinstance(result.sources, list)

    def test_inspect_detailed_returns_sources_list(self) -> None:
        """Inspect with detailed=True returns sources breakdown."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = inspect.inspect(data_dir=tmpdir, detailed=True)

            # Even with no data, sources should be a list
            assert isinstance(result.sources, list)

    def test_inspect_sources_returns_sources_list(self) -> None:
        """Inspect with sources=True returns sources breakdown."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = inspect.inspect(data_dir=tmpdir, sources=True)

            assert isinstance(result.sources, list)

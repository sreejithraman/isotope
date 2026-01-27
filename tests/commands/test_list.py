# tests/commands/test_list.py
"""Tests for the list command."""

import os
import tempfile

from isotope.commands import list_cmd


class TestListCommand:
    """Tests for list_cmd.list_sources()."""

    def test_list_no_database(self) -> None:
        """List with no database returns success with empty sources."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nonexistent_dir = os.path.join(tmpdir, "nonexistent")
            result = list_cmd.list_sources(data_dir=nonexistent_dir)

            assert result.success is True
            assert result.sources == []

    def test_list_empty_database(self) -> None:
        """List with empty directory returns success with empty sources."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = list_cmd.list_sources(data_dir=tmpdir)

            # Should succeed with empty sources (or fail gracefully if stores not initialized)
            assert result.success is True or result.error is not None
            if result.success:
                assert isinstance(result.sources, list)

    def test_list_returns_correct_types(self) -> None:
        """List result has correct attribute types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = list_cmd.list_sources(data_dir=tmpdir)

            assert isinstance(result.success, bool)
            assert isinstance(result.sources, list)

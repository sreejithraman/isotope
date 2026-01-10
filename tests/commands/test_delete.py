# tests/commands/test_delete.py
"""Tests for the delete command."""

import os
import tempfile

from isotope.commands import delete


class TestDeleteCommand:
    """Tests for delete.delete()."""

    def test_delete_no_database(self) -> None:
        """Delete with no database returns error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nonexistent_dir = os.path.join(tmpdir, "nonexistent")
            result = delete.delete(source="test.md", data_dir=nonexistent_dir)

            assert result.success is False
            assert result.error is not None
            assert "No database found" in result.error

    def test_delete_nonexistent_source(self) -> None:
        """Delete nonexistent source returns error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a directory that exists but has no matching source
            result = delete.delete(source="nonexistent.md", data_dir=tmpdir)

            # Should fail either because database doesn't exist or source not found
            assert result.success is False
            assert result.error is not None

    def test_delete_returns_correct_types(self) -> None:
        """Delete result has correct attribute types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = delete.delete(source="test.md", data_dir=tmpdir)

            assert isinstance(result.success, bool)
            assert isinstance(result.source, str)
            assert isinstance(result.chunks_deleted, int)
            assert isinstance(result.atoms_deleted, int)
            assert isinstance(result.questions_deleted, int)

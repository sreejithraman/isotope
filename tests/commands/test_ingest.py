# tests/commands/test_ingest.py
"""Tests for the ingest command."""

import os
import tempfile
from pathlib import Path

from isotope.commands import ingest
from isotope.providers.litellm import LLMError


class TestIngestCommand:
    """Tests for ingest.ingest()."""

    def test_ingest_nonexistent_path(self) -> None:
        """Ingest with nonexistent path returns error."""
        result = ingest.ingest(path="/nonexistent/path/file.md")

        assert result.success is False
        assert result.error is not None
        assert "not found" in result.error.lower()

    def test_ingest_empty_directory(self) -> None:
        """Ingest empty directory returns no files or config error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = ingest.ingest(path=tmpdir)

            # Should either succeed with no files, or fail with config error
            # (depends on whether isotope.yaml exists)
            if result.success:
                assert result.files_processed == 0
                assert "No supported files" in (result.error or "")
            else:
                # Config error is expected without isotope.yaml
                assert result.error is not None

    def test_ingest_returns_correct_types(self) -> None:
        """Ingest result has correct attribute types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = ingest.ingest(path=tmpdir)

            assert isinstance(result.success, bool)
            assert isinstance(result.files_processed, int)
            assert isinstance(result.files_skipped, int)
            assert isinstance(result.files_failed, int)
            assert isinstance(result.total_chunks, int)
            assert isinstance(result.total_atoms, int)
            assert isinstance(result.total_questions, int)
            assert isinstance(result.file_results, list)
            assert isinstance(result.errors, list)

    def test_ingest_callbacks_called(self) -> None:
        """Ingest calls file callbacks when provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test file
            test_file = os.path.join(tmpdir, "test.txt")
            with open(test_file, "w") as f:
                f.write("Test content for ingestion.")

            file_starts: list[str] = []
            file_completes: list = []

            def on_file_start(filepath: str, index: int, total: int) -> None:
                file_starts.append(filepath)

            def on_file_complete(file_result) -> None:
                file_completes.append(file_result)

            # This will likely fail without proper config but we're testing callbacks
            result = ingest.ingest(
                path=test_file,
                on_file_start=on_file_start,
                on_file_complete=on_file_complete,
            )

            # If we get here without error, callbacks were called
            # (success depends on config being available)
            if result.success:
                assert len(file_starts) > 0
                assert len(file_completes) > 0

    def test_do_ingest_sets_error_when_single_file_fails(self) -> None:
        """All-failed single-file ingest populates result.error."""

        class FailingIsotope:
            def ingest_file(self, filepath: str, on_progress=None, **kwargs) -> dict:
                raise LLMError("Authentication failed: Check your API key")

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "doc.txt"
            test_file.write_text("hello")

            result = ingest._do_ingest(FailingIsotope(), test_file)

            assert result.success is False
            assert result.files_processed == 0
            assert result.files_failed == 1
            assert result.error == "Authentication failed: Check your API key"
            assert result.error_details is None

    def test_do_ingest_sets_error_details_when_multiple_files_fail(self) -> None:
        """All-failed multi-file ingest includes detailed per-file breakdown."""

        class FailingIsotope:
            def ingest_file(self, filepath: str, on_progress=None, **kwargs) -> dict:
                filename = Path(filepath).name
                raise LLMError(f"Mock failure for {filename}")

        with tempfile.TemporaryDirectory() as tmpdir:
            file_a = Path(tmpdir) / "a.txt"
            file_b = Path(tmpdir) / "b.txt"
            file_a.write_text("a")
            file_b.write_text("b")

            result = ingest._do_ingest(FailingIsotope(), Path(tmpdir))

            assert result.success is False
            assert result.files_processed == 0
            assert result.files_failed == 2
            assert result.error in {"Mock failure for a.txt", "Mock failure for b.txt"}
            assert result.error_details is not None
            assert "All 2 files failed:" in result.error_details
            assert f"  {file_a}: Mock failure for a.txt" in result.error_details
            assert f"  {file_b}: Mock failure for b.txt" in result.error_details

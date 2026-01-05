# tests/test_cli.py
"""Tests for the CLI."""

import os
import tempfile

import pytest

pytest.importorskip("typer", reason="Tests require typer package (pip install isotope-rag[cli])")

from typer.testing import CliRunner

from isotope.cli import app


@pytest.fixture
def runner():
    return CliRunner()


class TestCliBasics:
    def test_help(self, runner):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "isotope" in result.output.lower() or "Isotope" in result.output

    def test_version(self, runner):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output


class TestConfigCommand:
    def test_config_shows_settings(self, runner):
        result = runner.invoke(app, ["config"])
        assert result.exit_code == 0
        assert "questions_per_atom" in result.output
        assert "diversity_scope" in result.output


class TestIngestCommand:
    def test_ingest_help(self, runner):
        result = runner.invoke(app, ["ingest", "--help"])
        assert result.exit_code == 0
        assert "path" in result.output.lower()

    def test_ingest_nonexistent_file(self, runner):
        result = runner.invoke(app, ["ingest", "/nonexistent/file.md"])
        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "error" in result.output.lower()


class TestQueryCommand:
    def test_query_help(self, runner):
        result = runner.invoke(app, ["query", "--help"])
        assert result.exit_code == 0
        assert "question" in result.output.lower() or "query" in result.output.lower()

    def test_query_no_database(self, runner):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, "data")

            result = runner.invoke(
                app,
                ["query", "test question", "--data-dir", data_dir],
            )
            # Should fail gracefully with no database
            assert result.exit_code != 0 or "not found" in result.output.lower()


class TestListCommand:
    def test_list_help(self, runner):
        result = runner.invoke(app, ["list", "--help"])
        assert result.exit_code == 0

    def test_list_empty_database(self, runner):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir)

            result = runner.invoke(
                app,
                ["list", "--data-dir", data_dir],
            )
            # Should handle empty database gracefully
            assert result.exit_code == 0


class TestStatusCommand:
    def test_status_help(self, runner):
        result = runner.invoke(app, ["status", "--help"])
        assert result.exit_code == 0

    def test_status_empty_database(self, runner):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir)

            result = runner.invoke(
                app,
                ["status", "--data-dir", data_dir],
            )
            assert result.exit_code == 0


class TestDeleteCommand:
    def test_delete_help(self, runner):
        result = runner.invoke(app, ["delete", "--help"])
        assert result.exit_code == 0
        assert "source" in result.output.lower()

    def test_delete_nonexistent_source(self, runner):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir)

            result = runner.invoke(
                app,
                ["delete", "nonexistent.md", "--data-dir", data_dir, "--force"],
            )
            # Should handle gracefully
            assert "not found" in result.output.lower() or result.exit_code == 0

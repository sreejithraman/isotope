# tests/loaders/test_text.py
"""Tests for text/markdown file loader."""

import os
import tempfile

import pytest

from isotope.loaders.base import Loader
from isotope.loaders.text import TextLoader
from isotope.models import Chunk


@pytest.fixture
def temp_dir():
    """Create a temporary directory with test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        with open(os.path.join(tmpdir, "simple.txt"), "w") as f:
            f.write("This is a simple text file.\n\nIt has two paragraphs.")

        with open(os.path.join(tmpdir, "doc.md"), "w") as f:
            f.write("# Title\n\nFirst paragraph.\n\n## Section\n\nSecond paragraph.")

        with open(os.path.join(tmpdir, "empty.txt"), "w") as f:
            f.write("")

        yield tmpdir


class TestTextLoader:
    def test_is_loader(self):
        assert isinstance(TextLoader(), Loader)

    def test_supports_txt(self):
        loader = TextLoader()
        assert loader.supports("file.txt") is True

    def test_supports_md(self):
        loader = TextLoader()
        assert loader.supports("file.md") is True

    def test_supports_markdown(self):
        loader = TextLoader()
        assert loader.supports("file.markdown") is True

    def test_does_not_support_pdf(self):
        loader = TextLoader()
        assert loader.supports("file.pdf") is False

    def test_does_not_support_docx(self):
        loader = TextLoader()
        assert loader.supports("file.docx") is False

    def test_load_text_file(self, temp_dir):
        from pathlib import Path

        loader = TextLoader()
        path = os.path.join(temp_dir, "simple.txt")
        chunks = loader.load(path)

        assert len(chunks) >= 1
        assert all(isinstance(c, Chunk) for c in chunks)
        # Source should be the resolved absolute path
        assert chunks[0].source == str(Path(path).resolve())

    def test_load_markdown_file(self, temp_dir):
        loader = TextLoader()
        chunks = loader.load(os.path.join(temp_dir, "doc.md"))

        assert len(chunks) >= 1
        assert chunks[0].metadata.get("type") == "markdown"

    def test_load_empty_file(self, temp_dir):
        loader = TextLoader()
        chunks = loader.load(os.path.join(temp_dir, "empty.txt"))

        assert chunks == []

    def test_load_nonexistent_file(self):
        loader = TextLoader()
        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/path/file.txt")

    def test_chunks_have_unique_ids(self, temp_dir):
        loader = TextLoader()
        chunks = loader.load(os.path.join(temp_dir, "doc.md"))

        if len(chunks) > 1:
            ids = [c.id for c in chunks]
            assert len(ids) == len(set(ids))

    def test_custom_chunk_size(self, temp_dir):
        # Create a longer file
        with open(os.path.join(temp_dir, "long.txt"), "w") as f:
            f.write("Word " * 1000)  # ~5000 chars

        loader = TextLoader(chunk_size=500, chunk_overlap=50)
        chunks = loader.load(os.path.join(temp_dir, "long.txt"))

        # Should have multiple chunks
        assert len(chunks) > 1

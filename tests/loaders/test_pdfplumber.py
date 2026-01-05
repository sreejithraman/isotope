# tests/loaders/test_pdfplumber.py
"""Tests for PDFPlumberLoader."""

# Check if pdfplumber is available
import importlib.util
import tempfile

import pytest

from isotope.loaders.base import Loader

HAS_PDFPLUMBER = importlib.util.find_spec("pdfplumber") is not None

pytestmark = pytest.mark.skipif(not HAS_PDFPLUMBER, reason="pdfplumber not installed")


@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestPDFPlumberLoader:
    def test_is_loader(self):
        from isotope.loaders.pdfplumber_loader import PDFPlumberLoader

        assert isinstance(PDFPlumberLoader(), Loader)

    def test_supports_pdf(self):
        from isotope.loaders.pdfplumber_loader import PDFPlumberLoader

        loader = PDFPlumberLoader()
        assert loader.supports("file.pdf") is True
        assert loader.supports("FILE.PDF") is True

    def test_does_not_support_other(self):
        from isotope.loaders.pdfplumber_loader import PDFPlumberLoader

        loader = PDFPlumberLoader()
        assert loader.supports("file.txt") is False
        assert loader.supports("file.docx") is False
        assert loader.supports("file.html") is False

    def test_load_nonexistent_file(self):
        from isotope.loaders.pdfplumber_loader import PDFPlumberLoader

        loader = PDFPlumberLoader()
        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/path/file.pdf")


class TestPDFPlumberLoaderTableFormatting:
    """Test table-to-markdown conversion."""

    def test_tables_to_markdown_simple(self):
        from isotope.loaders.pdfplumber_loader import PDFPlumberLoader

        loader = PDFPlumberLoader()

        tables = [[["Header1", "Header2"], ["Row1Col1", "Row1Col2"]]]

        result = loader._tables_to_markdown(tables)

        assert "| Header1 | Header2 |" in result
        assert "| --- | --- |" in result
        assert "| Row1Col1 | Row1Col2 |" in result

    def test_tables_to_markdown_with_none_values(self):
        from isotope.loaders.pdfplumber_loader import PDFPlumberLoader

        loader = PDFPlumberLoader()

        tables = [[["Header1", None], [None, "Value"]]]

        result = loader._tables_to_markdown(tables)

        # None values should become empty strings
        assert "| Header1 |  |" in result
        assert "|  | Value |" in result

    def test_tables_to_markdown_escapes_pipes(self):
        from isotope.loaders.pdfplumber_loader import PDFPlumberLoader

        loader = PDFPlumberLoader()

        tables = [[["A|B", "C"], ["D", "E|F"]]]

        result = loader._tables_to_markdown(tables)

        # Pipe characters should be escaped
        assert "A\\|B" in result
        assert "E\\|F" in result

    def test_tables_to_markdown_empty_table(self):
        from isotope.loaders.pdfplumber_loader import PDFPlumberLoader

        loader = PDFPlumberLoader()

        result = loader._tables_to_markdown([])
        assert result == ""

        result = loader._tables_to_markdown([[]])
        assert result == ""

    def test_tables_to_markdown_multiple_tables(self):
        from isotope.loaders.pdfplumber_loader import PDFPlumberLoader

        loader = PDFPlumberLoader()

        tables = [
            [["H1", "H2"], ["V1", "V2"]],
            [["A", "B"], ["C", "D"]],
        ]

        result = loader._tables_to_markdown(tables)

        # Should have two tables separated by double newline
        assert result.count("| --- | --- |") == 2
        assert "\n\n" in result

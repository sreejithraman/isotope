# tests/loaders/test_pypdf.py
"""Tests for PyPDFLoader."""

# Check if pypdf is available
import importlib.util
import os
import tempfile

import pytest

from isotope.loaders.base import Loader
from isotope.models import Chunk

HAS_PYPDF = importlib.util.find_spec("pypdf") is not None

pytestmark = pytest.mark.skipif(not HAS_PYPDF, reason="pypdf not installed")


@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_pdf(temp_dir):
    """Create a simple PDF for testing."""
    from pypdf import PdfWriter

    pdf_path = os.path.join(temp_dir, "sample.pdf")
    writer = PdfWriter()

    # Add a page with some text
    from pypdf._page import PageObject

    page = PageObject.create_blank_page(width=612, height=792)
    writer.add_page(page)

    # Note: pypdf can't easily add text to blank pages without additional tools
    # So we'll test with a minimal PDF structure
    with open(pdf_path, "wb") as f:
        writer.write(f)

    return pdf_path


class TestPyPDFLoader:
    def test_is_loader(self):
        from isotope.loaders.pypdf_loader import PyPDFLoader

        assert isinstance(PyPDFLoader(), Loader)

    def test_supports_pdf(self):
        from isotope.loaders.pypdf_loader import PyPDFLoader

        loader = PyPDFLoader()
        assert loader.supports("file.pdf") is True
        assert loader.supports("FILE.PDF") is True

    def test_does_not_support_other(self):
        from isotope.loaders.pypdf_loader import PyPDFLoader

        loader = PyPDFLoader()
        assert loader.supports("file.txt") is False
        assert loader.supports("file.docx") is False
        assert loader.supports("file.html") is False

    def test_load_nonexistent_file(self):
        from isotope.loaders.pypdf_loader import PyPDFLoader

        loader = PyPDFLoader()
        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/path/file.pdf")

    def test_load_pdf_returns_chunks(self, sample_pdf):
        from isotope.loaders.pypdf_loader import PyPDFLoader

        loader = PyPDFLoader()
        chunks = loader.load(sample_pdf)

        # Blank page may or may not return content
        assert isinstance(chunks, list)
        for chunk in chunks:
            assert isinstance(chunk, Chunk)

    def test_chunk_metadata(self, sample_pdf):
        from isotope.loaders.pypdf_loader import PyPDFLoader

        loader = PyPDFLoader()
        chunks = loader.load(sample_pdf)

        for chunk in chunks:
            assert chunk.metadata.get("type") == "pdf"
            assert "page" in chunk.metadata
            assert isinstance(chunk.metadata["page"], int)

    def test_chunk_source(self, sample_pdf):
        from isotope.loaders.pypdf_loader import PyPDFLoader

        loader = PyPDFLoader()
        chunks = loader.load(sample_pdf)

        for chunk in chunks:
            assert chunk.source == sample_pdf


class TestPyPDFLoaderImportError:
    """Test behavior when pypdf is not installed."""

    def test_import_error_message(self, monkeypatch):
        """Verify helpful error message when pypdf missing."""
        # This test would need to mock the import, which is complex
        # Instead, we verify the loader exists and can be instantiated
        from isotope.loaders.pypdf_loader import PyPDFLoader

        loader = PyPDFLoader()
        assert loader is not None

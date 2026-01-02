# src/isotopedb/loaders/__init__.py
"""File loaders for Isotope."""

from isotopedb.loaders.base import Loader
from isotopedb.loaders.registry import LoaderRegistry
from isotopedb.loaders.text import TextLoader

# Optional loaders - imported lazily to avoid ImportError when deps not installed
__all__ = ["Loader", "LoaderRegistry", "TextLoader"]


def __getattr__(name: str) -> type:
    """Lazy import optional loaders."""
    if name == "PyPDFLoader":
        from isotopedb.loaders.pypdf_loader import PyPDFLoader

        return PyPDFLoader
    elif name == "PDFPlumberLoader":
        from isotopedb.loaders.pdfplumber_loader import PDFPlumberLoader

        return PDFPlumberLoader
    elif name == "HTMLLoader":
        from isotopedb.loaders.html import HTMLLoader

        return HTMLLoader
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# src/isotope/loaders/__init__.py
"""File loaders for Isotope."""

from isotope.loaders.base import Loader
from isotope.loaders.registry import LoaderRegistry
from isotope.loaders.text import TextLoader

# Optional loaders - imported lazily to avoid ImportError when deps not installed
__all__ = ["Loader", "LoaderRegistry", "TextLoader"]


def __getattr__(name: str) -> type:
    """Lazy import optional loaders."""
    if name == "PyPDFLoader":
        from isotope.loaders.pypdf_loader import PyPDFLoader

        return PyPDFLoader
    elif name == "PDFPlumberLoader":
        from isotope.loaders.pdfplumber_loader import PDFPlumberLoader

        return PDFPlumberLoader
    elif name == "HTMLLoader":
        from isotope.loaders.html import HTMLLoader

        return HTMLLoader
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# src/isotope/loaders/registry.py
"""Loader registry for auto-selecting file loaders."""

from isotope.loaders.base import Loader
from isotope.loaders.text import TextLoader
from isotope.models import Chunk


class LoaderRegistry:
    """Registry for file loaders.

    Automatically selects the appropriate loader based on file extension.
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._loaders: list[Loader] = []

    def register(self, loader: Loader) -> None:
        """Register a loader."""
        self._loaders.append(loader)

    def find_loader(self, path: str) -> Loader | None:
        """Find a loader that supports the given path."""
        for loader in self._loaders:
            if loader.supports(path):
                return loader
        return None

    def load(self, path: str, source_id: str | None = None) -> list[Chunk]:
        """Load a file using the appropriate loader.

        Args:
            path: Path to the file to load
            source_id: Optional custom source identifier. If not provided,
                      the absolute path will be used as the source.

        Returns:
            List of Chunk objects

        Raises:
            ValueError: If no loader supports the file type
        """
        loader = self.find_loader(path)
        if loader is None:
            raise ValueError(f"No loader found for: {path}")
        return loader.load(path, source_id)

    @classmethod
    def default(cls) -> "LoaderRegistry":
        """Create a registry with all default loaders registered.

        Registers TextLoader (always available) plus optional loaders
        for PDF and HTML if their dependencies are installed.

        PDF loader priority: pdfplumber (better tables) > pypdf (lighter)
        """
        registry = cls()
        registry.register(TextLoader())

        # Optional: pdfplumber (preferred for tables)
        try:
            from isotope.loaders.pdfplumber_loader import PDFPlumberLoader

            registry.register(PDFPlumberLoader())
        except ImportError:
            # Fallback: pypdf (lighter)
            try:
                from isotope.loaders.pypdf_loader import PyPDFLoader

                registry.register(PyPDFLoader())
            except ImportError:
                pass

        # Optional: HTML support
        try:
            from isotope.loaders.html import HTMLLoader

            registry.register(HTMLLoader())
        except ImportError:
            pass

        return registry

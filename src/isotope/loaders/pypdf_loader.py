# src/isotope/loaders/pypdf_loader.py
"""PDF loader using pypdf - lightweight, pure Python."""

from pathlib import Path

from isotope.loaders.base import Loader
from isotope.models import Chunk


class PyPDFLoader(Loader):
    """Load PDF files using pypdf.

    A lightweight, pure Python PDF loader suitable for basic text extraction.
    For documents with tables or complex layouts, consider PDFPlumberLoader.

    Requires: pip install isotope-rag[pdf-text]
    """

    SUPPORTED_EXTENSIONS = {".pdf"}

    def supports(self, path: str) -> bool:
        """Check if this loader supports the given file."""
        return Path(path).suffix.lower() in self.SUPPORTED_EXTENSIONS

    def load(self, path: str, source_id: str | None = None) -> list[Chunk]:
        """Load a PDF file and return chunks (one per page).

        Args:
            path: Path to the PDF file
            source_id: Optional custom source identifier. If not provided,
                      the absolute path will be used as the source.

        Returns:
            List of Chunk objects, one per page with content

        Raises:
            ImportError: If pypdf is not installed
            FileNotFoundError: If file does not exist
        """
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError(
                "pypdf is required for PDF text extraction. "
                "Install with: pip install isotope-rag[pdf-text]"
            ) from None

        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        # Use source_id if provided, otherwise use absolute path
        source = source_id or str(file_path.resolve())

        chunks = []
        reader = PdfReader(path)

        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                chunks.append(
                    Chunk(
                        content=text.strip(),
                        source=source,
                        metadata={"type": "pdf", "page": page_num},
                    )
                )

        return chunks

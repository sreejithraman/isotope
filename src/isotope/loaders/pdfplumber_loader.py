# src/isotope/loaders/pdfplumber_loader.py
"""PDF loader using pdfplumber - best for tables and structured content."""

from pathlib import Path
from typing import Any

from isotope.loaders.base import Loader
from isotope.models import Chunk


class PDFPlumberLoader(Loader):
    """Load PDF files using pdfplumber.

    A feature-rich PDF loader with excellent table detection and layout analysis.
    Best for documents with tables, complex layouts, or structured content.

    Requires: pip install isotope-rag[pdf]
    """

    SUPPORTED_EXTENSIONS = {".pdf"}

    def supports(self, path: str) -> bool:
        """Check if this loader supports the given file."""
        return Path(path).suffix.lower() in self.SUPPORTED_EXTENSIONS

    def load(self, path: str, source_id: str | None = None) -> list[Chunk]:
        """Load a PDF file and return chunks (one per page).

        Extracts both text content and tables, converting tables to markdown format.

        Args:
            path: Path to the PDF file
            source_id: Optional custom source identifier. If not provided,
                      the absolute path will be used as the source.

        Returns:
            List of Chunk objects, one per page with content

        Raises:
            ImportError: If pdfplumber is not installed
            FileNotFoundError: If file does not exist
        """
        try:
            import pdfplumber
        except ImportError:
            raise ImportError(
                "pdfplumber is required for full PDF support with table extraction. "
                "Install with: pip install isotope-rag[pdf]"
            ) from None

        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        # Use source_id if provided, otherwise use absolute path
        source = source_id or str(file_path.resolve())

        chunks = []

        with pdfplumber.open(path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                # Extract text
                text = page.extract_text() or ""

                # Extract and format tables as markdown
                tables = page.extract_tables()
                table_md = self._tables_to_markdown(tables) if tables else ""

                # Combine text and tables
                content = f"{text}\n\n{table_md}".strip()

                if content:
                    chunks.append(
                        Chunk(
                            content=content,
                            source=source,
                            metadata={"type": "pdf", "page": page_num},
                        )
                    )

        return chunks

    def _tables_to_markdown(self, tables: list[list[list[Any]]]) -> str:
        """Convert pdfplumber tables to markdown format.

        Args:
            tables: List of tables, where each table is a list of rows,
                    and each row is a list of cell values

        Returns:
            Markdown-formatted tables as a string
        """
        md_tables = []

        for table in tables:
            if not table or not table[0]:
                continue

            # Header row
            header_cells = [str(cell or "").replace("|", "\\|") for cell in table[0]]
            header = "| " + " | ".join(header_cells) + " |"

            # Separator row
            separator = "| " + " | ".join("---" for _ in table[0]) + " |"

            # Data rows
            rows = []
            for row in table[1:]:
                cells = [str(cell or "").replace("|", "\\|") for cell in row]
                rows.append("| " + " | ".join(cells) + " |")

            md_tables.append("\n".join([header, separator] + rows))

        return "\n\n".join(md_tables)

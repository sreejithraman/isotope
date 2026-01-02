# src/isotopedb/loaders/html.py
"""HTML file loader - converts HTML to markdown for LLM consumption."""

from pathlib import Path

from isotopedb.loaders.base import Loader
from isotopedb.models import Chunk


class HTMLLoader(Loader):
    """Load HTML files and convert to markdown.

    Cleans HTML by removing scripts, styles, and navigation elements,
    then converts to markdown format for better LLM consumption.

    Requires: pip install isotopedb[html]
    """

    SUPPORTED_EXTENSIONS = {".html", ".htm"}

    # Tags to remove entirely (including their content)
    REMOVE_TAGS = ["script", "style", "nav", "footer", "header", "aside", "noscript"]

    def supports(self, path: str) -> bool:
        """Check if this loader supports the given file."""
        return Path(path).suffix.lower() in self.SUPPORTED_EXTENSIONS

    def load(self, path: str) -> list[Chunk]:
        """Load an HTML file and return chunks.

        Args:
            path: Path to the HTML file

        Returns:
            List of Chunk objects (typically one for the whole document)

        Raises:
            ImportError: If beautifulsoup4 or markdownify is not installed
            FileNotFoundError: If file does not exist
        """
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError(
                "beautifulsoup4 is required for HTML support. "
                "Install with: pip install isotopedb[html]"
            ) from None

        try:
            from markdownify import markdownify
        except ImportError:
            raise ImportError(
                "markdownify is required for HTML support. "
                "Install with: pip install isotopedb[html]"
            ) from None

        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        content = file_path.read_text(encoding="utf-8")

        if not content.strip():
            return []

        soup = BeautifulSoup(content, "html.parser")

        # Extract title before cleaning
        title = soup.title.string if soup.title else None

        # Remove unwanted tags
        for tag in soup(self.REMOVE_TAGS):
            tag.decompose()

        # Convert to markdown
        md_text = markdownify(str(soup), heading_style="ATX")

        # Clean up excessive whitespace
        md_text = self._clean_markdown(md_text)

        if not md_text.strip():
            return []

        return [
            Chunk(
                content=md_text.strip(),
                source=path,
                metadata={"type": "html", "title": title},
            )
        ]

    def _clean_markdown(self, text: str) -> str:
        """Clean up markdown output.

        Removes excessive blank lines and normalizes whitespace.
        """
        lines = text.split("\n")
        cleaned = []
        prev_blank = False

        for line in lines:
            is_blank = not line.strip()
            if is_blank and prev_blank:
                continue
            cleaned.append(line)
            prev_blank = is_blank

        return "\n".join(cleaned)

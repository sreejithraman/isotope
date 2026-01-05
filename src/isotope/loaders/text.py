# src/isotope/loaders/text.py
"""Text and Markdown file loader."""

from pathlib import Path

from isotope.loaders.base import Loader
from isotope.models import Chunk


class TextLoader(Loader):
    """Load plain text and markdown files into chunks.

    Splits content into chunks based on paragraph boundaries,
    respecting chunk_size limits with overlap.
    """

    SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".text"}

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
    ) -> None:
        """Initialize the text loader.

        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Overlap between adjacent chunks

        Raises:
            ValueError: If chunk_overlap >= chunk_size
        """
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def supports(self, path: str) -> bool:
        """Check if this loader supports the given file."""
        return Path(path).suffix.lower() in self.SUPPORTED_EXTENSIONS

    def load(self, path: str, source_id: str | None = None) -> list[Chunk]:
        """Load a text file and return chunks.

        Args:
            path: Path to the file to load
            source_id: Optional custom source identifier. If not provided,
                      the absolute path will be used as the source.
        """
        file_path = Path(path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        content = file_path.read_text(encoding="utf-8")

        if not content.strip():
            return []

        # Use source_id if provided, otherwise use absolute path
        source = source_id or str(file_path.resolve())

        # Determine file type
        is_markdown = file_path.suffix.lower() in {".md", ".markdown"}

        # Split into chunks
        chunks = self._split_into_chunks(content, source, is_markdown)

        return chunks

    def _split_into_chunks(
        self,
        content: str,
        source: str,
        is_markdown: bool,
    ) -> list[Chunk]:
        """Split content into chunks."""
        # Split on double newlines (paragraphs)
        paragraphs = content.split("\n\n")
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        if not paragraphs:
            return []

        chunks = []
        current_chunk = ""

        for para in paragraphs:
            # If paragraph itself is too long, split it further
            if len(para) > self.chunk_size:
                # Save current chunk if not empty
                if current_chunk:
                    chunks.append(self._create_chunk(current_chunk, source, is_markdown))
                    current_chunk = ""

                # Split long paragraph into smaller pieces
                para_chunks = self._split_long_text(para)
                for i, piece in enumerate(para_chunks):
                    if i < len(para_chunks) - 1:
                        chunks.append(self._create_chunk(piece, source, is_markdown))
                    else:
                        # Last piece becomes the current chunk
                        current_chunk = piece
            # If adding this paragraph exceeds chunk size
            elif len(current_chunk) + len(para) + 2 > self.chunk_size:
                # Save current chunk if not empty
                if current_chunk:
                    chunks.append(self._create_chunk(current_chunk, source, is_markdown))

                # Start new chunk with overlap
                if self.chunk_overlap > 0 and current_chunk:
                    # Take last chunk_overlap chars from current chunk
                    overlap_text = current_chunk[-self.chunk_overlap :]
                    current_chunk = overlap_text + "\n\n" + para
                else:
                    current_chunk = para
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para

        # Don't forget the last chunk
        if current_chunk:
            chunks.append(self._create_chunk(current_chunk, source, is_markdown))

        return chunks

    def _split_long_text(self, text: str) -> list[str]:
        """Split a long text that exceeds chunk_size."""
        pieces = []

        start = 0
        while start < len(text):
            end = start + self.chunk_size
            if end >= len(text):
                pieces.append(text[start:])
                break
            else:
                # Try to split at a space
                split_point = text.rfind(" ", start, end)
                if split_point == -1 or split_point <= start:
                    split_point = end
                pieces.append(text[start:split_point])
                # Move back for overlap
                start = max(start + 1, split_point - self.chunk_overlap)

        return pieces

    def _create_chunk(self, content: str, source: str, is_markdown: bool) -> Chunk:
        """Create a Chunk with appropriate metadata."""
        metadata = {}
        if is_markdown:
            metadata["type"] = "markdown"
        else:
            metadata["type"] = "text"

        return Chunk(
            content=content,
            source=source,
            metadata=metadata,
        )

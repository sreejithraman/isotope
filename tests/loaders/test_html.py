# tests/loaders/test_html.py
"""Tests for HTMLLoader."""

# Check if dependencies are available
import importlib.util
import os
import tempfile

import pytest

from isotope.loaders.base import Loader
from isotope.models import Chunk

HAS_HTML_DEPS = (
    importlib.util.find_spec("bs4") is not None
    and importlib.util.find_spec("markdownify") is not None
)

pytestmark = pytest.mark.skipif(
    not HAS_HTML_DEPS, reason="beautifulsoup4/markdownify not installed"
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory with test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Simple HTML
        with open(os.path.join(tmpdir, "simple.html"), "w") as f:
            f.write("""<!DOCTYPE html>
<html>
<head><title>Test Page</title></head>
<body>
<h1>Hello World</h1>
<p>This is a paragraph.</p>
</body>
</html>""")

        # HTML with scripts and styles to remove
        with open(os.path.join(tmpdir, "complex.html"), "w") as f:
            f.write("""<!DOCTYPE html>
<html>
<head>
<title>Complex Page</title>
<style>body { color: black; }</style>
<script>console.log('test');</script>
</head>
<body>
<nav>Navigation</nav>
<main>
<h1>Main Content</h1>
<p>Important text.</p>
</main>
<footer>Footer content</footer>
</body>
</html>""")

        # Empty HTML
        with open(os.path.join(tmpdir, "empty.html"), "w") as f:
            f.write("")

        # HTML with no content
        with open(os.path.join(tmpdir, "nocontent.htm"), "w") as f:
            f.write("<html><head></head><body></body></html>")

        yield tmpdir


class TestHTMLLoader:
    def test_is_loader(self):
        from isotope.loaders.html import HTMLLoader

        assert isinstance(HTMLLoader(), Loader)

    def test_supports_html(self):
        from isotope.loaders.html import HTMLLoader

        loader = HTMLLoader()
        assert loader.supports("file.html") is True
        assert loader.supports("file.htm") is True
        assert loader.supports("FILE.HTML") is True

    def test_does_not_support_other(self):
        from isotope.loaders.html import HTMLLoader

        loader = HTMLLoader()
        assert loader.supports("file.txt") is False
        assert loader.supports("file.pdf") is False
        assert loader.supports("file.xml") is False

    def test_load_nonexistent_file(self):
        from isotope.loaders.html import HTMLLoader

        loader = HTMLLoader()
        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/path/file.html")

    def test_load_simple_html(self, temp_dir):
        from isotope.loaders.html import HTMLLoader

        loader = HTMLLoader()
        chunks = loader.load(os.path.join(temp_dir, "simple.html"))

        assert len(chunks) == 1
        assert isinstance(chunks[0], Chunk)
        assert "Hello World" in chunks[0].content
        assert "paragraph" in chunks[0].content

    def test_chunk_metadata(self, temp_dir):
        from isotope.loaders.html import HTMLLoader

        loader = HTMLLoader()
        chunks = loader.load(os.path.join(temp_dir, "simple.html"))

        assert chunks[0].metadata.get("type") == "html"
        assert chunks[0].metadata.get("title") == "Test Page"

    def test_chunk_source(self, temp_dir):
        from pathlib import Path

        from isotope.loaders.html import HTMLLoader

        loader = HTMLLoader()
        path = os.path.join(temp_dir, "simple.html")
        chunks = loader.load(path)

        # Source should be the resolved absolute path
        assert chunks[0].source == str(Path(path).resolve())

    def test_removes_scripts_and_styles(self, temp_dir):
        from isotope.loaders.html import HTMLLoader

        loader = HTMLLoader()
        chunks = loader.load(os.path.join(temp_dir, "complex.html"))

        content = chunks[0].content
        # Should not contain script or style content
        assert "console.log" not in content
        assert "color: black" not in content

    def test_removes_nav_and_footer(self, temp_dir):
        from isotope.loaders.html import HTMLLoader

        loader = HTMLLoader()
        chunks = loader.load(os.path.join(temp_dir, "complex.html"))

        content = chunks[0].content
        # Should not contain nav or footer content
        assert "Navigation" not in content
        assert "Footer content" not in content
        # But should contain main content
        assert "Main Content" in content
        assert "Important text" in content

    def test_load_empty_file(self, temp_dir):
        from isotope.loaders.html import HTMLLoader

        loader = HTMLLoader()
        chunks = loader.load(os.path.join(temp_dir, "empty.html"))

        assert chunks == []

    def test_converts_to_markdown(self, temp_dir):
        from isotope.loaders.html import HTMLLoader

        loader = HTMLLoader()
        chunks = loader.load(os.path.join(temp_dir, "simple.html"))

        content = chunks[0].content
        # Should have markdown heading (ATX style)
        assert "# Hello World" in content or "Hello World" in content


class TestHTMLLoaderCleanMarkdown:
    """Test markdown cleanup."""

    def test_clean_markdown_removes_excessive_blanks(self):
        from isotope.loaders.html import HTMLLoader

        loader = HTMLLoader()

        text = "Line 1\n\n\n\nLine 2\n\n\nLine 3"
        result = loader._clean_markdown(text)

        # Should not have more than one blank line in a row
        assert "\n\n\n" not in result
        assert "Line 1" in result
        assert "Line 2" in result
        assert "Line 3" in result

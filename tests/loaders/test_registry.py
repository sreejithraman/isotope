# tests/loaders/test_registry.py
"""Tests for loader registry."""

import os
import tempfile

import pytest

from isotope.loaders.registry import LoaderRegistry
from isotope.loaders.text import TextLoader
from isotope.models import Chunk


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "test.md"), "w") as f:
            f.write("# Test\n\nContent here.")
        with open(os.path.join(tmpdir, "test.txt"), "w") as f:
            f.write("Plain text content.")
        yield tmpdir


class TestLoaderRegistry:
    def test_register_loader(self):
        registry = LoaderRegistry()
        registry.register(TextLoader())
        assert len(registry._loaders) == 1

    def test_find_loader_for_supported_file(self):
        registry = LoaderRegistry()
        registry.register(TextLoader())

        loader = registry.find_loader("file.md")
        assert loader is not None
        assert isinstance(loader, TextLoader)

    def test_find_loader_returns_none_for_unsupported(self):
        registry = LoaderRegistry()
        registry.register(TextLoader())

        loader = registry.find_loader("file.pdf")
        assert loader is None

    def test_load_auto_selects_loader(self, temp_dir):
        registry = LoaderRegistry()
        registry.register(TextLoader())

        chunks = registry.load(os.path.join(temp_dir, "test.md"))
        assert len(chunks) >= 1
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_load_raises_for_unsupported(self, temp_dir):
        registry = LoaderRegistry()
        registry.register(TextLoader())

        with pytest.raises(ValueError, match="No loader"):
            registry.load(os.path.join(temp_dir, "test.pdf"))

    def test_default_registry(self, temp_dir):
        registry = LoaderRegistry.default()

        # Should have TextLoader registered
        chunks = registry.load(os.path.join(temp_dir, "test.txt"))
        assert len(chunks) >= 1

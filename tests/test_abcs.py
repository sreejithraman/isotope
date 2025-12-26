# tests/test_abcs.py
"""Tests for remaining abstract base classes."""

import pytest
from abc import ABC

from isotopedb.atomizer.base import Atomizer
from isotopedb.dedup.base import Deduplicator
from isotopedb.loaders.base import Loader


class TestAtomizerABC:
    def test_is_abstract(self):
        assert issubclass(Atomizer, ABC)

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            Atomizer()

    def test_has_atomize_method(self):
        assert hasattr(Atomizer, "atomize")


class TestDeduplicatorABC:
    def test_is_abstract(self):
        assert issubclass(Deduplicator, ABC)

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            Deduplicator()

    def test_has_get_chunks_to_remove_method(self):
        assert hasattr(Deduplicator, "get_chunks_to_remove")


class TestLoaderABC:
    def test_is_abstract(self):
        assert issubclass(Loader, ABC)

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            Loader()

    def test_has_required_methods(self):
        assert hasattr(Loader, "load")
        assert hasattr(Loader, "supports")

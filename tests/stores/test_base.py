# tests/stores/test_base.py
"""Tests for storage abstract base classes."""

import pytest
from abc import ABC

from isotopedb.stores.base import DocStore, VectorStore
from isotopedb.models import Chunk, Question


class TestVectorStoreABC:
    def test_is_abstract(self):
        assert issubclass(VectorStore, ABC)

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            VectorStore()

    def test_has_required_methods(self):
        assert hasattr(VectorStore, "add")
        assert hasattr(VectorStore, "search")
        assert hasattr(VectorStore, "delete_by_chunk_ids")
        assert hasattr(VectorStore, "list_chunk_ids")


class TestDocStoreABC:
    def test_is_abstract(self):
        assert issubclass(DocStore, ABC)

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            DocStore()

    def test_has_required_methods(self):
        assert hasattr(DocStore, "put")
        assert hasattr(DocStore, "get")
        assert hasattr(DocStore, "get_many")
        assert hasattr(DocStore, "delete")
        assert hasattr(DocStore, "list_sources")
        assert hasattr(DocStore, "get_by_source")

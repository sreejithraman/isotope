"""Tests for the Atom model."""

import pytest
from pydantic import ValidationError

from isotope.models import Atom


class TestAtom:
    def test_create_atom_with_defaults(self):
        atom = Atom(content="Test content", chunk_id="chunk-123")
        assert atom.content == "Test content"
        assert atom.chunk_id == "chunk-123"
        assert atom.index == 0
        assert atom.id is not None

    def test_create_atom_with_custom_id(self):
        atom = Atom(id="custom-id", content="Test", chunk_id="c1")
        assert atom.id == "custom-id"

    def test_create_atom_with_index(self):
        atom = Atom(content="Test", chunk_id="c1", index=5)
        assert atom.index == 5

    def test_atom_id_is_unique(self):
        a1 = Atom(content="A", chunk_id="c1")
        a2 = Atom(content="B", chunk_id="c1")
        assert a1.id != a2.id

    def test_atom_requires_content(self):
        with pytest.raises(ValidationError):
            Atom(chunk_id="c1")

    def test_atom_requires_chunk_id(self):
        with pytest.raises(ValidationError):
            Atom(content="Test")

    def test_atom_serialization(self):
        atom = Atom(content="Test content", chunk_id="c1", index=3)
        data = atom.model_dump()
        assert data["content"] == "Test content"
        assert data["chunk_id"] == "c1"
        assert data["index"] == 3
        assert "id" in data

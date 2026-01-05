# tests/models/test_chunk.py
"""Tests for Chunk and Atom models."""

from isotope.models.atom import Atom
from isotope.models.chunk import Chunk


class TestChunk:
    def test_create_chunk_with_defaults(self):
        chunk = Chunk(content="Hello world", source="test.md")
        assert chunk.content == "Hello world"
        assert chunk.source == "test.md"
        assert chunk.id  # auto-generated
        assert chunk.metadata == {}

    def test_create_chunk_with_metadata(self):
        chunk = Chunk(
            content="Hello",
            source="test.pdf",
            metadata={"page": 1, "type": "pdf"},
        )
        assert chunk.metadata["page"] == 1
        assert chunk.metadata["type"] == "pdf"

    def test_chunk_id_is_unique(self):
        c1 = Chunk(content="a", source="x")
        c2 = Chunk(content="a", source="x")
        assert c1.id != c2.id


class TestAtom:
    def test_create_atom(self):
        atom = Atom(content="Python is a programming language.", chunk_id="chunk-123")
        assert atom.content == "Python is a programming language."
        assert atom.chunk_id == "chunk-123"
        assert atom.id  # auto-generated

    def test_atom_id_is_unique(self):
        a1 = Atom(content="fact", chunk_id="c1")
        a2 = Atom(content="fact", chunk_id="c1")
        assert a1.id != a2.id

    def test_create_atom_with_index(self):
        atom = Atom(content="First fact.", chunk_id="c1", index=0)
        assert atom.index == 0
        atom2 = Atom(content="Second fact.", chunk_id="c1", index=1)
        assert atom2.index == 1

    def test_atom_default_index_is_zero(self):
        atom = Atom(content="A fact.", chunk_id="c1")
        assert atom.index == 0

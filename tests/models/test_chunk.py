# tests/models/test_chunk.py
"""Tests for Chunk and Atom models."""

import pytest
from isotopedb.models.chunk import Chunk, Atom


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

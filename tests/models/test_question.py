# tests/models/test_question.py
"""Tests for Question model."""

import pytest
from isotopedb.models.chunk import Question


class TestQuestion:
    def test_create_question_minimal(self):
        q = Question(text="What is Python?", chunk_id="chunk-123")
        assert q.text == "What is Python?"
        assert q.chunk_id == "chunk-123"
        assert q.id  # auto-generated
        assert q.atom_id is None
        assert q.embedding is None

    def test_create_question_with_atom_id(self):
        q = Question(
            text="What is Python?",
            chunk_id="chunk-123",
            atom_id="atom-456",
        )
        assert q.atom_id == "atom-456"

    def test_create_question_with_embedding(self):
        embedding = [0.1, 0.2, 0.3]
        q = Question(
            text="What is Python?",
            chunk_id="chunk-123",
            embedding=embedding,
        )
        assert q.embedding == embedding

    def test_question_id_is_unique(self):
        q1 = Question(text="Q?", chunk_id="c1")
        q2 = Question(text="Q?", chunk_id="c1")
        assert q1.id != q2.id

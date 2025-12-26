# tests/models/test_question.py
"""Tests for Question and EmbeddedQuestion models."""

import pytest
from isotopedb.models.chunk import EmbeddedQuestion, Question


class TestQuestion:
    def test_create_question_minimal(self):
        q = Question(text="What is Python?", chunk_id="chunk-123")
        assert q.text == "What is Python?"
        assert q.chunk_id == "chunk-123"
        assert q.id  # auto-generated
        assert q.atom_id is None

    def test_create_question_with_atom_id(self):
        q = Question(
            text="What is Python?",
            chunk_id="chunk-123",
            atom_id="atom-456",
        )
        assert q.atom_id == "atom-456"

    def test_question_id_is_unique(self):
        q1 = Question(text="Q?", chunk_id="c1")
        q2 = Question(text="Q?", chunk_id="c1")
        assert q1.id != q2.id


class TestEmbeddedQuestion:
    def test_create_embedded_question(self):
        q = Question(text="What is Python?", chunk_id="chunk-123")
        embedding = [0.1, 0.2, 0.3]
        eq = EmbeddedQuestion(question=q, embedding=embedding)

        assert eq.question == q
        assert eq.embedding == embedding
        assert eq.question.text == "What is Python?"
        assert eq.question.chunk_id == "chunk-123"

    def test_embedded_question_requires_embedding(self):
        q = Question(text="What is Python?", chunk_id="chunk-123")
        with pytest.raises(Exception):  # Pydantic validation error
            EmbeddedQuestion(question=q)

    def test_embedded_question_preserves_question_id(self):
        q = Question(text="Test?", chunk_id="c1")
        eq = EmbeddedQuestion(question=q, embedding=[1.0, 0.0])
        assert eq.question.id == q.id

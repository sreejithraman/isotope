# tests/models/test_question.py
"""Tests for Question and EmbeddedQuestion models."""

import pytest
from pydantic import ValidationError

from isotope.models.question import EmbeddedQuestion, Question


class TestQuestion:
    def test_create_question(self):
        q = Question(
            text="What is Python?",
            chunk_id="chunk-123",
            atom_id="atom-456",
        )
        assert q.text == "What is Python?"
        assert q.chunk_id == "chunk-123"
        assert q.atom_id == "atom-456"
        assert q.id  # auto-generated

    def test_question_requires_atom_id(self):
        with pytest.raises(ValidationError):
            Question(text="What is Python?", chunk_id="chunk-123")

    def test_question_id_is_unique(self):
        q1 = Question(text="Q?", chunk_id="c1", atom_id="a1")
        q2 = Question(text="Q?", chunk_id="c1", atom_id="a1")
        assert q1.id != q2.id


class TestEmbeddedQuestion:
    def test_create_embedded_question(self):
        q = Question(text="What is Python?", chunk_id="chunk-123", atom_id="atom-456")
        embedding = [0.1, 0.2, 0.3]
        eq = EmbeddedQuestion(question=q, embedding=embedding)

        assert eq.question == q
        assert eq.embedding == embedding
        assert eq.question.text == "What is Python?"
        assert eq.question.chunk_id == "chunk-123"
        assert eq.question.atom_id == "atom-456"

    def test_embedded_question_requires_embedding(self):
        q = Question(text="What is Python?", chunk_id="chunk-123", atom_id="atom-456")
        with pytest.raises(ValidationError):
            EmbeddedQuestion(question=q)

    def test_embedded_question_preserves_question_id(self):
        q = Question(text="Test?", chunk_id="c1", atom_id="a1")
        eq = EmbeddedQuestion(question=q, embedding=[1.0, 0.0])
        assert eq.question.id == q.id

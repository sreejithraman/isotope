# tests/models/test_results.py
"""Tests for SearchResult and QueryResponse models."""

import pytest
from pydantic import ValidationError

from isotope.models.atom import Atom
from isotope.models.chunk import Chunk
from isotope.models.question import Question
from isotope.models.results import QueryResponse, SearchResult


class TestSearchResult:
    def test_create_search_result(self):
        chunk = Chunk(content="Python is great", source="test.md")
        atom = Atom(content="Python is great", chunk_id=chunk.id)
        question = Question(text="What is Python?", chunk_id=chunk.id, atom_id=atom.id)
        result = SearchResult(question=question, chunk=chunk, score=0.95, atom=atom)

        assert result.question == question
        assert result.chunk == chunk
        assert result.score == 0.95
        assert result.atom == atom

    def test_search_result_requires_atom(self):
        chunk = Chunk(content="Python is great", source="test.md")
        atom = Atom(content="Python is great", chunk_id=chunk.id)
        question = Question(text="What is Python?", chunk_id=chunk.id, atom_id=atom.id)

        with pytest.raises(ValidationError):
            SearchResult(question=question, chunk=chunk, score=0.95)  # Missing atom


class TestQueryResponse:
    def test_create_query_response_with_answer(self):
        chunk = Chunk(content="Python is great", source="test.md")
        atom = Atom(content="Python is great", chunk_id=chunk.id)
        question = Question(text="What is Python?", chunk_id=chunk.id, atom_id=atom.id)
        result = SearchResult(question=question, chunk=chunk, score=0.95, atom=atom)

        response = QueryResponse(
            query="Tell me about Python",
            answer="Python is a programming language.",
            results=[result],
        )
        assert response.query == "Tell me about Python"
        assert response.answer == "Python is a programming language."
        assert len(response.results) == 1

    def test_create_query_response_without_answer(self):
        response = QueryResponse(
            query="Tell me about Python",
            answer=None,
            results=[],
        )
        assert response.answer is None

# tests/models/test_results.py
"""Tests for SearchResult and QueryResponse models."""

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

    def test_search_result_defaults_to_none(self):
        """question and atom default to None."""
        chunk = Chunk(content="Python is great", source="test.md")
        result = SearchResult(chunk=chunk, score=0.95)
        assert result.question is None
        assert result.atom is None


class TestSearchResultOptionalFields:
    def test_create_search_result_without_question_and_atom(self):
        """Chunk-fallback results have no question or atom."""
        chunk = Chunk(content="Python is great", source="test.md")
        result = SearchResult(chunk=chunk, score=0.85, question=None, atom=None)
        assert result.question is None
        assert result.atom is None
        assert result.chunk == chunk
        assert result.score == 0.85

    def test_create_search_result_with_all_fields(self):
        """Question-match results still work with all fields."""
        chunk = Chunk(content="Python is great", source="test.md")
        atom = Atom(content="Python is great", chunk_id=chunk.id)
        question = Question(text="What is Python?", chunk_id=chunk.id, atom_id=atom.id)
        result = SearchResult(question=question, chunk=chunk, score=0.95, atom=atom)
        assert result.question == question
        assert result.atom == atom


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

# tests/models/test_results.py
"""Tests for SearchResult and QueryResponse models."""

import pytest
from isotopedb.models.chunk import Chunk, Question
from isotopedb.models.results import QueryResponse, SearchResult


class TestSearchResult:
    def test_create_search_result(self):
        chunk = Chunk(content="Python is great", source="test.md")
        question = Question(text="What is Python?", chunk_id=chunk.id)
        result = SearchResult(question=question, chunk=chunk, score=0.95)

        assert result.question == question
        assert result.chunk == chunk
        assert result.score == 0.95


class TestQueryResponse:
    def test_create_query_response_with_answer(self):
        chunk = Chunk(content="Python is great", source="test.md")
        question = Question(text="What is Python?", chunk_id=chunk.id)
        result = SearchResult(question=question, chunk=chunk, score=0.95)

        response = QueryResponse(
            query="Tell me about Python",
            answer="Python is a programming language.",
            results=[result],
        )
        assert response.query == "Tell me about Python"
        assert response.answer == "Python is a programming language."
        assert len(response.results) == 1

    def test_create_query_response_raw_mode(self):
        response = QueryResponse(
            query="Tell me about Python",
            answer=None,
            results=[],
        )
        assert response.answer is None

# src/isotopedb/models/results.py
"""Result data models for Isotope queries."""

from pydantic import BaseModel

from isotopedb.models.chunk import Chunk, Question


class SearchResult(BaseModel):
    """A single matched question + its chunk."""

    question: Question
    chunk: Chunk
    score: float


class QueryResponse(BaseModel):
    """Full response to a user query."""

    query: str
    answer: str | None
    results: list[SearchResult]

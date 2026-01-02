# src/isotopedb/models/results.py
"""Result data models for Isotope queries."""

from pydantic import BaseModel

from isotopedb.models.atom import Atom
from isotopedb.models.chunk import Chunk
from isotopedb.models.question import Question


class SearchResult(BaseModel):
    """A single matched question + its chunk + atom."""

    question: Question
    chunk: Chunk
    score: float
    atom: Atom  # Required - every result includes its atom


class QueryResponse(BaseModel):
    """Full response to a user query."""

    query: str
    answer: str | None
    results: list[SearchResult]

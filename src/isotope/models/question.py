# src/isotope/models/question.py
"""Question data models."""

from uuid import uuid4

from pydantic import BaseModel, Field


class Question(BaseModel):
    """An atomic question that an atom/chunk answers."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    text: str
    chunk_id: str
    atom_id: str  # Required - every question is generated from an atom


class EmbeddedQuestion(BaseModel):
    """A Question paired with its embedding vector. Used during ingestion."""

    question: Question
    embedding: list[float]

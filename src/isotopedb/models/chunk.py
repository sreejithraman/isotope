# src/isotopedb/models/chunk.py
"""Chunk data model."""

from uuid import uuid4

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A piece of content that can answer questions."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str
    source: str
    metadata: dict = Field(default_factory=dict)

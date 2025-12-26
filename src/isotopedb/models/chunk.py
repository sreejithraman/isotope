# src/isotopedb/models/chunk.py
"""Chunk and Atom data models."""

from uuid import uuid4

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A piece of content that can answer questions."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str
    source: str
    metadata: dict = Field(default_factory=dict)


class Atom(BaseModel):
    """An atomic statement extracted from a chunk."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str
    chunk_id: str

# src/isotope/models/atom.py
"""Atom data model."""

from uuid import uuid4

from pydantic import BaseModel, Field


class Atom(BaseModel):
    """An atomic statement extracted from a chunk."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str
    chunk_id: str
    index: int = 0  # Position within chunk (j in paper notation)

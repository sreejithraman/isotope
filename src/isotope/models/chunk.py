# src/isotope/models/chunk.py
"""Chunk data model."""

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A piece of content that can answer questions."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str
    source: str
    metadata: dict[str, Any] = Field(default_factory=dict)

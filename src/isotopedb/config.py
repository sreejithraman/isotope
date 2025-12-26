# src/isotopedb/config.py
"""Configuration management for Isotope."""

from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Isotope configuration settings."""

    model_config = SettingsConfigDict(
        env_prefix="ISOTOPE_",
        env_file=".env",
        extra="ignore",
    )

    # LLM (question generation)
    llm_model: str = "gpt-4o-mini"

    # Embeddings
    embedding_model: str = "text-embedding-3-small"

    # Atomization
    atomizer: Literal["sentence", "llm"] = "sentence"

    # Question generation
    questions_per_atom: int = 5
    question_prompt: str | None = None

    # Question diversity deduplication
    question_diversity_threshold: float | None = 0.85

    # Storage
    data_dir: str = "./isotope_data"
    vector_store: Literal["chroma"] = "chroma"
    doc_store: Literal["sqlite"] = "sqlite"

    # Re-ingestion deduplication
    dedup_strategy: Literal["none", "source_aware"] = "source_aware"

    # Retrieval
    default_k: int = 5

    @field_validator("question_diversity_threshold", mode="before")
    @classmethod
    def parse_threshold(cls, v):
        if v == "" or v is None:
            return None
        return float(v)

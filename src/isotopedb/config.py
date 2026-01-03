# src/isotopedb/config.py
"""Configuration management for IsotopeDB.

This module contains behavioral settings that apply regardless of which
LLM provider is used. Provider-specific configuration (model names, API keys)
is handled by the provider modules (e.g., isotopedb.litellm).
"""

from typing import Literal, cast

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Behavioral settings for IsotopeDB.

    These settings control how IsotopeDB behaves, independent of which
    LLM/embedding provider is used. Set via environment variables with
    the ISOTOPE_ prefix.

    Example:
        export ISOTOPE_QUESTIONS_PER_ATOM=20
        export ISOTOPE_DIVERSITY_SCOPE=per_chunk
    """

    model_config = SettingsConfigDict(
        env_prefix="ISOTOPE_",
        extra="ignore",
    )

    # Question generation
    questions_per_atom: int = 15
    question_prompt: str | None = None

    # Atomization
    atomizer_prompt: str | None = None

    # Question diversity deduplication
    question_diversity_threshold: float | None = 0.85
    diversity_scope: Literal["global", "per_chunk", "per_atom"] = "global"

    # Re-ingestion deduplication
    dedup_strategy: Literal["none", "source_aware"] = "source_aware"

    # Retrieval
    default_k: int = 5
    synthesis_prompt: str | None = None

    @field_validator("question_diversity_threshold", mode="before")
    @classmethod
    def parse_threshold(cls, v: object) -> float | None:
        """Parse threshold, treating empty string as None (disabled)."""
        if v == "" or v is None:
            return None
        return float(v)  # type: ignore[arg-type]

    @field_validator("diversity_scope", mode="before")
    @classmethod
    def parse_diversity_scope(cls, v: object) -> Literal["global", "per_chunk", "per_atom"]:
        """Parse diversity scope, normalizing to lowercase."""
        if v is None or v == "":
            return "global"
        value = str(v).lower()
        if value not in {"global", "per_chunk", "per_atom"}:
            raise ValueError("diversity_scope must be global, per_chunk, or per_atom")
        return cast(Literal["global", "per_chunk", "per_atom"], value)

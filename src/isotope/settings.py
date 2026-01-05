# src/isotope/settings.py
"""Configuration management for Isotope.

This module contains behavioral settings that apply regardless of which
LLM provider is used. Settings are passed programmatically - the library
does not read from environment variables (following SDK best practices
like OpenAI, Stripe, boto3).

For applications that want env-based config, read env vars at the
application layer and pass values explicitly to Isotope factory methods.
"""

from typing import Literal

from pydantic import BaseModel


class Settings(BaseModel):
    """Behavioral settings for Isotope.

    These settings control how Isotope behaves, independent of which
    LLM/embedding provider is used.

    Example:
        settings = Settings(
            questions_per_atom=20,
            diversity_scope="per_chunk",
        )
    """

    # Question generation
    questions_per_atom: int = 15
    question_generator_prompt: str | None = None

    # Atomization
    atomizer_prompt: str | None = None

    # Question diversity deduplication
    question_diversity_threshold: float | None = 0.85
    diversity_scope: Literal["global", "per_chunk", "per_atom"] = "global"

    # Retrieval
    default_k: int = 5
    synthesis_prompt: str | None = None
    synthesis_temperature: float | None = 0.3

    # Async ingestion
    max_concurrent_questions: int = 10

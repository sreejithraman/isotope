# src/isotope/settings.py
"""Configuration management for Isotope.

This module contains behavioral settings that apply regardless of which
LLM provider is used. Settings are passed programmatically - the library
does not read from environment variables (following SDK best practices
like OpenAI, Stripe, boto3).

For applications that want env-based config, read env vars at the
application layer and pass values explicitly to Isotope factory methods.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

# Rate limit profile definitions
RATE_LIMIT_PROFILES: dict[str, dict[str, int]] = {
    "aggressive": {
        "max_concurrent_questions": 10,
        "num_retries": 3,
    },
    "conservative": {
        "max_concurrent_questions": 2,
        "num_retries": 5,
    },
}


class Settings(BaseModel):
    """Behavioral settings for Isotope.

    These settings control how Isotope behaves, independent of which
    LLM/embedding provider is used.

    Example:
        settings = Settings(
            questions_per_atom=20,
            diversity_scope="per_chunk",
        )

        # Or use a rate limit profile for free API tiers
        settings = Settings.with_profile("conservative")
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

    # Retry configuration (LiteLLM handles exponential backoff for RateLimitError)
    num_retries: int = 3

    @classmethod
    def with_profile(
        cls,
        profile: Literal["aggressive", "conservative"],
        **overrides,
    ) -> Settings:
        """Create Settings with a rate limit profile.

        Profiles bundle settings optimized for different API tier limits:
        - "aggressive": For paid API tiers with high rate limits (default behavior)
        - "conservative": For free tiers or APIs with strict rate limits

        Args:
            profile: The rate limit profile to use.
            **overrides: Additional settings to override profile defaults.

        Returns:
            Settings instance with profile values applied.

        Example:
            # For free API tiers (e.g., Gemini free tier)
            settings = Settings.with_profile("conservative")

            # With additional overrides
            settings = Settings.with_profile("conservative", questions_per_atom=10)
        """
        if profile not in RATE_LIMIT_PROFILES:
            raise ValueError(
                f"Unknown profile '{profile}'. "
                f"Available profiles: {list(RATE_LIMIT_PROFILES.keys())}"
            )

        profile_settings = RATE_LIMIT_PROFILES[profile].copy()
        profile_settings.update(overrides)
        return cls(**profile_settings)

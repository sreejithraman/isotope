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

from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

if TYPE_CHECKING:
    from isotope.question_generator.base import BatchConfig

# Rate limit profile definitions
RATE_LIMIT_PROFILES: dict[str, dict[str, int]] = {
    "aggressive": {
        "max_concurrent_llm_calls": 10,
        "num_retries": 5,
    },
    "conservative": {
        "max_concurrent_llm_calls": 2,
        "num_retries": 5,
    },
}

# Generation preset definitions for different deployment scenarios
# - "cloud": Many concurrent single-atom calls (optimized for cloud APIs)
# - "local": Multi-atom batching with low concurrency (optimized for local models)
GENERATION_PRESETS: dict[str, dict[str, int]] = {
    "cloud": {
        "batch_size": 1,
        "max_concurrent_llm_calls": 50,
    },
    "local": {
        "batch_size": 5,
        "max_concurrent_llm_calls": 2,
    },
}

# Model prefixes that indicate local models (used for auto-detection)
LOCAL_MODEL_PREFIXES = ("ollama/", "llama.cpp/", "local/")


def detect_generation_preset(model: str) -> Literal["cloud", "local"]:
    """Auto-detect the appropriate generation preset based on model name.

    Args:
        model: The model identifier (e.g., "ollama/llama3.2", "openai/gpt-4o")

    Returns:
        "local" for local models, "cloud" for cloud APIs
    """
    if model.lower().startswith(LOCAL_MODEL_PREFIXES):
        return "local"
    return "cloud"


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

    # Atomization
    use_sentence_atomizer: bool = False  # True = fast sentence-based, False = LLM quality

    # Question generation
    questions_per_atom: int = 5
    question_generator_prompt: str | None = None

    # Generation batching (for multi-atom prompts)
    # Use generation_preset for convenience, or set batch_size directly
    generation_preset: Literal["cloud", "local"] | None = None
    batch_size: int | None = None  # Atoms per LLM prompt (None = use preset default)

    # Custom atomization prompt (only used when use_sentence_atomizer=False)
    atomizer_prompt: str | None = None

    # Question diversity deduplication
    question_diversity_threshold: float | None = 0.85
    diversity_scope: Literal["global", "per_chunk", "per_atom"] = "global"

    # Retrieval
    default_k: int = 5
    synthesis_prompt: str | None = None
    synthesis_temperature: float | None = 0.3

    # Concurrency for async ingestion (question generation + LLM atomization)
    max_concurrent_llm_calls: int = 10

    # Retry configuration (LiteLLM handles exponential backoff for RateLimitError)
    num_retries: int = 5

    @classmethod
    def with_profile(
        cls,
        profile: Literal["aggressive", "conservative"],
        **overrides: Any,
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

        profile_settings: dict[str, Any] = RATE_LIMIT_PROFILES[profile].copy()
        profile_settings.update(overrides)
        return cls(**profile_settings)

    def get_batch_config(self, model: str | None = None) -> tuple[int, int]:
        """Resolve batch configuration based on settings and model.

        Precedence:
        1. Explicit batch_size setting (if set)
        2. generation_preset (if set)
        3. Auto-detect from model name (if provided)
        4. Default to cloud preset

        Args:
            model: Optional model name for auto-detection (e.g., "ollama/llama3.2")

        Returns:
            Tuple of (batch_size, max_concurrent)
        """

        # Determine preset
        preset: Literal["cloud", "local"]
        if self.generation_preset is not None:
            preset = self.generation_preset
        elif model is not None:
            preset = detect_generation_preset(model)
        else:
            preset = "cloud"

        # Get preset defaults
        preset_config = GENERATION_PRESETS[preset]
        batch_size = preset_config["batch_size"]
        max_concurrent = preset_config["max_concurrent_llm_calls"]

        # Override with explicit settings (use model_fields_set to detect explicit values)
        if self.batch_size is not None:
            batch_size = self.batch_size
        if "max_concurrent_llm_calls" in self.model_fields_set:
            max_concurrent = self.max_concurrent_llm_calls

        return batch_size, max_concurrent

    def build_batch_config(self, model: str | None = None) -> BatchConfig:
        """Build a BatchConfig from settings.

        Args:
            model: Optional model name for auto-detection

        Returns:
            BatchConfig instance
        """
        from isotope.question_generator.base import BatchConfig

        batch_size, max_concurrent = self.get_batch_config(model)
        return BatchConfig(batch_size=batch_size, max_concurrent=max_concurrent)

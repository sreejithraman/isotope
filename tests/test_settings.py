# tests/test_config.py
"""Tests for configuration.

Settings is now a plain BaseModel (no env var reading).
The library is programmatic-first - env vars are read by the CLI application layer.
"""

from isotope.settings import Settings


class TestSettings:
    def test_default_settings(self):
        """Test Settings has correct defaults."""
        settings = Settings()
        assert settings.questions_per_atom == 15
        assert settings.question_diversity_threshold == 0.85
        assert settings.diversity_scope == "global"
        assert settings.default_k == 5
        assert settings.question_generator_prompt is None
        assert settings.atomizer_prompt is None
        assert settings.synthesis_prompt is None

    def test_settings_with_custom_values(self):
        """Test Settings accepts custom values."""
        settings = Settings(
            questions_per_atom=20,
            question_diversity_threshold=0.9,
            diversity_scope="per_chunk",
            default_k=10,
            question_generator_prompt="custom prompt",
            atomizer_prompt="atomizer prompt",
            synthesis_prompt="synthesis prompt",
        )
        assert settings.questions_per_atom == 20
        assert settings.question_diversity_threshold == 0.9
        assert settings.diversity_scope == "per_chunk"
        assert settings.default_k == 10
        assert settings.question_generator_prompt == "custom prompt"
        assert settings.atomizer_prompt == "atomizer prompt"
        assert settings.synthesis_prompt == "synthesis prompt"

    def test_settings_with_disabled_threshold(self):
        """Test Settings with None threshold (disabled)."""
        settings = Settings(question_diversity_threshold=None)
        assert settings.question_diversity_threshold is None

    def test_settings_all_diversity_scopes(self):
        """Test all valid diversity_scope values."""
        for scope in ["global", "per_chunk", "per_atom"]:
            settings = Settings(diversity_scope=scope)  # type: ignore[arg-type]
            assert settings.diversity_scope == scope

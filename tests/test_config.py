# tests/test_config.py
"""Tests for configuration."""

from isotopedb.config import Settings


class TestSettings:
    def test_default_settings(self):
        """Test optional settings have correct defaults."""
        settings = Settings()
        assert settings.questions_per_atom == 15
        assert settings.question_diversity_threshold == 0.85
        assert settings.diversity_scope == "global"
        assert settings.dedup_strategy == "source_aware"
        assert settings.default_k == 5
        assert settings.question_prompt is None

    def test_settings_from_env(self, monkeypatch):
        """Test settings loaded from environment variables."""
        monkeypatch.setenv("ISOTOPE_QUESTIONS_PER_ATOM", "10")
        monkeypatch.setenv("ISOTOPE_DIVERSITY_SCOPE", "per_chunk")
        monkeypatch.setenv("ISOTOPE_DEDUP_STRATEGY", "none")
        monkeypatch.setenv("ISOTOPE_DEFAULT_K", "10")

        settings = Settings()
        assert settings.questions_per_atom == 10
        assert settings.diversity_scope == "per_chunk"
        assert settings.dedup_strategy == "none"
        assert settings.default_k == 10

    def test_question_diversity_threshold_none(self, monkeypatch):
        """Test empty string threshold is treated as None (disabled)."""
        monkeypatch.setenv("ISOTOPE_QUESTION_DIVERSITY_THRESHOLD", "")
        settings = Settings()
        assert settings.question_diversity_threshold is None

    def test_question_diversity_threshold_custom(self, monkeypatch):
        """Test custom threshold value."""
        monkeypatch.setenv("ISOTOPE_QUESTION_DIVERSITY_THRESHOLD", "0.9")
        settings = Settings()
        assert settings.question_diversity_threshold == 0.9

    def test_custom_question_prompt(self, monkeypatch):
        """Test custom question prompt from env var."""
        custom_prompt = "Generate questions about: {atom}"
        monkeypatch.setenv("ISOTOPE_QUESTION_PROMPT", custom_prompt)
        settings = Settings()
        assert settings.question_prompt == custom_prompt

    def test_diversity_scope_values(self, monkeypatch):
        """Test all valid diversity_scope values."""
        for scope in ["global", "per_chunk", "per_atom"]:
            monkeypatch.setenv("ISOTOPE_DIVERSITY_SCOPE", scope)
            settings = Settings()
            assert settings.diversity_scope == scope

    def test_diversity_scope_case_insensitive(self, monkeypatch):
        """Test diversity_scope is case insensitive."""
        monkeypatch.setenv("ISOTOPE_DIVERSITY_SCOPE", "PER_CHUNK")
        settings = Settings()
        assert settings.diversity_scope == "per_chunk"

    def test_diversity_scope_default_on_empty(self, monkeypatch):
        """Test empty diversity_scope defaults to global."""
        monkeypatch.setenv("ISOTOPE_DIVERSITY_SCOPE", "")
        settings = Settings()
        assert settings.diversity_scope == "global"

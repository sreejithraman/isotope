# tests/test_config.py
"""Tests for configuration."""

import os
import pytest
from isotopedb.config import Settings


class TestSettings:
    def test_default_settings(self):
        settings = Settings()
        assert settings.llm_model == "gpt-4o-mini"
        assert settings.embedding_model == "text-embedding-3-small"
        assert settings.atomizer == "sentence"
        assert settings.questions_per_atom == 5
        assert settings.question_diversity_threshold == 0.85
        assert settings.data_dir == "./isotope_data"
        assert settings.vector_store == "chroma"
        assert settings.doc_store == "sqlite"
        assert settings.dedup_strategy == "source_aware"
        assert settings.default_k == 5

    def test_settings_from_env(self, monkeypatch):
        monkeypatch.setenv("ISOTOPE_LLM_MODEL", "gemini/gemini-1.5-flash")
        monkeypatch.setenv("ISOTOPE_QUESTIONS_PER_ATOM", "10")
        monkeypatch.setenv("ISOTOPE_ATOMIZER", "llm")

        settings = Settings()
        assert settings.llm_model == "gemini/gemini-1.5-flash"
        assert settings.questions_per_atom == 10
        assert settings.atomizer == "llm"

    def test_question_diversity_threshold_none(self, monkeypatch):
        monkeypatch.setenv("ISOTOPE_QUESTION_DIVERSITY_THRESHOLD", "")
        settings = Settings()
        # Empty string should be treated as None/disabled
        # We'll handle this with a validator

    def test_custom_question_prompt(self, monkeypatch):
        custom_prompt = "Generate questions about: {atom}"
        monkeypatch.setenv("ISOTOPE_QUESTION_PROMPT", custom_prompt)
        settings = Settings()
        assert settings.question_prompt == custom_prompt

# tests/commands/test_init.py
"""Tests for the init command."""

from __future__ import annotations

from pathlib import Path

import pytest

from isotope.commands import init as init_cmd
from isotope.commands.base import InitPrompt, PromptRequest


class TestGetSettingsForInit:
    """Tests for get_settings_for_init()."""

    def test_local_model_settings(self) -> None:
        """Local models should use conservative settings."""
        settings = init_cmd.get_settings_for_init(
            is_local=True,
            rate_limited=None,
            priority="balanced",
        )

        assert settings["use_sentence_atomizer"] is True
        assert settings["questions_per_atom"] == 5
        assert settings["max_concurrent_llm_calls"] == 1

    def test_rate_limited_speed_priority(self) -> None:
        """Rate-limited API with speed priority."""
        settings = init_cmd.get_settings_for_init(
            is_local=False,
            rate_limited=True,
            priority="speed",
        )

        assert settings["use_sentence_atomizer"] is True
        assert settings["questions_per_atom"] == 3
        assert settings["max_concurrent_llm_calls"] == 2

    def test_rate_limited_quality_priority(self) -> None:
        """Rate-limited API with quality priority."""
        settings = init_cmd.get_settings_for_init(
            is_local=False,
            rate_limited=True,
            priority="quality",
        )

        assert settings["use_sentence_atomizer"] is False
        assert settings["questions_per_atom"] == 5
        assert settings["max_concurrent_llm_calls"] == 2

    def test_high_limit_speed_priority(self) -> None:
        """High-limit API with speed priority."""
        settings = init_cmd.get_settings_for_init(
            is_local=False,
            rate_limited=False,
            priority="speed",
        )

        assert settings["use_sentence_atomizer"] is True
        assert settings["max_concurrent_llm_calls"] == 10

    def test_high_limit_quality_priority(self) -> None:
        """High-limit API with quality priority."""
        settings = init_cmd.get_settings_for_init(
            is_local=False,
            rate_limited=False,
            priority="quality",
        )

        assert settings["use_sentence_atomizer"] is False
        assert settings["questions_per_atom"] == 10
        assert settings["max_concurrent_llm_calls"] == 10

    def test_high_limit_balanced_returns_empty(self) -> None:
        """High-limit API with balanced priority uses defaults."""
        settings = init_cmd.get_settings_for_init(
            is_local=False,
            rate_limited=False,
            priority="balanced",
        )

        assert settings == {}


class TestGenerateConfigContent:
    """Tests for generate_config_content()."""

    def test_generates_valid_yaml_structure(self) -> None:
        """Generated content has required YAML fields."""
        content = init_cmd.generate_config_content(
            llm_model="openai/gpt-5-mini",
            embedding_model="openai/text-embedding-3-small",
            settings={},
        )

        assert "provider: litellm" in content
        assert "llm_model: openai/gpt-5-mini" in content
        assert "embedding_model: openai/text-embedding-3-small" in content

    def test_includes_settings_when_provided(self) -> None:
        """Settings are included in generated config."""
        content = init_cmd.generate_config_content(
            llm_model="openai/gpt-5-mini",
            embedding_model="openai/text-embedding-3-small",
            settings={"use_sentence_atomizer": True, "questions_per_atom": 5},
        )

        assert "settings:" in content
        assert "use_sentence_atomizer: true" in content
        assert "questions_per_atom: 5" in content

    def test_boolean_values_are_lowercase(self) -> None:
        """Boolean values are lowercase YAML format."""
        content = init_cmd.generate_config_content(
            llm_model="model",
            embedding_model="embed",
            settings={"use_sentence_atomizer": False},
        )

        assert "use_sentence_atomizer: false" in content


class TestSaveApiKeyToEnv:
    """Tests for save_api_key_to_env()."""

    def test_creates_new_env_file(self, tmp_path: Path) -> None:
        """Creates .env file if it doesn't exist."""
        env_path = tmp_path / ".env"

        init_cmd.save_api_key_to_env("TEST_KEY", "test-value", env_path)

        assert env_path.exists()
        assert "TEST_KEY=test-value" in env_path.read_text()

    def test_appends_to_existing_env_file(self, tmp_path: Path) -> None:
        """Appends new key to existing .env file."""
        env_path = tmp_path / ".env"
        env_path.write_text("EXISTING_KEY=existing-value\n")

        init_cmd.save_api_key_to_env("NEW_KEY", "new-value", env_path)

        content = env_path.read_text()
        assert "EXISTING_KEY=existing-value" in content
        assert "NEW_KEY=new-value" in content

    def test_updates_existing_key(self, tmp_path: Path) -> None:
        """Updates existing key value."""
        env_path = tmp_path / ".env"
        env_path.write_text("TEST_KEY=old-value\n")

        init_cmd.save_api_key_to_env("TEST_KEY", "new-value", env_path)

        content = env_path.read_text()
        assert "TEST_KEY=new-value" in content
        assert "old-value" not in content


class TestUpdateGitignoreForEnv:
    """Tests for update_gitignore_for_env()."""

    def test_adds_env_to_empty_gitignore(self, tmp_path: Path) -> None:
        """Adds .env to empty .gitignore."""
        gitignore_path = tmp_path / ".gitignore"
        gitignore_path.write_text("")

        updated = init_cmd.update_gitignore_for_env(gitignore_path)

        assert updated is True
        assert ".env" in gitignore_path.read_text()

    def test_returns_false_if_gitignore_missing(self, tmp_path: Path) -> None:
        """Returns False if .gitignore doesn't exist."""
        gitignore_path = tmp_path / ".gitignore"

        updated = init_cmd.update_gitignore_for_env(gitignore_path)

        assert updated is False

    def test_returns_false_if_env_already_present(self, tmp_path: Path) -> None:
        """Returns False if .env already in .gitignore."""
        gitignore_path = tmp_path / ".gitignore"
        gitignore_path.write_text(".env\n")

        updated = init_cmd.update_gitignore_for_env(gitignore_path)

        assert updated is False

    def test_adds_env_when_only_env_example_present(self, tmp_path: Path) -> None:
        """Adds .env even when .env.example is already present (exact match check)."""
        gitignore_path = tmp_path / ".gitignore"
        gitignore_path.write_text(".env.example\n")

        updated = init_cmd.update_gitignore_for_env(gitignore_path)

        assert updated is True
        content = gitignore_path.read_text()
        assert ".env.example" in content
        assert "\n.env\n" in content


class TestInitNonInteractive:
    """Tests for init_non_interactive()."""

    def test_creates_config_file(self, tmp_path: Path) -> None:
        """Creates isotope.yaml config file."""
        result = init_cmd.init_non_interactive(
            provider="litellm",
            llm_model="openai/gpt-5-mini",
            embedding_model="openai/text-embedding-3-small",
            config_dir=tmp_path,
        )

        assert result.success is True
        assert result.config_path == str(tmp_path / "isotope.yaml")
        assert (tmp_path / "isotope.yaml").exists()

    def test_fails_if_config_exists_without_overwrite(self, tmp_path: Path) -> None:
        """Fails if config exists and overwrite=False."""
        config_path = tmp_path / "isotope.yaml"
        config_path.write_text("existing config")

        result = init_cmd.init_non_interactive(
            config_dir=tmp_path,
            overwrite=False,
        )

        assert result.success is False
        assert "already exists" in (result.error or "")

    def test_overwrites_if_overwrite_true(self, tmp_path: Path) -> None:
        """Overwrites config if overwrite=True."""
        config_path = tmp_path / "isotope.yaml"
        config_path.write_text("old content")

        result = init_cmd.init_non_interactive(
            config_dir=tmp_path,
            overwrite=True,
        )

        assert result.success is True
        assert "old content" not in config_path.read_text()

    def test_custom_provider_creates_template(self, tmp_path: Path) -> None:
        """Custom provider creates template config."""
        result = init_cmd.init_non_interactive(
            provider="custom",
            config_dir=tmp_path,
        )

        assert result.success is True
        content = (tmp_path / "isotope.yaml").read_text()
        assert "provider: custom" in content
        assert "embedder:" in content


class TestInit:
    """Tests for the interactive init() function."""

    def test_init_with_preset_models_skips_prompts(self, tmp_path: Path) -> None:
        """When models are preset, skips model selection prompts."""
        prompts_received: list[PromptRequest] = []

        def mock_prompt(request: PromptRequest) -> str:
            prompts_received.append(request)
            # Return defaults for all prompts
            if request.choices:
                return request.choices[0]
            return request.default or ""

        result = init_cmd.init(
            on_prompt=mock_prompt,
            llm_model="openai/gpt-5-mini",
            embedding_model="openai/text-embedding-3-small",
            config_dir=tmp_path,
        )

        assert result.success is True
        # Should not have received model selection prompts
        prompt_types = [p.prompt_type for p in prompts_received]
        assert InitPrompt.LLM_MODEL not in prompt_types
        assert InitPrompt.EMBEDDING_MODEL not in prompt_types

    def test_init_cancellation_raises_exception(self, tmp_path: Path) -> None:
        """Cancelling init raises InitCancelled."""
        config_path = tmp_path / "isotope.yaml"
        config_path.write_text("existing")

        def mock_prompt(request: PromptRequest) -> str:
            if request.prompt_type == InitPrompt.OVERWRITE_CONFIG:
                return "no"
            return request.default or ""

        with pytest.raises(init_cmd.InitCancelled):
            init_cmd.init(
                on_prompt=mock_prompt,
                config_dir=tmp_path,
            )

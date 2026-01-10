# tests/commands/test_config.py
"""Tests for the config command."""

from isotope.commands import config_cmd


class TestConfigCommand:
    """Tests for config_cmd.config()."""

    def test_config_returns_success(self) -> None:
        """Config always returns success (settings have defaults)."""
        result = config_cmd.config()

        assert result.success is True

    def test_config_has_default_provider(self) -> None:
        """Config returns default provider when not configured."""
        result = config_cmd.config()

        assert result.provider == "litellm"

    def test_config_has_settings(self) -> None:
        """Config returns list of behavioral settings."""
        result = config_cmd.config()

        assert isinstance(result.settings, list)
        assert len(result.settings) > 0

        # Check that settings have required attributes
        for setting in result.settings:
            assert hasattr(setting, "name")
            assert hasattr(setting, "value")
            assert hasattr(setting, "source")

    def test_config_includes_common_settings(self) -> None:
        """Config includes common settings like questions_per_atom."""
        result = config_cmd.config()

        setting_names = [s.name for s in result.settings]
        assert "questions_per_atom" in setting_names
        assert "use_sentence_atomizer" in setting_names

    def test_config_returns_data_dir(self) -> None:
        """Config returns data directory."""
        result = config_cmd.config()

        assert result.data_dir is not None
        assert isinstance(result.data_dir, str)

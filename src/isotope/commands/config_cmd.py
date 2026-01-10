# src/isotope/commands/config_cmd.py
"""Config command - display current configuration.

This module provides the config display logic that both CLI and TUI use.
"""

from __future__ import annotations

import os
from pathlib import Path

from isotope.commands.base import ConfigResult, SettingInfo
from isotope.config import (
    DEFAULT_DATA_DIR,
    build_settings,
    find_config_file,
    get_settings_from_env,
    get_settings_from_yaml,
    load_config,
)


def _get_setting_source(
    key: str,
    yaml_settings: dict,
    env_settings: dict,
) -> str:
    """Determine the source of a setting value."""
    if key in env_settings:
        return "env var"
    if key in yaml_settings:
        return "yaml"
    return "default"


def config(
    config_path: str | Path | None = None,
) -> ConfigResult:
    """Get current configuration settings.

    Args:
        config_path: Override config file path

    Returns:
        ConfigResult with all settings and their sources
    """
    # Load all config sources
    cli_config = load_config(config_path)
    env_settings = get_settings_from_env()
    yaml_settings = get_settings_from_yaml(cli_config)

    # Build the effective settings
    settings = build_settings(cli_config, env_settings)

    # Find config file for display
    found_config_path = find_config_file()

    # Build result
    result = ConfigResult(success=True)
    result.config_path = str(found_config_path) if found_config_path else None

    # Provider config
    result.provider = cli_config.get("provider", "litellm")

    if result.provider == "litellm":
        result.llm_model = cli_config.get("llm_model") or os.environ.get(
            "ISOTOPE_LITELLM_LLM_MODEL"
        )
        result.embedding_model = cli_config.get("embedding_model") or os.environ.get(
            "ISOTOPE_LITELLM_EMBEDDING_MODEL"
        )

    # Data directory
    result.data_dir = cli_config.get("data_dir") or DEFAULT_DATA_DIR

    # Build settings list with sources
    setting_keys = [
        ("use_sentence_atomizer", str(settings.use_sentence_atomizer)),
        ("questions_per_atom", str(settings.questions_per_atom)),
        (
            "diversity_threshold",
            str(settings.question_diversity_threshold)
            if settings.question_diversity_threshold is not None
            else "disabled",
        ),
        ("diversity_scope", settings.diversity_scope),
        ("max_concurrent_llm_calls", str(settings.max_concurrent_llm_calls)),
        ("num_retries", str(settings.num_retries)),
        ("default_k", str(settings.default_k)),
    ]

    for key, value in setting_keys:
        source_key = key
        if key == "diversity_threshold":
            source_key = "question_diversity_threshold"

        result.settings.append(
            SettingInfo(
                name=key,
                value=value,
                source=_get_setting_source(source_key, yaml_settings, env_settings),
            )
        )

    return result

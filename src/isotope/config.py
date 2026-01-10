# src/isotope/config.py
"""Configuration loading utilities for Isotope.

This module provides configuration loading that can be used by:
- CLI commands
- TUI screens
- External applications using Isotope as a library

It handles:
- Finding and loading isotope.yaml config files
- Loading .env files for API keys
- Building Settings objects from multiple sources
- Creating Isotope instances from configuration
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict, cast

if TYPE_CHECKING:
    from isotope.isotope import Isotope
    from isotope.settings import Settings
    from isotope.stores import (
        ChromaEmbeddedQuestionStore,
        SQLiteAtomStore,
        SQLiteChunkStore,
        SQLiteSourceRegistry,
    )

try:
    import yaml  # type: ignore[import-untyped]

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from isotope.question_generator import FilterScope

# Default paths
DEFAULT_DATA_DIR = "./isotope_data"
CONFIG_FILES = ["isotope.yaml", "isotope.yml", ".isotoperc"]
ENV_FILE = ".env"


class StoreBundle(TypedDict):
    """Bundle of store instances for read-only operations."""

    embedded_question_store: ChromaEmbeddedQuestionStore
    chunk_store: SQLiteChunkStore
    atom_store: SQLiteAtomStore
    source_registry: SQLiteSourceRegistry


@dataclass
class ConfigError:
    """Error during configuration loading."""

    message: str
    suggestion: str | None = None


def load_env_file(env_path: str | Path = ENV_FILE) -> None:
    """Load environment variables from .env file if it exists.

    This loads API keys saved by `isotope init` so they're available
    for subsequent operations.

    Args:
        env_path: Path to .env file (default: .env in current directory)
    """
    path = Path(env_path)
    if not path.exists():
        return

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()
                # Don't override existing env vars
                if key not in os.environ:
                    os.environ[key] = value


def find_config_file(start_dir: Path | None = None) -> Path | None:
    """Find configuration file in current directory or parent directories.

    Args:
        start_dir: Directory to start searching from (default: cwd)

    Returns:
        Path to config file if found, None otherwise
    """
    current = start_dir or Path.cwd()
    for _ in range(10):  # Limit search depth
        for config_name in CONFIG_FILES:
            config_path = current / config_name
            if config_path.exists():
                return config_path
        parent = current.parent
        if parent == current:
            break
        current = parent
    return None


# Valid configuration keys for validation
VALID_ROOT_KEYS = {
    # Provider config
    "provider",
    "llm_model",
    "embedding_model",
    "data_dir",
    # Custom provider
    "embedder",
    "question_generator",
    "atomizer",
    "embedder_kwargs",
    "question_generator_kwargs",
    "atomizer_kwargs",
    # Settings section
    "settings",
}

VALID_SETTINGS_KEYS = {
    "use_sentence_atomizer",
    "questions_per_atom",
    "question_generator_prompt",
    "atomizer_prompt",
    "question_diversity_threshold",
    "diversity_threshold",  # alias
    "diversity_scope",
    "default_k",
    "synthesis_prompt",
    "synthesis_temperature",
    "max_concurrent_llm_calls",
    "num_retries",
    "rate_limit_profile",
    "batch_size",
    "generation_preset",
}


def validate_config(config: dict[str, Any], config_path: Path | None = None) -> list[str]:
    """Validate config and return warnings about unknown keys.

    Args:
        config: The loaded configuration dictionary
        config_path: Path to config file (for error messages)

    Returns:
        List of warning messages (empty if no issues)
    """
    warnings = []

    # Check root level keys
    unknown_root = set(config.keys()) - VALID_ROOT_KEYS
    if unknown_root:
        path_str = str(config_path) if config_path else "config"
        warnings.append(f"Unknown config keys in {path_str}: {', '.join(sorted(unknown_root))}")

    # Check settings section keys
    settings = config.get("settings", {})
    if isinstance(settings, dict):
        unknown_settings = set(settings.keys()) - VALID_SETTINGS_KEYS
        if unknown_settings:
            warnings.append(f"Unknown settings keys: {', '.join(sorted(unknown_settings))}")

    return warnings


def load_config(config_path: Path | str | None = None) -> dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Explicit path to config file, or None to search

    Returns:
        Configuration dictionary (empty if no config found)
    """
    config_path = Path(config_path) if config_path is not None else find_config_file()

    if config_path is None:
        return {}

    if not YAML_AVAILABLE:
        # Can't load YAML without pyyaml
        return {}

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    return config


def _parse_threshold(value: str | None) -> float | None:
    """Parse diversity threshold from string, treating empty string as None."""
    if value is None or value == "":
        return None
    return float(value)


def _parse_diversity_scope(value: str | None) -> FilterScope:
    """Parse diversity scope from string, defaulting to 'global'."""
    if value is None or value == "":
        return "global"
    normalized = value.lower()
    if normalized not in {"global", "per_chunk", "per_atom"}:
        return "global"
    return cast(FilterScope, normalized)


def _safe_int(value: str | None) -> int | None:
    """Parse int from string, returning None on invalid value."""
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def get_settings_from_env() -> dict[str, Any]:
    """Read behavioral settings from ISOTOPE_* environment variables.

    Returns values that were explicitly set (not defaults), to allow proper
    precedence: YAML settings are used unless overridden by env vars.

    Returns:
        Dictionary of setting name -> value for explicitly set env vars
    """
    result: dict[str, Any] = {}

    # Only include values that are explicitly set in environment
    if (val := _safe_int(os.environ.get("ISOTOPE_QUESTIONS_PER_ATOM"))) is not None:
        result["questions_per_atom"] = val
    if "ISOTOPE_QUESTION_GENERATOR_PROMPT" in os.environ:
        result["question_generator_prompt"] = (
            os.environ["ISOTOPE_QUESTION_GENERATOR_PROMPT"] or None
        )
    if "ISOTOPE_ATOMIZER_PROMPT" in os.environ:
        result["atomizer_prompt"] = os.environ["ISOTOPE_ATOMIZER_PROMPT"] or None
    if "ISOTOPE_QUESTION_DIVERSITY_THRESHOLD" in os.environ:
        result["question_diversity_threshold"] = _parse_threshold(
            os.environ["ISOTOPE_QUESTION_DIVERSITY_THRESHOLD"]
        )
    if "ISOTOPE_DIVERSITY_SCOPE" in os.environ:
        result["diversity_scope"] = _parse_diversity_scope(os.environ["ISOTOPE_DIVERSITY_SCOPE"])
    if (val := _safe_int(os.environ.get("ISOTOPE_DEFAULT_K"))) is not None:
        result["default_k"] = val
    if "ISOTOPE_SYNTHESIS_PROMPT" in os.environ:
        result["synthesis_prompt"] = os.environ["ISOTOPE_SYNTHESIS_PROMPT"] or None
    if (val := _safe_int(os.environ.get("ISOTOPE_MAX_CONCURRENT_LLM_CALLS"))) is not None:
        result["max_concurrent_llm_calls"] = val
    if (val := _safe_int(os.environ.get("ISOTOPE_NUM_RETRIES"))) is not None:
        result["num_retries"] = val
    if "ISOTOPE_RATE_LIMIT_PROFILE" in os.environ:
        result["rate_limit_profile"] = os.environ["ISOTOPE_RATE_LIMIT_PROFILE"]
    if "ISOTOPE_USE_SENTENCE_ATOMIZER" in os.environ:
        result["use_sentence_atomizer"] = os.environ["ISOTOPE_USE_SENTENCE_ATOMIZER"].lower() in (
            "true",
            "1",
            "yes",
        )

    return result


def get_settings_from_yaml(config: dict[str, Any]) -> dict[str, Any]:
    """Extract settings from YAML config file.

    Settings can be in a 'settings:' section or at the root level for backwards compat.

    Args:
        config: The loaded YAML configuration

    Returns:
        Dictionary of setting name -> value
    """
    result: dict[str, Any] = {}

    # Get settings from the 'settings:' section if present
    yaml_settings = config.get("settings", {}) or {}

    # Map YAML keys to Settings field names
    key_mappings = {
        "use_sentence_atomizer": "use_sentence_atomizer",
        "questions_per_atom": "questions_per_atom",
        "question_generator_prompt": "question_generator_prompt",
        "atomizer_prompt": "atomizer_prompt",
        "question_diversity_threshold": "question_diversity_threshold",
        "diversity_threshold": "question_diversity_threshold",  # alias
        "diversity_scope": "diversity_scope",
        "default_k": "default_k",
        "synthesis_prompt": "synthesis_prompt",
        "synthesis_temperature": "synthesis_temperature",
        "max_concurrent_llm_calls": "max_concurrent_llm_calls",
        "num_retries": "num_retries",
        "rate_limit_profile": "rate_limit_profile",
        "batch_size": "batch_size",
        "generation_preset": "generation_preset",
    }

    for yaml_key, settings_key in key_mappings.items():
        if yaml_key in yaml_settings:
            result[settings_key] = yaml_settings[yaml_key]

    return result


def build_settings(
    config: dict[str, Any] | None = None,
    env_settings: dict[str, Any] | None = None,
) -> Settings:
    """Build Settings object from YAML config and env vars.

    Precedence (highest to lowest):
    1. Environment variables (for CI/CD override)
    2. YAML settings: section
    3. Settings class defaults

    Args:
        config: YAML configuration dictionary
        env_settings: Environment variable overrides (if None, reads from env)

    Returns:
        Configured Settings instance
    """
    from isotope.settings import Settings

    config = config or {}
    yaml_settings = get_settings_from_yaml(config)
    env_settings = env_settings if env_settings is not None else get_settings_from_env()

    # Merge: env vars override YAML, which overrides defaults
    merged = {**yaml_settings, **env_settings}

    # Handle rate_limit_profile specially - it affects multiple settings
    rate_limit_profile = merged.pop("rate_limit_profile", None)

    if rate_limit_profile:
        # Start with profile, then apply overrides
        settings = Settings.with_profile(rate_limit_profile, **merged)
    else:
        # Use merged settings directly
        settings = Settings(**merged)

    return settings


def get_stores(data_dir: str | Path) -> StoreBundle:
    """Get store instances for read-only operations (list, status, delete).

    This doesn't require provider configuration since it only accesses stores.

    Args:
        data_dir: Path to data directory

    Returns:
        Bundle of store instances
    """
    from isotope.stores import (
        ChromaEmbeddedQuestionStore,
        SQLiteAtomStore,
        SQLiteChunkStore,
        SQLiteSourceRegistry,
    )

    data_dir = str(data_dir)
    return {
        "embedded_question_store": ChromaEmbeddedQuestionStore(os.path.join(data_dir, "chroma")),
        "chunk_store": SQLiteChunkStore(os.path.join(data_dir, "chunks.db")),
        "atom_store": SQLiteAtomStore(os.path.join(data_dir, "atoms.db")),
        "source_registry": SQLiteSourceRegistry(os.path.join(data_dir, "sources.db")),
    }


def import_class(class_path: str) -> type[Any]:
    """Import a class from a dotted path like 'my_package.module.ClassName'.

    Args:
        class_path: Dotted path to class

    Returns:
        The imported class
    """
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return cast(type[Any], getattr(module, class_name))


def is_local_model(model: str) -> bool:
    """Check if a model is a local model (doesn't need API key).

    Args:
        model: Model name (e.g., "ollama/llama3", "gpt-4o-mini")

    Returns:
        True if the model runs locally
    """
    model_lower = model.lower()
    return any(
        pattern in model_lower
        for pattern in [
            "ollama",
            "local",
            "llama.cpp",
            "llamacpp",
            "gguf",
            "ggml",
        ]
    )


@dataclass
class IsotopeConfig:
    """Configuration for creating an Isotope instance."""

    provider: str
    llm_model: str | None
    embedding_model: str | None
    data_dir: str
    settings: Settings
    llm_api_key: str | None = None
    embedding_api_key: str | None = None
    # Custom provider fields
    embedder_class: str | None = None
    question_generator_class: str | None = None
    atomizer_class: str | None = None
    embedder_kwargs: dict[str, Any] | None = None
    question_generator_kwargs: dict[str, Any] | None = None
    atomizer_kwargs: dict[str, Any] | None = None


def get_isotope_config(
    data_dir: str | None = None,
    config_path: str | Path | None = None,
) -> IsotopeConfig | ConfigError:
    """Get configuration for creating an Isotope instance.

    This extracts configuration without creating the instance, allowing
    the caller to handle errors and missing values appropriately.

    Args:
        data_dir: Override data directory
        config_path: Override config file path

    Returns:
        IsotopeConfig with all settings, or ConfigError if invalid
    """
    config = load_config(config_path)
    effective_data_dir = data_dir or config.get("data_dir") or DEFAULT_DATA_DIR
    provider = config.get("provider", "litellm")

    # Build settings from YAML and env vars
    env_settings = get_settings_from_env()
    settings = build_settings(config, env_settings)

    if provider == "litellm":
        llm_model = config.get("llm_model") or os.environ.get("ISOTOPE_LITELLM_LLM_MODEL")
        embedding_model = config.get("embedding_model") or os.environ.get(
            "ISOTOPE_LITELLM_EMBEDDING_MODEL"
        )

        if not llm_model or not embedding_model:
            return ConfigError(
                message="LiteLLM provider requires llm_model and embedding_model.",
                suggestion="Run 'isotope init' to configure, or set in isotope.yaml",
            )

        llm_api_key = os.environ.get("ISOTOPE_LLM_API_KEY")
        embedding_api_key = os.environ.get("ISOTOPE_EMBEDDING_API_KEY")

        return IsotopeConfig(
            provider=provider,
            llm_model=llm_model,
            embedding_model=embedding_model,
            data_dir=effective_data_dir,
            settings=settings,
            llm_api_key=llm_api_key,
            embedding_api_key=embedding_api_key,
        )

    elif provider == "custom":
        embedder_class = config.get("embedder")
        question_generator_class = config.get("question_generator")
        atomizer_class = config.get("atomizer")

        if not all([embedder_class, question_generator_class, atomizer_class]):
            return ConfigError(
                message="Custom provider requires embedder, question_generator, and atomizer.",
                suggestion="Add these to isotope.yaml as dotted class paths",
            )

        return IsotopeConfig(
            provider=provider,
            llm_model=None,
            embedding_model=None,
            data_dir=effective_data_dir,
            settings=settings,
            embedder_class=embedder_class,
            question_generator_class=question_generator_class,
            atomizer_class=atomizer_class,
            embedder_kwargs=config.get("embedder_kwargs", {}),
            question_generator_kwargs=config.get("question_generator_kwargs", {}),
            atomizer_kwargs=config.get("atomizer_kwargs", {}),
        )

    else:
        return ConfigError(
            message=f"Unknown provider '{provider}'",
            suggestion="Supported providers: litellm, custom",
        )


def create_isotope(config: IsotopeConfig) -> Isotope:
    """Create an Isotope instance from configuration.

    Args:
        config: Configuration for the Isotope instance

    Returns:
        Configured Isotope instance

    Raises:
        ImportError: If custom provider classes cannot be imported
    """
    from isotope.configuration import LiteLLMProvider, LocalStorage
    from isotope.isotope import Isotope

    if config.provider == "litellm":
        if not config.llm_model or not config.embedding_model:
            raise ValueError("LiteLLM provider requires llm_model and embedding_model")

        return Isotope(
            provider=LiteLLMProvider(
                llm=config.llm_model,
                embedding=config.embedding_model,
                atomizer_type="sentence" if config.settings.use_sentence_atomizer else "llm",
                llm_api_key=config.llm_api_key,
                embedding_api_key=config.embedding_api_key,
            ),
            storage=LocalStorage(config.data_dir),
            settings=config.settings,
        )

    elif config.provider == "custom":
        from dataclasses import dataclass as dc

        if not all([config.embedder_class, config.question_generator_class, config.atomizer_class]):
            raise ValueError("Custom provider requires all class paths")

        # Narrowed by the check above
        assert config.embedder_class is not None
        assert config.question_generator_class is not None
        assert config.atomizer_class is not None

        embedder_cls = import_class(config.embedder_class)
        question_generator_cls = import_class(config.question_generator_class)
        atomizer_cls = import_class(config.atomizer_class)

        embedder = embedder_cls(**(config.embedder_kwargs or {}))
        question_generator = question_generator_cls(**(config.question_generator_kwargs or {}))
        atomizer = atomizer_cls(**(config.atomizer_kwargs or {}))

        @dc(frozen=True)
        class _CustomProvider:
            """Inline provider for custom implementations."""

            _embedder: Any
            _atomizer: Any
            _question_generator: Any

            def build_embedder(self, settings: Settings) -> Any:
                return self._embedder

            def build_atomizer(self, settings: Settings) -> Any:
                return self._atomizer

            def build_question_generator(self, settings: Settings) -> Any:
                return self._question_generator

            def build_llm_client(self, settings: Settings | None = None) -> Any:
                raise NotImplementedError(
                    "Custom provider does not support build_llm_client. "
                    "Use --raw flag with 'isotope query' or extend your provider."
                )

        return Isotope(
            provider=_CustomProvider(
                _embedder=embedder,
                _atomizer=atomizer,
                _question_generator=question_generator,
            ),
            storage=LocalStorage(config.data_dir),
            settings=config.settings,
        )

    else:
        raise ValueError(f"Unknown provider: {config.provider}")


def get_isotope(
    data_dir: str | None = None,
    config_path: str | Path | None = None,
) -> Isotope | ConfigError:
    """Create an Isotope instance based on configuration.

    This is a convenience function that combines get_isotope_config and
    create_isotope. For more control, use those functions separately.

    Args:
        data_dir: Override data directory
        config_path: Override config file path

    Returns:
        Configured Isotope instance, or ConfigError if configuration is invalid
    """
    config = get_isotope_config(data_dir, config_path)
    if isinstance(config, ConfigError):
        return config
    return create_isotope(config)

# src/isotope/cli.py
"""Command-line interface for Isotope.

The CLI uses a configuration file (isotope.yaml) to determine which
provider to use. See docs for configuration options.
"""

from __future__ import annotations

import importlib
import os
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
    import typer
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
    from rich.table import Table
except ImportError as e:
    raise SystemExit(
        "CLI requires additional dependencies.\nInstall with: pip install isotope-rag[cli]"
    ) from e

try:
    import yaml  # type: ignore[import-untyped]

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from isotope import __version__
from isotope.question_generator import FilterScope

app = typer.Typer(
    name="isotope",
    help="Isotope - Reverse RAG database. Index questions, not chunks.",
    no_args_is_help=True,
)
questions_app = typer.Typer(help="Inspect generated questions")
app.add_typer(questions_app, name="questions")
console = Console()


def version_callback(value: bool) -> None:
    if value:
        console.print(f"isotope {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """Isotope - Reverse RAG database."""
    _load_env_file()


DEFAULT_DATA_DIR = "./isotope_data"
CONFIG_FILES = ["isotope.yaml", "isotope.yml", ".isotoperc"]
ENV_FILE = ".env"


def _load_env_file() -> None:
    """Load environment variables from .env file if it exists.

    This is called automatically at CLI startup so that API keys saved by
    `isotope init` are available for subsequent commands.
    """
    env_path = Path(ENV_FILE)
    if not env_path.exists():
        return

    with open(env_path) as f:
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


def _collect_api_keys() -> tuple[str | None, str | None]:
    """Collect API keys from user interactively.

    Returns:
        Tuple of (llm_api_key, embedding_api_key). Either can be None if not needed.
    """
    console.print()
    llm_key = typer.prompt(
        "Enter your LLM API key (leave empty if not needed)",
        default="",
        show_default=False,
        hide_input=True,
    )
    llm_key = llm_key.strip() if llm_key else None

    if not llm_key:
        return None, None

    # Ask about embedding API key
    console.print()
    embed_choice = typer.prompt(
        "Enter your embedding API key:\n"
        "  [1] Same as LLM\n"
        "  [2] None (not needed)\n"
        "  [3] Different key\n"
        "Choose",
        default="1",
    )

    if embed_choice == "1":
        embed_key = llm_key
    elif embed_choice == "3":
        embed_key = typer.prompt(
            "Enter embedding API key",
            default="",
            show_default=False,
            hide_input=True,
        )
        embed_key = embed_key.strip() if embed_key else None
    else:
        embed_key = None

    return llm_key, embed_key


def _save_api_key_to_env(env_var: str, api_key: str) -> bool:
    """Save API key to .env file."""
    env_path = Path(ENV_FILE)

    # Read existing content
    existing_lines: list[str] = []
    if env_path.exists():
        with open(env_path) as f:
            existing_lines = f.readlines()

    # Check if key already exists
    key_exists = False
    new_lines = []
    for line in existing_lines:
        if line.strip().startswith(f"{env_var}="):
            new_lines.append(f"{env_var}={api_key}\n")
            key_exists = True
        else:
            new_lines.append(line)

    # Add key if it doesn't exist
    if not key_exists:
        if new_lines and not new_lines[-1].endswith("\n"):
            new_lines.append("\n")
        new_lines.append(f"{env_var}={api_key}\n")

    # Write file
    with open(env_path, "w") as f:
        f.writelines(new_lines)

    return True


def _save_api_keys_and_update_gitignore(llm_key: str | None, embed_key: str | None) -> None:
    """Save API keys to .env and ensure .env is in .gitignore."""
    saved_any = False

    if llm_key:
        _save_api_key_to_env("ISOTOPE_LLM_API_KEY", llm_key)
        os.environ["ISOTOPE_LLM_API_KEY"] = llm_key
        saved_any = True

    if embed_key and embed_key != llm_key:
        _save_api_key_to_env("ISOTOPE_EMBEDDING_API_KEY", embed_key)
        os.environ["ISOTOPE_EMBEDDING_API_KEY"] = embed_key
        saved_any = True

    if saved_any:
        console.print(f"[green]Saved API key(s) to {ENV_FILE}[/green]")

        # Ensure .env is in .gitignore
        gitignore_path = Path(".gitignore")
        if gitignore_path.exists():
            content = gitignore_path.read_text()
            if ".env" not in content:
                with open(gitignore_path, "a") as f:
                    f.write("\n# Environment variables\n.env\n")
                console.print("[dim]Added .env to .gitignore[/dim]")


def _parse_threshold(value: str | None) -> float | None:
    """Parse diversity threshold from env var, treating empty string as None."""
    if value is None or value == "":
        return None
    return float(value)


def _parse_diversity_scope(value: str | None) -> FilterScope:
    """Parse diversity scope from env var, defaulting to 'global'."""
    if value is None or value == "":
        return "global"
    normalized = value.lower()
    if normalized not in {"global", "per_chunk", "per_atom"}:
        console.print(f"[yellow]Warning: Invalid diversity_scope '{value}', defaulting[/yellow]")
        return "global"
    return cast(FilterScope, normalized)


def _safe_int_env(env_var: str) -> int | None:
    """Parse int from env var with user-friendly error on invalid value."""
    value = os.environ.get(env_var)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        console.print(
            f"[yellow]Warning: Invalid integer for {env_var}='{value}', ignoring[/yellow]"
        )
        return None


def _get_behavioral_settings_from_env() -> dict:
    """Read behavioral settings from ISOTOPE_* environment variables.

    This is the CLI (application layer) reading env vars and preparing
    them to pass explicitly to the library.

    Returns values that were explicitly set (not defaults), to allow proper
    precedence: YAML settings are used unless overridden by env vars.
    """
    result: dict = {}

    # Only include values that are explicitly set in environment
    if (val := _safe_int_env("ISOTOPE_QUESTIONS_PER_ATOM")) is not None:
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
    if (val := _safe_int_env("ISOTOPE_DEFAULT_K")) is not None:
        result["default_k"] = val
    if "ISOTOPE_SYNTHESIS_PROMPT" in os.environ:
        result["synthesis_prompt"] = os.environ["ISOTOPE_SYNTHESIS_PROMPT"] or None
    if (val := _safe_int_env("ISOTOPE_MAX_CONCURRENT_LLM_CALLS")) is not None:
        result["max_concurrent_llm_calls"] = val
    if (val := _safe_int_env("ISOTOPE_NUM_RETRIES")) is not None:
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


def _get_settings_from_yaml(config: dict) -> dict:
    """Extract settings from YAML config file.

    Settings can be in a 'settings:' section or at the root level for backwards compat.
    """
    result: dict = {}

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

    # Also check root level for backwards compat (use_sentence_atomizer, rate_limit_profile)
    if "use_sentence_atomizer" in config and "use_sentence_atomizer" not in result:
        result["use_sentence_atomizer"] = config["use_sentence_atomizer"]
    if "rate_limit_profile" in config and "rate_limit_profile" not in result:
        result["rate_limit_profile"] = config["rate_limit_profile"]

    return result


def _build_settings(config: dict, env_settings: dict) -> Settings:
    """Build Settings object from YAML config and env vars.

    Precedence (highest to lowest):
    1. Environment variables (for CI/CD override)
    2. YAML settings: section
    3. Settings class defaults
    """
    from isotope.settings import Settings

    yaml_settings = _get_settings_from_yaml(config)

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


class StoreBundle(TypedDict):
    embedded_question_store: ChromaEmbeddedQuestionStore
    chunk_store: SQLiteChunkStore
    atom_store: SQLiteAtomStore
    source_registry: SQLiteSourceRegistry


def get_stores(data_dir: str) -> StoreBundle:
    """Get store instances for read-only operations (list, status, delete).

    This doesn't require provider configuration since it only accesses stores.
    """
    from isotope.stores import (
        ChromaEmbeddedQuestionStore,
        SQLiteAtomStore,
        SQLiteChunkStore,
        SQLiteSourceRegistry,
    )

    return {
        "embedded_question_store": ChromaEmbeddedQuestionStore(os.path.join(data_dir, "chroma")),
        "chunk_store": SQLiteChunkStore(os.path.join(data_dir, "chunks.db")),
        "atom_store": SQLiteAtomStore(os.path.join(data_dir, "atoms.db")),
        "source_registry": SQLiteSourceRegistry(os.path.join(data_dir, "sources.db")),
    }


def find_config_file() -> Path | None:
    """Find configuration file in current directory or parent directories."""
    current = Path.cwd()
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


# Known valid config keys for YAML validation
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
    # Legacy/backwards compat
    "use_sentence_atomizer",
    "rate_limit_profile",
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


def _validate_config(config: dict[str, Any], config_path: Path) -> None:
    """Validate config and warn about unknown keys."""
    # Check root level keys
    unknown_root = set(config.keys()) - VALID_ROOT_KEYS
    if unknown_root:
        console.print(
            f"[yellow]Warning: Unknown config keys in {config_path}: "
            f"{', '.join(sorted(unknown_root))}[/yellow]"
        )
        console.print("[dim]These keys will be ignored. Check for typos.[/dim]")

    # Check settings section keys
    settings = config.get("settings", {})
    if isinstance(settings, dict):
        unknown_settings = set(settings.keys()) - VALID_SETTINGS_KEYS
        if unknown_settings:
            console.print(
                f"[yellow]Warning: Unknown settings keys: "
                f"{', '.join(sorted(unknown_settings))}[/yellow]"
            )
            console.print("[dim]These settings will be ignored. Check for typos.[/dim]")


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = find_config_file()

    if config_path is None:
        return {}

    if not YAML_AVAILABLE:
        console.print("[yellow]Warning: PyYAML not installed. Config file ignored.[/yellow]")
        console.print("[dim]Install with: pip install pyyaml[/dim]")
        return {}

    with open(config_path) as f:
        config = yaml.safe_load(f) or {}

    # Validate config and warn about unknown keys
    _validate_config(config, config_path)

    return config


def import_class(class_path: str) -> type[Any]:
    """Import a class from a dotted path like 'my_package.module.ClassName'."""
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return cast(type[Any], getattr(module, class_name))


def get_isotope(
    data_dir: str | None = None,
    config_path: str | None = None,
) -> Isotope:
    """Create an Isotope instance based on configuration.

    Configuration is loaded from (highest to lowest precedence):
    1. Environment variables (for CI/CD override)
    2. YAML config file (isotope.yaml settings: section)
    3. Settings class defaults

    Provider config is loaded from:
    1. Explicit config_path if provided
    2. isotope.yaml in current/parent directories
    3. Falls back to LiteLLM with env vars if no config found
    """
    from dataclasses import dataclass

    from isotope.configuration import LiteLLMProvider, LocalStorage
    from isotope.isotope import Isotope

    config = load_config(Path(config_path) if config_path else None)
    effective_data_dir = data_dir or config.get("data_dir") or DEFAULT_DATA_DIR

    provider = config.get("provider", "litellm")

    # Build settings from YAML and env vars (env vars override YAML)
    env_settings = _get_behavioral_settings_from_env()
    settings = _build_settings(config, env_settings)

    if provider == "litellm":
        # LiteLLM provider
        llm_model = config.get("llm_model") or os.environ.get("ISOTOPE_LITELLM_LLM_MODEL")
        embedding_model = config.get("embedding_model") or os.environ.get(
            "ISOTOPE_LITELLM_EMBEDDING_MODEL"
        )

        if not llm_model or not embedding_model:
            console.print(
                "[red]Error: LiteLLM provider requires llm_model and embedding_model.[/red]"
            )
            console.print()
            console.print("Either create isotope.yaml:")
            console.print("  provider: litellm")
            console.print("  llm_model: openai/gpt-5-mini-2025-08-07")
            console.print("  embedding_model: openai/text-embedding-3-small")
            console.print()
            console.print("Or set environment variables:")
            console.print("  export ISOTOPE_LITELLM_LLM_MODEL=openai/gpt-5-mini-2025-08-07")
            console.print("  export ISOTOPE_LITELLM_EMBEDDING_MODEL=openai/text-embedding-3-small")
            raise typer.Exit(1)

        # Check for API key and warn if missing (for non-local models)
        if not _is_local_model(llm_model) and not os.environ.get("ISOTOPE_LLM_API_KEY"):
            console.print()
            console.print("[yellow]Warning: No ISOTOPE_LLM_API_KEY found.[/yellow]")
            console.print("Run 'isotope init' to configure your API key, or set it manually:")
            console.print("  export ISOTOPE_LLM_API_KEY=your-api-key")
            console.print()
            console.print(
                "[dim]Provider-specific env vars (e.g., OPENAI_API_KEY) may still work.[/dim]"
            )
            console.print()

        # use_sentence_atomizer now comes from Settings (via YAML or env var)
        # API keys from env vars (set by isotope init or manually)
        llm_api_key = os.environ.get("ISOTOPE_LLM_API_KEY")
        embedding_api_key = os.environ.get("ISOTOPE_EMBEDDING_API_KEY")
        return Isotope(
            provider=LiteLLMProvider(
                llm=llm_model,
                embedding=embedding_model,
                atomizer_type="sentence" if settings.use_sentence_atomizer else "llm",
                llm_api_key=llm_api_key,
                embedding_api_key=embedding_api_key,
            ),
            storage=LocalStorage(effective_data_dir),
            settings=settings,
        )

    elif provider == "custom":
        # Custom provider - import classes dynamically
        embedder_class = config.get("embedder")
        question_generator_class = config.get("question_generator")
        atomizer_class = config.get("atomizer")

        if not all([embedder_class, question_generator_class, atomizer_class]):
            console.print(
                "[red]Error: Custom provider requires embedder, "
                "question_generator, and atomizer.[/red]"
            )
            console.print()
            console.print("Example isotope.yaml:")
            console.print("  provider: custom")
            console.print("  embedder: my_package.MyEmbedder")
            console.print("  question_generator: my_package.MyGenerator")
            console.print("  atomizer: my_package.MyAtomizer")
            raise typer.Exit(1)

        if (
            not isinstance(embedder_class, str)
            or not isinstance(question_generator_class, str)
            or not isinstance(atomizer_class, str)
        ):
            console.print("[red]Error: Custom provider class paths must be strings.[/red]")
            raise typer.Exit(1)

        try:
            embedder_cls = import_class(embedder_class)
            question_generator_cls = import_class(question_generator_class)
            atomizer_cls = import_class(atomizer_class)
        except (ImportError, AttributeError) as e:
            console.print(f"[red]Error importing custom class: {e}[/red]")
            raise typer.Exit(1) from None

        # Get kwargs for each class
        embedder_kwargs = config.get("embedder_kwargs", {})
        question_generator_kwargs = config.get("question_generator_kwargs", {})
        atomizer_kwargs = config.get("atomizer_kwargs", {})

        embedder = embedder_cls(**embedder_kwargs)
        question_generator = question_generator_cls(**question_generator_kwargs)
        atomizer = atomizer_cls(**atomizer_kwargs)

        # Create an inline provider that wraps the custom components
        @dataclass(frozen=True)
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
            storage=LocalStorage(effective_data_dir),
            settings=settings,
        )

    else:
        console.print(f"[red]Error: Unknown provider '{provider}'[/red]")
        console.print("Supported providers: litellm, custom")
        raise typer.Exit(1)


def _get_setting_source(
    key: str, yaml_settings: dict, env_settings: dict, yaml_config: dict
) -> str:
    """Determine the source of a setting value."""
    if key in env_settings:
        return "env var"
    if key in yaml_settings:
        return "yaml"
    # Check root-level for backwards compat
    if key in yaml_config:
        return "yaml (root)"
    return "default"


@app.command()
def config(
    config_file: str = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to config file",
    ),
) -> None:
    """Show current configuration settings."""

    # Load all config sources
    cli_config = load_config(Path(config_file) if config_file else None)
    env_settings = _get_behavioral_settings_from_env()
    yaml_settings = _get_settings_from_yaml(cli_config)

    # Build the effective settings
    settings = _build_settings(cli_config, env_settings)

    table = Table(title="Isotope Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Source", style="dim")

    # Provider config (from config file or env)
    provider = cli_config.get("provider", "litellm")
    table.add_row("provider", provider, "yaml" if cli_config else "default")

    if provider == "litellm":
        llm_model = cli_config.get("llm_model") or os.environ.get("ISOTOPE_LITELLM_LLM_MODEL")
        embedding_model = cli_config.get("embedding_model") or os.environ.get(
            "ISOTOPE_LITELLM_EMBEDDING_MODEL"
        )
        table.add_row(
            "llm_model",
            llm_model or "(not set)",
            "yaml" if cli_config.get("llm_model") else "env var",
        )
        table.add_row(
            "embedding_model",
            embedding_model or "(not set)",
            "yaml" if cli_config.get("embedding_model") else "env var",
        )
    elif provider == "custom":
        table.add_row("embedder", cli_config.get("embedder", "(not set)"), "yaml")
        table.add_row(
            "question_generator", cli_config.get("question_generator", "(not set)"), "yaml"
        )
        table.add_row("atomizer", cli_config.get("atomizer", "(not set)"), "yaml")

    # Data directory
    data_dir = cli_config.get("data_dir") or DEFAULT_DATA_DIR
    table.add_row(
        "data_dir",
        data_dir,
        "yaml" if cli_config.get("data_dir") else "default",
    )

    # Separator for behavioral settings
    table.add_row("", "", "")

    # Behavioral settings - show all with effective values and sources
    table.add_row(
        "use_sentence_atomizer",
        str(settings.use_sentence_atomizer),
        _get_setting_source("use_sentence_atomizer", yaml_settings, env_settings, cli_config),
    )
    table.add_row(
        "questions_per_atom",
        str(settings.questions_per_atom),
        _get_setting_source("questions_per_atom", yaml_settings, env_settings, cli_config),
    )
    table.add_row(
        "diversity_threshold",
        str(settings.question_diversity_threshold)
        if settings.question_diversity_threshold is not None
        else "disabled",
        _get_setting_source(
            "question_diversity_threshold", yaml_settings, env_settings, cli_config
        ),
    )
    table.add_row(
        "diversity_scope",
        settings.diversity_scope,
        _get_setting_source("diversity_scope", yaml_settings, env_settings, cli_config),
    )
    table.add_row(
        "max_concurrent_llm_calls",
        str(settings.max_concurrent_llm_calls),
        _get_setting_source("max_concurrent_llm_calls", yaml_settings, env_settings, cli_config),
    )
    table.add_row(
        "num_retries",
        str(settings.num_retries),
        _get_setting_source("num_retries", yaml_settings, env_settings, cli_config),
    )
    table.add_row(
        "default_k",
        str(settings.default_k),
        _get_setting_source("default_k", yaml_settings, env_settings, cli_config),
    )

    console.print(table)

    # Show config file location
    config_path = find_config_file()
    if config_path:
        console.print(f"\n[dim]Config file: {config_path}[/dim]")
    else:
        console.print("\n[dim]No config file found. Using env vars / defaults.[/dim]")

    console.print("\n[dim]Precedence: env var > yaml settings > default[/dim]")


@app.command()
def ingest(
    path: str = typer.Argument(..., help="File or directory to ingest"),
    data_dir: str = typer.Option(
        None,
        "--data-dir",
        "-d",
        help="Data directory (default: from settings)",
    ),
    config_file: str = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to config file",
    ),
    plain: bool = typer.Option(
        False,
        "--plain",
        help="Plain output (no colors/formatting)",
    ),
    no_progress: bool = typer.Option(
        False,
        "--no-progress",
        help="Disable progress bars",
    ),
) -> None:
    """Ingest a file or directory into the database."""
    from isotope.loaders import LoaderRegistry

    # Check path exists
    if not os.path.exists(path):
        console.print(f"[red]Error: Path not found: {path}[/red]")
        raise typer.Exit(1)

    # Create Isotope
    try:
        iso = get_isotope(data_dir, config_file)
    except Exception as e:
        console.print(f"[red]Error creating isotope: {e}[/red]")
        raise typer.Exit(1) from None

    # Find files to ingest
    registry = LoaderRegistry.default()

    if os.path.isfile(path):
        files = [path]
    else:
        # Directory - find all supported files
        files = []
        for root, _, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(root, filename)
                if registry.find_loader(filepath):
                    files.append(filepath)

    if not files:
        console.print("[yellow]No supported files found.[/yellow]")
        raise typer.Exit(0)

    # Ingest each file
    total_chunks = 0
    total_atoms = 0
    total_questions = 0
    total_questions_filtered = 0
    skipped_count = 0
    ingested_count = 0
    failed_count = 0

    # Determine if we should show progress bars
    show_progress = not plain and not no_progress and console.is_terminal

    def process_result(result: dict, filepath: str) -> None:
        """Process ingestion result and update counters."""
        nonlocal total_chunks, total_atoms, total_questions, total_questions_filtered
        nonlocal skipped_count, ingested_count

        if result.get("skipped"):
            skipped_count += 1
        else:
            ingested_count += 1
            total_chunks += result.get("chunks", 0)
            total_atoms += result.get("atoms", 0)
            total_questions += result.get("questions", 0)
            total_questions_filtered += result.get("questions_filtered", 0)

    # Stage names for progress display
    stage_names = {
        "storing": "Storing",
        "atomizing": "Atomizing",
        "generating": "Generating",
        "embedding": "Embedding",
        "filtering": "Filtering",
        "indexing": "Indexing",
    }

    def ingest_files_with_progress(progress: Progress) -> None:
        """Ingest files with progress bar display."""
        nonlocal failed_count
        file_count = len(files)
        files_task = progress.add_task(
            "", total=file_count, stage="Files", progress_text=f"0/{file_count}"
        )
        stage_task = progress.add_task("", total=100, stage="", visible=False, progress_text="")

        for i, filepath in enumerate(files):
            filename = os.path.basename(filepath)
            progress.update(
                files_task,
                description=filename,
                completed=i,
                progress_text=f"{i + 1}/{file_count}",
            )

            def on_progress(event: str, current: int, total: int, message: str) -> None:
                stage_name = stage_names.get(event, event.capitalize())
                if total > 1:
                    pct = int(100 * current / total)
                    progress.update(
                        stage_task,
                        visible=True,
                        stage=stage_name,
                        progress_text=f"{pct}%",
                        description=f"({current}/{total})",
                        total=total,
                        completed=current,
                    )
                else:
                    # Indeterminate progress - use spinner
                    progress.update(
                        stage_task,
                        visible=True,
                        stage=stage_name,
                        progress_text="",
                        description=message,
                        total=None,
                    )

            try:
                result = iso.ingest_file(filepath, on_progress=on_progress)
                process_result(result, filepath)
            except (OSError, ValueError, RuntimeError) as e:
                failed_count += 1
                err_msg = f"Warning: Failed to ingest {filepath} ({type(e).__name__}): {e}"
                console.print(f"[yellow]{err_msg}[/yellow]")

            progress.update(stage_task, visible=False)

        progress.update(files_task, completed=file_count, progress_text="Done", description="")

    def ingest_files_simple() -> None:
        """Ingest files with simple console output."""
        nonlocal failed_count
        for filepath in files:
            try:
                result = iso.ingest_file(filepath)
                process_result(result, filepath)

                if not plain:
                    if result.get("skipped"):
                        console.print(f"[dim]Skipped {filepath}: {result['reason']}[/dim]")
                    else:
                        console.print(f"[green]Ingested {filepath}[/green]")
            except (OSError, ValueError, RuntimeError) as e:
                failed_count += 1
                err_msg = f"Warning: Failed to ingest {filepath} ({type(e).__name__}): {e}"
                console.print(f"[yellow]{err_msg}[/yellow]")

    if show_progress:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.fields[stage]:>12}", justify="right"),
            BarColumn(bar_width=20),
            TextColumn("{task.fields[progress_text]}", style="cyan"),
            TextColumn("{task.description}", style="dim"),
            console=console,
        ) as progress:
            ingest_files_with_progress(progress)
    else:
        ingest_files_simple()

    # Summary
    if plain:
        console.print(f"Ingested {ingested_count} files ({total_chunks} chunks)")
        console.print(f"Created {total_atoms} atoms")
        console.print(f"Indexed {total_questions} questions")
        if total_questions_filtered > 0:
            console.print(f"Filtered {total_questions_filtered} similar questions")
        if skipped_count > 0:
            console.print(f"Skipped {skipped_count} unchanged files")
    else:
        console.print()
        console.print(f"[green]Ingested {ingested_count} files ({total_chunks} chunks)[/green]")
        console.print(f"[green]Created {total_atoms} atoms[/green]")
        console.print(f"[green]Indexed {total_questions} questions[/green]")
        if total_questions_filtered > 0:
            console.print(f"[dim]Filtered {total_questions_filtered} similar questions[/dim]")
        if skipped_count > 0:
            console.print(f"[dim]Skipped {skipped_count} unchanged files[/dim]")

    # Exit with error if all files failed
    if failed_count > 0 and ingested_count == 0:
        raise typer.Exit(1)


@app.command()
def query(
    question: str = typer.Argument(..., help="Question to ask"),
    data_dir: str = typer.Option(
        None,
        "--data-dir",
        "-d",
        help="Data directory (default: from settings)",
    ),
    config_file: str = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to config file",
    ),
    k: int = typer.Option(
        None,
        "--k",
        "-k",
        help="Number of results to return",
    ),
    raw: bool = typer.Option(
        False,
        "--raw",
        "-r",
        help="Return raw chunks without LLM synthesis",
    ),
    show_matched_questions: bool = typer.Option(
        False,
        "--show-matched-questions",
        "-q",
        help="Show which generated questions matched",
    ),
    plain: bool = typer.Option(
        False,
        "--plain",
        help="Plain output (no colors/formatting)",
    ),
) -> None:
    """Query the database with a question."""
    cli_config = load_config(Path(config_file) if config_file else None)
    effective_data_dir = data_dir or cli_config.get("data_dir") or DEFAULT_DATA_DIR

    # Check data dir exists
    if not os.path.exists(effective_data_dir):
        console.print(f"[red]Error: Data directory not found: {effective_data_dir}[/red]")
        console.print("[dim]Run 'isotope ingest' first to create the database.[/dim]")
        raise typer.Exit(1)

    # Create Isotope and retriever
    try:
        iso = get_isotope(data_dir, config_file)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from None

    # For retriever, build LLM client from config if available
    llm_client = None
    if not raw:
        llm_model = cli_config.get("llm_model") or os.environ.get("ISOTOPE_LITELLM_LLM_MODEL")
        if llm_model:
            from isotope.providers.litellm import LiteLLMClient

            llm_client = LiteLLMClient(model=llm_model)

    retriever = iso.retriever(
        default_k=k,
        llm_client=llm_client,
    )

    response = retriever.get_answer(question)

    if not response.results:
        if plain:
            console.print("No results found.")
        else:
            console.print("[yellow]No results found.[/yellow]")
        raise typer.Exit(0)

    if plain:
        # Plain output
        if response.answer:
            console.print(f"Answer: {response.answer}")
            console.print()

        console.print("Sources:")
        for i, result in enumerate(response.results, 1):
            console.print(f"  [{i}] {result.chunk.source} (score: {result.score:.3f})")
            console.print(f"      {result.chunk.content[:100]}...")
            if show_matched_questions:
                console.print(f"      Matched: {result.question.text}")
    else:
        # Rich output
        if response.answer:
            console.print(
                Panel(
                    Markdown(response.answer),
                    title="Answer",
                    border_style="green",
                )
            )
            console.print()

        console.print("[bold]Sources:[/bold]")
        for i, result in enumerate(response.results, 1):
            console.print(
                f"  [{i}] [cyan]{result.chunk.source}[/cyan] [dim](score: {result.score:.3f})[/dim]"
            )
            # Show first 100 chars of content
            preview = result.chunk.content[:100].replace("\n", " ")
            if len(result.chunk.content) > 100:
                preview += "..."
            console.print(f"      [dim]{preview}[/dim]")
            if show_matched_questions:
                console.print(f"      [yellow]Matched:[/yellow] {result.question.text}")


@app.command(name="list")
def list_sources(
    data_dir: str = typer.Option(
        None,
        "--data-dir",
        "-d",
        help="Data directory (default: from settings)",
    ),
    config_file: str = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to config file",
    ),
    plain: bool = typer.Option(
        False,
        "--plain",
        help="Plain output (no colors/formatting)",
    ),
) -> None:
    """List all indexed sources."""
    cli_config = load_config(Path(config_file) if config_file else None)
    effective_data_dir = data_dir or cli_config.get("data_dir") or DEFAULT_DATA_DIR

    if not os.path.exists(effective_data_dir):
        console.print("No database found. Run 'isotope ingest' first.")
        raise typer.Exit(0)

    try:
        stores = get_stores(effective_data_dir)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from None

    sources = stores["chunk_store"].list_sources()

    if not sources:
        if plain:
            console.print("No sources indexed.")
        else:
            console.print("[dim]No sources indexed.[/dim]")
        raise typer.Exit(0)

    if plain:
        console.print(f"Indexed sources ({len(sources)}):")
        for source in sorted(sources):
            chunks = stores["chunk_store"].get_by_source(source)
            console.print(f"  {source} ({len(chunks)} chunks)")
    else:
        table = Table(title=f"Indexed Sources ({len(sources)})")
        table.add_column("Source", style="cyan")
        table.add_column("Chunks", justify="right")

        for source in sorted(sources):
            chunks = stores["chunk_store"].get_by_source(source)
            table.add_row(source, str(len(chunks)))

        console.print(table)


@app.command()
def status(
    data_dir: str = typer.Option(
        None,
        "--data-dir",
        "-d",
        help="Data directory (default: from settings)",
    ),
    config_file: str = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to config file",
    ),
    detailed: bool = typer.Option(
        False,
        "--detailed",
        help="Show question distribution per source",
    ),
    plain: bool = typer.Option(
        False,
        "--plain",
        help="Plain output (no colors/formatting)",
    ),
) -> None:
    """Show database statistics."""
    cli_config = load_config(Path(config_file) if config_file else None)
    effective_data_dir = data_dir or cli_config.get("data_dir") or DEFAULT_DATA_DIR

    if not os.path.exists(effective_data_dir):
        if plain:
            console.print("No database found.")
        else:
            console.print("[dim]No database found. Run 'isotope ingest' first.[/dim]")
        raise typer.Exit(0)

    try:
        stores = get_stores(effective_data_dir)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from None

    sources = stores["chunk_store"].list_sources()
    total_chunks = stores["chunk_store"].count_chunks()
    total_atoms = stores["atom_store"].count_atoms()
    total_questions = stores["embedded_question_store"].count_questions()

    if plain:
        console.print("Database Status:")
        console.print(f"  Data directory: {effective_data_dir}")
        console.print(f"  Sources: {len(sources)}")
        console.print(f"  Chunks: {total_chunks}")
        console.print(f"  Atoms: {total_atoms}")
        console.print(f"  Indexed questions: {total_questions}")

        if detailed and sources:
            console.print()
            console.print("Question Distribution by Source:")
            for source in sorted(sources):
                chunk_ids = stores["chunk_store"].get_chunk_ids_by_source(source)
                num_chunks = len(chunk_ids)
                num_questions = stores["embedded_question_store"].count_by_chunk_ids(chunk_ids)
                console.print(f"  {source}: {num_chunks} chunks, {num_questions} questions")
    else:
        table = Table(title="Database Status")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")

        table.add_row("Data directory", effective_data_dir)
        table.add_row("Sources", str(len(sources)))
        table.add_row("Chunks", str(total_chunks))
        table.add_row("Atoms", str(total_atoms))
        table.add_row("Indexed questions", str(total_questions))

        console.print(table)

        if detailed and sources:
            console.print()
            detail_table = Table(title="Question Distribution by Source")
            detail_table.add_column("Source", style="cyan")
            detail_table.add_column("Chunks", justify="right")
            detail_table.add_column("Questions", justify="right", style="green")

            for source in sorted(sources):
                chunk_ids = stores["chunk_store"].get_chunk_ids_by_source(source)
                num_chunks = len(chunk_ids)
                num_questions = stores["embedded_question_store"].count_by_chunk_ids(chunk_ids)
                detail_table.add_row(source, str(num_chunks), str(num_questions))

            console.print(detail_table)


@app.command()
def delete(
    source: str = typer.Argument(..., help="Source path to delete"),
    data_dir: str = typer.Option(
        None,
        "--data-dir",
        "-d",
        help="Data directory (default: from settings)",
    ),
    config_file: str = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to config file",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Skip confirmation prompt",
    ),
    plain: bool = typer.Option(
        False,
        "--plain",
        help="Plain output (no colors/formatting)",
    ),
) -> None:
    """Delete a source and all its chunks from the database."""
    cli_config = load_config(Path(config_file) if config_file else None)
    effective_data_dir = data_dir or cli_config.get("data_dir") or DEFAULT_DATA_DIR

    if not os.path.exists(effective_data_dir):
        console.print("No database found.")
        raise typer.Exit(1)

    try:
        stores = get_stores(effective_data_dir)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from None

    # Find chunks for this source
    chunk_ids = stores["chunk_store"].get_chunk_ids_by_source(source)

    if not chunk_ids:
        if plain:
            console.print(f"Source not found: {source}")
        else:
            console.print(f"[yellow]Source not found: {source}[/yellow]")
        raise typer.Exit(1)

    # Confirm deletion
    if not force:
        console.print(f"[yellow]About to delete {len(chunk_ids)} chunks from {source}[/yellow]")
        confirm = typer.confirm("Continue?")
        if not confirm:
            console.print("Cancelled.")
            raise typer.Exit(0)

    # Delete from all stores (matching Isotope.delete_source() logic)
    stores["embedded_question_store"].delete_by_chunk_ids(chunk_ids)
    stores["atom_store"].delete_by_chunk_ids(chunk_ids)
    stores["chunk_store"].delete_by_source(source)
    stores["source_registry"].delete(source)

    if plain:
        console.print(f"Deleted {len(chunk_ids)} chunks from {source}")
    else:
        console.print(f"[green]Deleted {len(chunk_ids)} chunks from {source}[/green]")


def _is_local_model(model: str) -> bool:
    """Check if model is a local model (Ollama, etc.)."""
    from isotope.settings import LOCAL_MODEL_PREFIXES

    return model.lower().startswith(LOCAL_MODEL_PREFIXES)


def _get_settings_for_init(
    is_local: bool, rate_limited: bool | None, priority: str
) -> dict[str, Any]:
    """Get settings based on user's environment and priority.

    Returns a dict of settings that differ from defaults.
    """
    # Settings matrix from plan:
    # | Rate-limited? | Priority | sentence_atomizer | questions | llm_calls |
    # | Yes           | Speed    | true              | 3         | 2         |
    # | Yes           | Quality  | false             | 5         | 2         |
    # | Yes           | Balanced | false             | 5         | 2         |
    # | No            | Speed    | true              | 5         | 10        |
    # | No            | Quality  | false             | 10        | 10        |
    # | No            | Balanced | false             | 5         | 10        | (defaults)
    # | Local (auto)  | any      | true              | 5         | 1         |

    if is_local:
        # Local models: low concurrency, sentence atomizer for speed
        return {
            "use_sentence_atomizer": True,
            "questions_per_atom": 5,
            "max_concurrent_llm_calls": 1,
        }

    if rate_limited:
        # Rate-limited APIs
        if priority == "speed":
            return {
                "use_sentence_atomizer": True,
                "questions_per_atom": 3,
                "max_concurrent_llm_calls": 2,
            }
        elif priority == "quality":
            return {
                "use_sentence_atomizer": False,
                "questions_per_atom": 5,
                "max_concurrent_llm_calls": 2,
            }
        else:  # balanced
            return {
                "use_sentence_atomizer": False,
                "questions_per_atom": 5,
                "max_concurrent_llm_calls": 2,
            }
    else:
        # High-limit APIs
        if priority == "speed":
            return {
                "use_sentence_atomizer": True,
                "questions_per_atom": 5,
                "max_concurrent_llm_calls": 10,
            }
        elif priority == "quality":
            return {
                "use_sentence_atomizer": False,
                "questions_per_atom": 10,
                "max_concurrent_llm_calls": 10,
            }
        else:  # balanced - these are the defaults, return empty
            return {}


def _generate_config_content(
    llm_model: str,
    embedding_model: str,
    settings: dict[str, Any],
) -> str:
    """Generate isotope.yaml content with settings."""
    # Start with provider config
    content = f"""# Isotope Configuration
provider: litellm

# LiteLLM model identifiers
llm_model: {llm_model}
embedding_model: {embedding_model}
"""

    # Add settings section if we have non-default settings
    if settings:
        content += "\n# Settings (configured based on your environment)\nsettings:\n"
        for key, value in settings.items():
            if isinstance(value, bool):
                content += f"  {key}: {str(value).lower()}\n"
            else:
                content += f"  {key}: {value}\n"

    # Add commented defaults section
    content += """
# Uncomment to customize (showing defaults):
#   use_sentence_atomizer: false  # true = fast, false = LLM quality
#   questions_per_atom: 5         # more = better recall, higher cost
#   diversity_scope: global       # global | per_chunk | per_atom
#   max_concurrent_llm_calls: 10  # parallel LLM requests
#
# Advanced settings:
#   num_retries: 5
#   diversity_threshold: 0.85
#   default_k: 5
"""

    return content


@app.command()
def init(
    provider: str = typer.Option(
        "litellm",
        "--provider",
        "-p",
        help="Provider to use (litellm or custom)",
    ),
    llm_model: str = typer.Option(
        None,
        "--llm-model",
        help="LLM model (for litellm provider)",
    ),
    embedding_model: str = typer.Option(
        None,
        "--embedding-model",
        help="Embedding model (for litellm provider)",
    ),
) -> None:
    """Initialize a new isotope.yaml configuration file."""
    config_path = Path("isotope.yaml")

    if config_path.exists():
        console.print(f"[yellow]Config file already exists: {config_path}[/yellow]")
        if not typer.confirm("Overwrite?"):
            raise typer.Exit(0)

    if provider == "litellm":
        # Prompt for models if not provided
        effective_llm = llm_model
        effective_embedding = embedding_model

        if not effective_llm:
            effective_llm = typer.prompt(
                "Enter your LLM model",
                default="openai/gpt-5-mini-2025-08-07",
            )

        if not effective_embedding:
            effective_embedding = typer.prompt(
                "Enter your embedding model",
                default="openai/text-embedding-3-small",
            )

        # Detect if local model
        is_local = _is_local_model(effective_llm)

        rate_limited: bool | None = None
        priority = "balanced"

        if is_local:
            console.print(
                f"\n[dim]Detected local model ({effective_llm}). "
                "Configuring for single-GPU operation.[/dim]"
            )
        else:
            # Ask about rate limits
            console.print()
            rate_limit_choice = typer.prompt(
                "Are you on a rate-limited or free tier API?\n"
                "  [1] Yes - configure for rate limits\n"
                "  [2] No - I have high rate limits\n"
                "  [3] Not sure - use safe defaults\n"
                "Choose",
                default="3",
            )
            # Treat "Not sure" as rate-limited for safety
            is_high_rate_limit = rate_limit_choice == "2"
            rate_limited = not is_high_rate_limit

            # Ask about priority
            console.print()
            priority_choice = typer.prompt(
                "What's your priority?\n"
                "  [1] Retrieval quality (slower, more API calls)\n"
                "  [2] Speed & cost savings (faster, fewer calls)\n"
                "  [3] Balanced\n"
                "Choose",
                default="3",
            )
            if priority_choice == "1":
                priority = "quality"
            elif priority_choice == "2":
                priority = "speed"
            else:
                priority = "balanced"

        # Get settings based on choices
        settings = _get_settings_for_init(is_local, rate_limited, priority)

        # Generate config content
        content = _generate_config_content(effective_llm, effective_embedding, settings)

        config_path.write_text(content)
        console.print(f"\n[green]Created {config_path}[/green]")

        # Collect API keys from user
        if not is_local:
            llm_key, embed_key = _collect_api_keys()
            if llm_key or embed_key:
                _save_api_keys_and_update_gitignore(llm_key, embed_key)

        console.print()
        console.print("[bold]Ready![/bold] Try these commands:")
        console.print("  isotope ingest ./docs")
        console.print("  isotope query 'your question'")

    else:
        # Custom provider
        content = """# Isotope Configuration
provider: custom

# Custom implementation classes (dotted import paths)
embedder: my_package.MyEmbedder
question_generator: my_package.MyGenerator
atomizer: my_package.MyAtomizer

# Optional: kwargs passed to each class
# embedder_kwargs:
#   region: us-east-1
# question_generator_kwargs: {}
# atomizer_kwargs: {}

# Optional: Data directory
# data_dir: ./isotope_data
"""
        config_path.write_text(content)
        console.print(f"[green]Created {config_path}[/green]")
        console.print()
        console.print("Next steps:")
        console.print("  1. Implement your custom classes")
        console.print("  2. Update the class paths in isotope.yaml")
        console.print("  3. Ingest documents: isotope ingest ./docs")


@questions_app.command(name="sample")
def sample_questions(
    source: str = typer.Option(
        None,
        "--source",
        "-s",
        help="Filter by source file",
    ),
    n: int = typer.Option(
        5,
        "-n",
        help="Number of questions to sample",
    ),
    data_dir: str = typer.Option(
        None,
        "--data-dir",
        "-d",
        help="Data directory (default: from settings)",
    ),
    config_file: str = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to config file",
    ),
    plain: bool = typer.Option(
        False,
        "--plain",
        help="Plain output (no colors/formatting)",
    ),
) -> None:
    """Show a random sample of generated questions."""
    cli_config = load_config(Path(config_file) if config_file else None)
    effective_data_dir = data_dir or cli_config.get("data_dir") or DEFAULT_DATA_DIR

    if not os.path.exists(effective_data_dir):
        if plain:
            console.print("No database found. Run 'isotope ingest' first.")
        else:
            console.print("[dim]No database found. Run 'isotope ingest' first.[/dim]")
        raise typer.Exit(0)

    try:
        stores = get_stores(effective_data_dir)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from None

    # Get chunk_ids for source filtering if specified
    chunk_ids: list[str] | None = None
    if source:
        chunk_ids = stores["chunk_store"].get_chunk_ids_by_source(source)
        if not chunk_ids:
            if plain:
                console.print(f"No questions found for source: {source}")
            else:
                console.print(f"[yellow]No questions found for source: {source}[/yellow]")
            raise typer.Exit(0)

    questions = stores["embedded_question_store"].sample(n=n, chunk_ids=chunk_ids)

    if not questions:
        if plain:
            console.print("No questions indexed. Run 'isotope ingest' first.")
        else:
            console.print("[dim]No questions indexed. Run 'isotope ingest' first.[/dim]")
        raise typer.Exit(0)

    total = stores["embedded_question_store"].count_questions()
    if chunk_ids:
        total = stores["embedded_question_store"].count_by_chunk_ids(chunk_ids)

    if plain:
        console.print(f"Sample Questions ({len(questions)} of {total}):")
        for i, q in enumerate(questions, 1):
            console.print(f"  {i}. {q.text}")
    else:
        title = f"Sample Questions ({len(questions)} of {total})"
        if source:
            title += f" from {source}"

        table = Table(title=title)
        table.add_column("#", style="dim", width=3)
        table.add_column("Question", style="cyan")

        for i, q in enumerate(questions, 1):
            table.add_row(str(i), q.text)

        console.print(table)

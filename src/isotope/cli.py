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
    pass


DEFAULT_DATA_DIR = "./isotope_data"
CONFIG_FILES = ["isotope.yaml", "isotope.yml", ".isotoperc"]


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


def _get_behavioral_settings_from_env() -> dict:
    """Read behavioral settings from ISOTOPE_* environment variables.

    This is the CLI (application layer) reading env vars and preparing
    them to pass explicitly to the library.
    """
    return {
        "questions_per_atom": int(os.environ.get("ISOTOPE_QUESTIONS_PER_ATOM", "15")),
        "question_generator_prompt": os.environ.get("ISOTOPE_QUESTION_GENERATOR_PROMPT") or None,
        "atomizer_prompt": os.environ.get("ISOTOPE_ATOMIZER_PROMPT") or None,
        "diversity_threshold": _parse_threshold(
            os.environ.get("ISOTOPE_QUESTION_DIVERSITY_THRESHOLD", "0.85")
        ),
        "diversity_scope": _parse_diversity_scope(os.environ.get("ISOTOPE_DIVERSITY_SCOPE")),
        "default_k": int(os.environ.get("ISOTOPE_DEFAULT_K", "5")),
        "synthesis_prompt": os.environ.get("ISOTOPE_SYNTHESIS_PROMPT") or None,
        "max_concurrent_questions": int(os.environ.get("ISOTOPE_MAX_CONCURRENT_QUESTIONS", "10")),
    }


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

    Configuration is loaded from:
    1. Explicit config_path if provided
    2. isotope.yaml in current/parent directories
    3. Falls back to LiteLLM with env vars if no config found
    """
    from dataclasses import dataclass

    from isotope.configuration import LiteLLMProvider, LocalStorage
    from isotope.isotope import Isotope
    from isotope.settings import Settings

    config = load_config(Path(config_path) if config_path else None)
    effective_data_dir = data_dir or config.get("data_dir") or DEFAULT_DATA_DIR

    provider = config.get("provider", "litellm")

    # Get behavioral settings from env vars (CLI reads env, passes to library)
    env_settings = _get_behavioral_settings_from_env()

    # Build Settings from env vars
    settings = Settings(
        questions_per_atom=env_settings["questions_per_atom"],
        question_generator_prompt=env_settings["question_generator_prompt"],
        atomizer_prompt=env_settings["atomizer_prompt"],
        question_diversity_threshold=env_settings["diversity_threshold"],
        diversity_scope=env_settings["diversity_scope"],
        default_k=env_settings["default_k"],
        synthesis_prompt=env_settings["synthesis_prompt"],
        max_concurrent_questions=env_settings["max_concurrent_questions"],
    )

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
            console.print("  llm_model: openai/gpt-4o")
            console.print("  embedding_model: openai/text-embedding-3-small")
            console.print()
            console.print("Or set environment variables:")
            console.print("  export ISOTOPE_LITELLM_LLM_MODEL=openai/gpt-4o")
            console.print("  export ISOTOPE_LITELLM_EMBEDDING_MODEL=openai/text-embedding-3-small")
            raise typer.Exit(1)

        use_sentence_atomizer = config.get("use_sentence_atomizer", False)

        return Isotope(
            provider=LiteLLMProvider(
                llm=llm_model,
                embedding=embedding_model,
                atomizer_type="sentence" if use_sentence_atomizer else "llm",
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

            def build_embedder(self) -> Any:
                return self._embedder

            def build_atomizer(self, settings: Settings) -> Any:
                return self._atomizer

            def build_question_generator(self, settings: Settings) -> Any:
                return self._question_generator

            def build_llm_client(self) -> Any:
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
    # Read behavioral settings from env vars (application layer)
    env_settings = _get_behavioral_settings_from_env()
    cli_config = load_config(Path(config_file) if config_file else None)

    table = Table(title="Isotope Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Source", style="dim")

    # Provider config (from config file or env)
    provider = cli_config.get("provider", "litellm")
    table.add_row("provider", provider, "config file" if cli_config else "default")

    if provider == "litellm":
        llm_model = cli_config.get("llm_model") or os.environ.get("ISOTOPE_LITELLM_LLM_MODEL")
        embedding_model = cli_config.get("embedding_model") or os.environ.get(
            "ISOTOPE_LITELLM_EMBEDDING_MODEL"
        )
        table.add_row(
            "llm_model",
            llm_model or "(not set)",
            "config file" if cli_config.get("llm_model") else "env var",
        )
        table.add_row(
            "embedding_model",
            embedding_model or "(not set)",
            "config file" if cli_config.get("embedding_model") else "env var",
        )
    elif provider == "custom":
        table.add_row("embedder", cli_config.get("embedder", "(not set)"), "config file")
        table.add_row(
            "question_generator", cli_config.get("question_generator", "(not set)"), "config file"
        )
        table.add_row("atomizer", cli_config.get("atomizer", "(not set)"), "config file")

    # Behavioral settings (from env vars, read by CLI)
    table.add_row("", "", "")  # Separator
    table.add_row("questions_per_atom", str(env_settings["questions_per_atom"]), "env var")
    threshold = env_settings["diversity_threshold"]
    table.add_row(
        "question_diversity_threshold",
        str(threshold) if threshold is not None else "disabled",
        "env var",
    )
    table.add_row("diversity_scope", env_settings["diversity_scope"], "env var")
    table.add_row("default_k", str(env_settings["default_k"]), "env var")
    table.add_row(
        "max_concurrent_questions",
        str(env_settings["max_concurrent_questions"]),
        "env var",
    )

    console.print(table)

    # Show config file location
    config_path = find_config_file()
    if config_path:
        console.print(f"\n[dim]Config file: {config_path}[/dim]")
    else:
        console.print("\n[dim]No config file found. Using env vars / defaults.[/dim]")


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

    for filepath in files:
        try:
            result = iso.ingest_file(filepath)

            if result.get("skipped"):
                skipped_count += 1
                if not plain:
                    console.print(f"[dim]Skipped {filepath}: {result['reason']}[/dim]")
            else:
                ingested_count += 1
                total_chunks += result.get("chunks", 0)
                total_atoms += result.get("atoms", 0)
                total_questions += result.get("questions", 0)
                total_questions_filtered += result.get("questions_filtered", 0)
                if not plain:
                    console.print(f"[green]Ingested {filepath}[/green]")
        except (OSError, ValueError, RuntimeError) as e:
            console.print(
                f"[yellow]Warning: Failed to ingest {filepath} ({type(e).__name__}): {e}[/yellow]"
            )

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
        content = f"""# Isotope Configuration
provider: litellm

# LiteLLM model identifiers
llm_model: {llm_model or "openai/gpt-4o"}
embedding_model: {embedding_model or "openai/text-embedding-3-small"}

# Optional: Use sentence-based atomizer instead of LLM
# use_sentence_atomizer: false

# Optional: Data directory
# data_dir: ./isotope_data
"""
    else:
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
    if provider == "litellm":
        console.print("  1. Set your API key: export OPENAI_API_KEY=...")
        console.print("  2. Ingest documents: isotope ingest ./docs")
        console.print("  3. Query: isotope query 'your question'")
    else:
        console.print("  1. Implement your custom classes")
        console.print("  2. Update the class paths in isotope.yaml")
        console.print("  3. Ingest documents: isotope ingest ./docs")

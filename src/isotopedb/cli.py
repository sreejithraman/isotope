# src/isotopedb/cli.py
"""Command-line interface for IsotopeDB.

The CLI uses a configuration file (isotope.yaml) to determine which
provider to use. See docs for configuration options.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict, cast

if TYPE_CHECKING:
    from isotopedb.isotope import Isotope
    from isotopedb.stores import ChromaVectorStore, SQLiteAtomStore, SQLiteDocStore

try:
    import typer
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table
except ImportError as e:
    raise SystemExit(
        "CLI requires additional dependencies.\nInstall with: pip install isotopedb[cli]"
    ) from e

try:
    import yaml  # type: ignore[import-untyped]

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from isotopedb import __version__
from isotopedb.config import Settings

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


class StoreBundle(TypedDict):
    vector_store: ChromaVectorStore
    doc_store: SQLiteDocStore
    atom_store: SQLiteAtomStore


def get_stores(data_dir: str) -> StoreBundle:
    """Get store instances for read-only operations (list, status, delete).

    This doesn't require provider configuration since it only accesses stores.
    """
    from isotopedb.stores import ChromaVectorStore, SQLiteAtomStore, SQLiteDocStore

    return {
        "vector_store": ChromaVectorStore(os.path.join(data_dir, "chroma")),
        "doc_store": SQLiteDocStore(os.path.join(data_dir, "docs.db")),
        "atom_store": SQLiteAtomStore(os.path.join(data_dir, "atoms.db")),
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
    from isotopedb.isotope import Isotope

    config = load_config(Path(config_path) if config_path else None)
    effective_data_dir = data_dir or config.get("data_dir") or DEFAULT_DATA_DIR

    provider = config.get("provider", "litellm")

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

        return Isotope.with_litellm(
            llm_model=llm_model,
            embedding_model=embedding_model,
            data_dir=effective_data_dir,
            use_sentence_atomizer=use_sentence_atomizer,
        )

    elif provider == "custom":
        # Custom provider - import classes dynamically
        embedder_class = config.get("embedder")
        generator_class = config.get("generator")
        atomizer_class = config.get("atomizer")

        if not all([embedder_class, generator_class, atomizer_class]):
            console.print(
                "[red]Error: Custom provider requires embedder, generator, and atomizer.[/red]"
            )
            console.print()
            console.print("Example isotope.yaml:")
            console.print("  provider: custom")
            console.print("  embedder: my_package.MyEmbedder")
            console.print("  generator: my_package.MyGenerator")
            console.print("  atomizer: my_package.MyAtomizer")
            raise typer.Exit(1)

        if (
            not isinstance(embedder_class, str)
            or not isinstance(generator_class, str)
            or not isinstance(atomizer_class, str)
        ):
            console.print("[red]Error: Custom provider class paths must be strings.[/red]")
            raise typer.Exit(1)

        try:
            embedder_cls = import_class(embedder_class)
            generator_cls = import_class(generator_class)
            atomizer_cls = import_class(atomizer_class)
        except (ImportError, AttributeError) as e:
            console.print(f"[red]Error importing custom class: {e}[/red]")
            raise typer.Exit(1) from None

        # Get kwargs for each class
        embedder_kwargs = config.get("embedder_kwargs", {})
        generator_kwargs = config.get("generator_kwargs", {})
        atomizer_kwargs = config.get("atomizer_kwargs", {})

        embedder = embedder_cls(**embedder_kwargs)
        generator = generator_cls(**generator_kwargs)
        atomizer = atomizer_cls(**atomizer_kwargs)

        return Isotope.with_local_stores(
            embedder=embedder,
            atomizer=atomizer,
            generator=generator,
            data_dir=effective_data_dir,
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
    settings = Settings()
    cli_config = load_config(Path(config_file) if config_file else None)

    table = Table(title="IsotopeDB Configuration")
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
        table.add_row("generator", cli_config.get("generator", "(not set)"), "config file")
        table.add_row("atomizer", cli_config.get("atomizer", "(not set)"), "config file")

    # Behavioral settings (from env vars)
    table.add_row("", "", "")  # Separator
    table.add_row("questions_per_atom", str(settings.questions_per_atom), "env var")
    threshold = settings.question_diversity_threshold
    table.add_row(
        "question_diversity_threshold",
        str(threshold) if threshold is not None else "disabled",
        "env var",
    )
    table.add_row("diversity_scope", settings.diversity_scope, "env var")
    table.add_row("dedup_strategy", settings.dedup_strategy, "env var")
    table.add_row("default_k", str(settings.default_k), "env var")

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
    from isotopedb.loaders import LoaderRegistry

    # Check path exists
    if not os.path.exists(path):
        console.print(f"[red]Error: Path not found: {path}[/red]")
        raise typer.Exit(1)

    # Create Isotope and ingestor
    try:
        iso = get_isotope(data_dir, config_file)
        ingestor = iso.ingestor()
    except Exception as e:
        console.print(f"[red]Error creating ingestor: {e}[/red]")
        raise typer.Exit(1) from None

    # Load files
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

    # Load chunks
    all_chunks = []
    for filepath in files:
        try:
            chunks = registry.load(filepath)
            all_chunks.extend(chunks)
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to load {filepath}: {e}[/yellow]")

    if not all_chunks:
        console.print("[yellow]No content to ingest.[/yellow]")
        raise typer.Exit(0)

    # Ingest with progress
    if plain:
        result = ingestor.ingest_chunks(all_chunks)
        console.print(f"Ingested {result['chunks']} chunks")
        console.print(f"Created {result['atoms']} atoms")
        console.print(f"Generated {result['questions']} questions")
        if result.get("chunks_removed", 0) > 0:
            console.print(f"Removed {result['chunks_removed']} old chunks")
    else:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Ingesting...", total=None)

            def on_progress(event: str, current: int, total: int, message: str) -> None:
                progress.update(task, description=f"{event}: {message}")

            result = ingestor.ingest_chunks(all_chunks, on_progress=on_progress)

        console.print()
        console.print(f"[green]Ingested {result['chunks']} chunks[/green]")
        console.print(f"[green]Created {result['atoms']} atoms[/green]")
        console.print(f"[green]Generated {result['questions']} questions[/green]")
        if result.get("chunks_removed", 0) > 0:
            console.print(f"[yellow]Removed {result['chunks_removed']} old chunks[/yellow]")
        if result.get("questions_filtered", 0) > 0:
            console.print(f"[dim]Filtered {result['questions_filtered']} similar questions[/dim]")


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

    # For retriever, get LLM model from config if available
    llm_model: str | None = None
    if not raw:
        llm_model = cli_config.get("llm_model") or os.environ.get("ISOTOPE_LITELLM_LLM_MODEL")

    retriever = iso.retriever(
        default_k=k,
        llm_model=llm_model,
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

    sources = stores["doc_store"].list_sources()

    if not sources:
        if plain:
            console.print("No sources indexed.")
        else:
            console.print("[dim]No sources indexed.[/dim]")
        raise typer.Exit(0)

    if plain:
        console.print(f"Indexed sources ({len(sources)}):")
        for source in sorted(sources):
            chunks = stores["doc_store"].get_by_source(source)
            console.print(f"  {source} ({len(chunks)} chunks)")
    else:
        table = Table(title=f"Indexed Sources ({len(sources)})")
        table.add_column("Source", style="cyan")
        table.add_column("Chunks", justify="right")

        for source in sorted(sources):
            chunks = stores["doc_store"].get_by_source(source)
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

    sources = stores["doc_store"].list_sources()
    total_chunks = stores["doc_store"].count_chunks()
    total_atoms = stores["atom_store"].count_atoms()
    total_questions = stores["vector_store"].count_questions()

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
    chunks = stores["doc_store"].get_by_source(source)

    if not chunks:
        if plain:
            console.print(f"Source not found: {source}")
        else:
            console.print(f"[yellow]Source not found: {source}[/yellow]")
        raise typer.Exit(1)

    # Confirm deletion
    if not force and not plain:
        console.print(f"[yellow]About to delete {len(chunks)} chunks from {source}[/yellow]")
        confirm = typer.confirm("Continue?")
        if not confirm:
            console.print("Cancelled.")
            raise typer.Exit(0)

    # Delete from all stores
    chunk_ids = [c.id for c in chunks]

    stores["vector_store"].delete_by_chunk_ids(chunk_ids)
    stores["atom_store"].delete_by_chunk_ids(chunk_ids)
    stores["doc_store"].delete_many(chunk_ids)

    if plain:
        console.print(f"Deleted {len(chunks)} chunks from {source}")
    else:
        console.print(f"[green]Deleted {len(chunks)} chunks from {source}[/green]")


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
        content = f"""# IsotopeDB Configuration
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
        content = """# IsotopeDB Configuration
provider: custom

# Custom implementation classes (dotted import paths)
embedder: my_package.MyEmbedder
generator: my_package.MyGenerator
atomizer: my_package.MyAtomizer

# Optional: kwargs passed to each class
# embedder_kwargs:
#   region: us-east-1
# generator_kwargs: {}
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

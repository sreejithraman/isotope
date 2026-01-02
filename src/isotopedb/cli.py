# src/isotopedb/cli.py
"""Command-line interface for Isotope."""

import os

try:
    import typer
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table
except ImportError as e:
    raise SystemExit(
        "CLI requires additional dependencies.\n"
        "Install with: pip install isotopedb[cli]"
    ) from e

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


def get_isotope(data_dir: str | None = None):
    """Create an Isotope instance for the given data directory."""
    from isotopedb.isotope import Isotope

    return Isotope(data_dir=data_dir)


@app.command()
def config() -> None:
    """Show current configuration settings."""
    settings = Settings()

    table = Table(title="Isotope Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("llm_model", settings.llm_model)
    table.add_row("embedding_model", settings.embedding_model)
    table.add_row("atomizer", settings.atomizer)
    table.add_row("questions_per_atom", str(settings.questions_per_atom))
    table.add_row(
        "question_diversity_threshold",
        str(settings.question_diversity_threshold) if settings.question_diversity_threshold is not None else "disabled",
    )
    table.add_row("data_dir", settings.data_dir)
    table.add_row("vector_store", settings.vector_store)
    table.add_row("doc_store", settings.doc_store)
    table.add_row("dedup_strategy", settings.dedup_strategy)
    table.add_row("default_k", str(settings.default_k))

    console.print(table)


@app.command()
def ingest(
    path: str = typer.Argument(..., help="File or directory to ingest"),
    data_dir: str = typer.Option(
        None,
        "--data-dir",
        "-d",
        help="Data directory (default: from settings)",
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
    iso = get_isotope(data_dir)
    ingestor = iso.ingestor()

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

            def on_progress(event: str, current: int, total: int, message: str):
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
    settings = Settings()
    effective_data_dir = data_dir or settings.data_dir

    # Check data dir exists
    if not os.path.exists(effective_data_dir):
        console.print(f"[red]Error: Data directory not found: {effective_data_dir}[/red]")
        console.print("[dim]Run 'isotope ingest' first to create the database.[/dim]")
        raise typer.Exit(1)

    # Create Isotope and retriever
    iso = get_isotope(data_dir)
    retriever = iso.retriever(
        default_k=k,
        llm_model="" if raw else None,  # "" disables LLM synthesis
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
            console.print(Panel(
                Markdown(response.answer),
                title="Answer",
                border_style="green",
            ))
            console.print()

        console.print("[bold]Sources:[/bold]")
        for i, result in enumerate(response.results, 1):
            console.print(
                f"  [{i}] [cyan]{result.chunk.source}[/cyan] "
                f"[dim](score: {result.score:.3f})[/dim]"
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
    plain: bool = typer.Option(
        False,
        "--plain",
        help="Plain output (no colors/formatting)",
    ),
) -> None:
    """List all indexed sources."""
    settings = Settings()
    effective_data_dir = data_dir or settings.data_dir

    if not os.path.exists(effective_data_dir):
        console.print("No database found. Run 'isotope ingest' first.")
        raise typer.Exit(0)

    iso = get_isotope(data_dir)
    sources = iso.doc_store.list_sources()

    if not sources:
        if plain:
            console.print("No sources indexed.")
        else:
            console.print("[dim]No sources indexed.[/dim]")
        raise typer.Exit(0)

    if plain:
        console.print(f"Indexed sources ({len(sources)}):")
        for source in sorted(sources):
            chunks = iso.doc_store.get_by_source(source)
            console.print(f"  {source} ({len(chunks)} chunks)")
    else:
        table = Table(title=f"Indexed Sources ({len(sources)})")
        table.add_column("Source", style="cyan")
        table.add_column("Chunks", justify="right")

        for source in sorted(sources):
            chunks = iso.doc_store.get_by_source(source)
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
    plain: bool = typer.Option(
        False,
        "--plain",
        help="Plain output (no colors/formatting)",
    ),
) -> None:
    """Show database statistics."""
    settings = Settings()
    effective_data_dir = data_dir or settings.data_dir

    if not os.path.exists(effective_data_dir):
        if plain:
            console.print("No database found.")
        else:
            console.print("[dim]No database found. Run 'isotope ingest' first.[/dim]")
        raise typer.Exit(0)

    iso = get_isotope(data_dir)

    sources = iso.doc_store.list_sources()
    chunk_ids = iso.atom_store.list_chunk_ids()
    question_chunk_ids = iso.vector_store.list_chunk_ids()

    # Count totals
    total_chunks = 0
    for source in sources:
        chunks = iso.doc_store.get_by_source(source)
        total_chunks += len(chunks)

    total_atoms = 0
    for chunk_id in chunk_ids:
        atoms = iso.atom_store.get_by_chunk(chunk_id)
        total_atoms += len(atoms)

    if plain:
        console.print("Database Status:")
        console.print(f"  Data directory: {effective_data_dir}")
        console.print(f"  Sources: {len(sources)}")
        console.print(f"  Chunks: {total_chunks}")
        console.print(f"  Atoms: {total_atoms}")
        console.print(f"  Questions indexed: {len(question_chunk_ids)} chunks")
    else:
        table = Table(title="Database Status")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")

        table.add_row("Data directory", effective_data_dir)
        table.add_row("Sources", str(len(sources)))
        table.add_row("Chunks", str(total_chunks))
        table.add_row("Atoms", str(total_atoms))
        table.add_row("Indexed questions (by chunk)", str(len(question_chunk_ids)))

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
    settings = Settings()
    effective_data_dir = data_dir or settings.data_dir

    if not os.path.exists(effective_data_dir):
        console.print("No database found.")
        raise typer.Exit(1)

    iso = get_isotope(data_dir)

    # Find chunks for this source
    chunks = iso.doc_store.get_by_source(source)

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

    iso.vector_store.delete_by_chunk_ids(chunk_ids)
    iso.atom_store.delete_by_chunk_ids(chunk_ids)
    for chunk_id in chunk_ids:
        iso.doc_store.delete(chunk_id)

    if plain:
        console.print(f"Deleted {len(chunks)} chunks from {source}")
    else:
        console.print(f"[green]Deleted {len(chunks)} chunks from {source}[/green]")

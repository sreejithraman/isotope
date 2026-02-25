# src/isotope/cli/app.py
"""Command-line interface for Isotope.

This module provides a thin Typer wrapper around the commands layer.
Each command:
1. Parses args (via Typer)
2. Creates progress callbacks (for Rich display)
3. Calls commands module functions
4. Renders results with Rich
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Iterator
from contextlib import contextmanager

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

from isotope import __version__
from isotope.commands import (
    CommandStage,
    ProgressUpdate,
    config_cmd,
    delete,
    ingest,
    init,
    inspect,
    query,
    questions,
)
from isotope.commands.base import ConfirmRequest, FileIngestResult, IngestResult, PromptRequest
from isotope.commands.init import InitCancelled
from isotope.config import load_env_file

app = typer.Typer(
    name="isotope",
    help="Isotope - Reverse RAG database. Index questions, not chunks.",
    no_args_is_help=True,
)
console = Console()

# Global verbose flag - set by main callback
_verbose = False


def _print_error(error: str | None, error_details: str | None = None, plain: bool = False) -> None:
    """Print an error message, showing details if verbose mode is enabled."""
    if _verbose and error_details:
        if plain:
            console.print(f"Error: {error}")
            console.print("\nDetails:")
            console.print(error_details)
        else:
            console.print(f"[red]Error: {error}[/red]")
            console.print()
            console.print("[dim]Details:[/dim]")
            console.print(f"[dim]{error_details}[/dim]")
    else:
        if plain:
            console.print(f"Error: {error}")
        else:
            console.print(f"[red]Error: {error}[/red]")


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
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Show detailed error messages for debugging.",
    ),
) -> None:
    """Isotope - Reverse RAG database."""
    global _verbose
    _verbose = verbose
    load_env_file()


# Stage names for progress display
STAGE_NAMES = {
    CommandStage.STORING: "Storing",
    CommandStage.ATOMIZING: "Atomizing",
    CommandStage.GENERATING: "Generating",
    CommandStage.EMBEDDING: "Embedding",
    CommandStage.FILTERING: "Filtering",
    CommandStage.INDEXING: "Indexing",
    CommandStage.LOADING: "Loading",
    CommandStage.PROCESSING: "Processing",
    CommandStage.COMPLETE: "Complete",
}


@app.command(name="ingest", rich_help_panel="Core")
def ingest_cmd(
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
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Re-ingest even if content hasn't changed",
    ),
) -> None:
    """Ingest a file or directory into the database."""
    # Determine if we should show progress bars
    show_progress = not plain and not no_progress and console.is_terminal

    if show_progress:
        _ingest_with_progress(path, data_dir, config_file, force=force)
    else:
        _ingest_simple(path, data_dir, config_file, plain, force=force)


def _ingest_with_progress(
    path: str,
    data_dir: str | None,
    config_file: str | None,
    force: bool = False,
) -> None:
    """Ingest with Rich progress bars."""
    with _file_logging_context(data_dir, config_file):
        _ingest_with_progress_inner(path, data_dir, config_file, force)


@contextmanager
def _file_logging_context(data_dir: str | None, config_file: str | None) -> Iterator[None]:
    """Set up file logging for CLI commands that need persistent logs."""
    from isotope.config import (
        DEFAULT_DATA_DIR,
        load_config,
        setup_file_logging,
        teardown_file_logging,
    )

    cfg = load_config(config_file)
    effective_data_dir = data_dir or cfg.get("data_dir") or DEFAULT_DATA_DIR
    handler, prev_level = setup_file_logging(effective_data_dir)
    try:
        yield
    finally:
        teardown_file_logging(handler, prev_level)


def _ingest_with_progress_inner(
    path: str,
    data_dir: str | None,
    config_file: str | None,
    force: bool = False,
) -> None:
    """Ingest with Rich progress bars (inner, after logging is wired)."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.fields[stage]:>12}", justify="right"),
        BarColumn(bar_width=20),
        TextColumn("{task.fields[progress_text]}", style="cyan"),
        TextColumn("{task.description}", style="dim"),
        console=console,
    ) as progress:
        files_task = progress.add_task("", total=None, stage="Files", progress_text="")
        stage_task = progress.add_task("", total=100, stage="", visible=False, progress_text="")

        current_file_index = 0
        total_files = 0

        def on_file_start(filepath: str, index: int, total: int) -> None:
            nonlocal current_file_index, total_files
            current_file_index = index
            total_files = total
            filename = os.path.basename(filepath)
            progress.update(
                files_task,
                description=filename,
                completed=index,
                total=total,
                progress_text=f"{index + 1}/{total}",
            )

        def on_progress(update: ProgressUpdate) -> None:
            stage_name = STAGE_NAMES.get(update.stage, update.stage.value)
            if update.total > 1:
                pct = update.percentage
                progress.update(
                    stage_task,
                    visible=True,
                    stage=stage_name,
                    progress_text=f"{pct}%",
                    description=f"({update.current}/{update.total})",
                    total=update.total,
                    completed=update.current,
                )
            else:
                progress.update(
                    stage_task,
                    visible=True,
                    stage=stage_name,
                    progress_text="",
                    description=update.message or "",
                    total=None,
                )

        def on_file_complete(result: FileIngestResult) -> None:
            progress.update(stage_task, visible=False)

        result = asyncio.run(
            ingest.aingest(
                path=path,
                data_dir=data_dir,
                config_path=config_file,
                on_progress=on_progress,
                on_file_start=on_file_start,
                on_file_complete=on_file_complete,
                force=force,
            )
        )

        progress.update(
            files_task,
            completed=total_files,
            progress_text="Done",
            description="",
        )

    _render_ingest_result(result, plain=False)


def _ingest_simple(
    path: str,
    data_dir: str | None,
    config_file: str | None,
    plain: bool,
    force: bool = False,
) -> None:
    """Ingest with simple console output."""
    with _file_logging_context(data_dir, config_file):
        _ingest_simple_inner(path, data_dir, config_file, plain, force)


def _ingest_simple_inner(
    path: str,
    data_dir: str | None,
    config_file: str | None,
    plain: bool,
    force: bool = False,
) -> None:
    """Ingest with simple console output (inner, after logging is wired)."""

    def on_file_complete(file_result: FileIngestResult) -> None:
        if not plain:
            if file_result.failed:
                console.print(f"[red]Failed {file_result.filepath}: {file_result.reason}[/red]")
            elif file_result.skipped:
                console.print(f"[dim]Skipped {file_result.filepath}: {file_result.reason}[/dim]")
            else:
                console.print(f"[green]Ingested {file_result.filepath}[/green]")

    result = asyncio.run(
        ingest.aingest(
            path=path,
            data_dir=data_dir,
            config_path=config_file,
            on_file_complete=on_file_complete,
            force=force,
        )
    )

    _render_ingest_result(result, plain=plain)


def _render_ingest_result(result: IngestResult, plain: bool) -> None:
    """Render ingest result to console."""
    if not result.success:
        _print_error(result.error, result.error_details, plain=plain)
        raise typer.Exit(1)

    # Show warning if error is set but success=True (e.g., "No supported files found")
    if result.error:
        if plain:
            console.print(f"Warning: {result.error}")
        else:
            console.print(f"[yellow]Warning: {result.error}[/yellow]")

    if plain:
        console.print(f"Ingested {result.files_processed} files ({result.total_chunks} chunks)")
        console.print(f"Created {result.total_atoms} atoms")
        console.print(f"Indexed {result.total_questions} questions")
        if result.total_questions_filtered > 0:
            console.print(f"Filtered {result.total_questions_filtered} similar questions")
        if result.files_skipped > 0:
            console.print(f"Skipped {result.files_skipped} unchanged files")
    else:
        console.print()
        console.print(
            f"[green]Ingested {result.files_processed} files ({result.total_chunks} chunks)[/green]"
        )
        console.print(f"[green]Created {result.total_atoms} atoms[/green]")
        console.print(f"[green]Indexed {result.total_questions} questions[/green]")
        if result.total_questions_filtered > 0:
            console.print(
                f"[dim]Filtered {result.total_questions_filtered} similar questions[/dim]"
            )
        if result.files_skipped > 0:
            console.print(f"[dim]Skipped {result.files_skipped} unchanged files[/dim]")

    if result.files_failed > 0 and result.files_processed == 0:
        raise typer.Exit(1)


@app.command(name="query", rich_help_panel="Core")
def query_cmd(
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
    result = query.query(
        question=question,
        data_dir=data_dir,
        config_path=config_file,
        k=k,
        raw=raw,
        show_matched_questions=show_matched_questions,
    )

    if not result.success:
        _print_error(result.error, result.error_details, plain=plain)
        raise typer.Exit(1)

    if not result.results:
        if plain:
            console.print("No results found.")
        else:
            console.print("[yellow]No results found.[/yellow]")
        raise typer.Exit(0)

    if plain:
        if result.answer:
            console.print(f"Answer: {result.answer}")
            console.print()

        console.print("Sources:")
        for i, r in enumerate(result.results, 1):
            console.print(f"  [{i}] {r.source} (score: {r.score:.3f})")
            console.print(f"      {r.content[:100]}...")
            if r.matched_question:
                console.print(f"      Matched: {r.matched_question}")
    else:
        if result.answer:
            console.print(
                Panel(
                    Markdown(result.answer),
                    title="Answer",
                    border_style="green",
                )
            )
            console.print()

        console.print("[bold]Sources:[/bold]")
        for i, r in enumerate(result.results, 1):
            console.print(f"  [{i}] [cyan]{r.source}[/cyan] [dim](score: {r.score:.3f})[/dim]")
            preview = r.content[:100].replace("\n", " ")
            if len(r.content) > 100:
                preview += "..."
            console.print(f"      [dim]{preview}[/dim]")
            if r.matched_question:
                console.print(f"      [yellow]Matched:[/yellow] {r.matched_question}")


def _run_inspect(
    data_dir: str | None,
    config_file: str | None,
    sources: bool,
    detailed: bool,
    plain: bool,
) -> None:
    """Shared logic for inspect command and its aliases."""
    from isotope.config import DEFAULT_DATA_DIR, load_config

    # Get effective data_dir for display
    config = load_config(config_file)
    effective_data_dir = data_dir or config.get("data_dir") or DEFAULT_DATA_DIR

    result = inspect.inspect(
        data_dir=data_dir,
        config_path=config_file,
        sources=sources,
        detailed=detailed,
    )

    if not result.success:
        _print_error(result.error, result.error_details, plain=plain)
        raise typer.Exit(1)

    if result.total_sources == 0:
        if plain:
            console.print("No database found.")
        else:
            console.print("[dim]No database found. Run 'isotope ingest' first.[/dim]")
        raise typer.Exit(0)

    # Sources-only mode: simple table like the old `list` command
    if sources and not detailed:
        if plain:
            console.print(f"Indexed sources ({result.total_sources}):")
            for source in result.sources:
                console.print(f"  {source.source} ({source.chunk_count} chunks)")
        else:
            table = Table(title=f"Indexed Sources ({result.total_sources})")
            table.add_column("Source", style="cyan")
            table.add_column("Chunks", justify="right")

            for source in result.sources:
                table.add_row(source.source, str(source.chunk_count))

            console.print(table)
        return

    # Default: aggregate stats
    if plain:
        console.print("Database Status:")
        console.print(f"  Data directory: {effective_data_dir}")
        console.print(f"  Sources: {result.total_sources}")
        console.print(f"  Chunks: {result.total_chunks}")
        console.print(f"  Atoms: {result.total_atoms}")
        console.print(f"  Indexed questions: {result.total_questions}")

        if detailed and result.sources:
            console.print()
            console.print("Question Distribution by Source:")
            for source in result.sources:
                console.print(
                    f"  {source.source}: {source.chunk_count} chunks, "
                    f"{source.question_count} questions"
                )
    else:
        table = Table(title="Database Status")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")

        table.add_row("Data directory", effective_data_dir)
        table.add_row("Sources", str(result.total_sources))
        table.add_row("Chunks", str(result.total_chunks))
        table.add_row("Atoms", str(result.total_atoms))
        table.add_row("Indexed questions", str(result.total_questions))

        console.print(table)

        if detailed and result.sources:
            console.print()
            detail_table = Table(title="Question Distribution by Source")
            detail_table.add_column("Source", style="cyan")
            detail_table.add_column("Chunks", justify="right")
            detail_table.add_column("Questions", justify="right", style="green")

            for source in result.sources:
                detail_table.add_row(
                    source.source,
                    str(source.chunk_count),
                    str(source.question_count),
                )

            console.print(detail_table)


@app.command(name="inspect", rich_help_panel="Database")
def inspect_cmd(
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
    sources: bool = typer.Option(
        False,
        "--sources",
        help="Show per-source breakdown with chunk counts",
    ),
    detailed: bool = typer.Option(
        False,
        "--detailed",
        help="Show per-source breakdown with chunks, atoms, and questions",
    ),
    plain: bool = typer.Option(
        False,
        "--plain",
        help="Plain output (no colors/formatting)",
    ),
) -> None:
    """Show database statistics."""
    _run_inspect(data_dir, config_file, sources, detailed, plain)


@app.command(name="list", hidden=True)
def list_alias_cmd(
    data_dir: str = typer.Option(None, "--data-dir", "-d"),
    config_file: str = typer.Option(None, "--config", "-c"),
    plain: bool = typer.Option(False, "--plain"),
) -> None:
    """Alias for 'inspect --sources'."""
    _run_inspect(data_dir, config_file, sources=True, detailed=False, plain=plain)


@app.command(name="status", hidden=True)
def status_alias_cmd(
    data_dir: str = typer.Option(None, "--data-dir", "-d"),
    config_file: str = typer.Option(None, "--config", "-c"),
    detailed: bool = typer.Option(False, "--detailed"),
    plain: bool = typer.Option(False, "--plain"),
) -> None:
    """Alias for 'inspect'."""
    _run_inspect(data_dir, config_file, sources=False, detailed=detailed, plain=plain)


@app.command(name="delete", rich_help_panel="Database")
def delete_cmd(
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

    def cli_confirm(request: ConfirmRequest) -> bool:
        """CLI confirmation callback using typer.confirm."""
        if request.details:
            if plain:
                console.print(request.details)
            else:
                console.print(f"[yellow]{request.details}[/yellow]")
        return typer.confirm("Continue?")

    # Use callback only if not --force
    on_confirm = None if force else cli_confirm

    result = delete.delete(
        source=source,
        data_dir=data_dir,
        config_path=config_file,
        on_confirm=on_confirm,
    )

    if not result.success:
        # Handle cancellation gracefully (exit 0, not error)
        if result.error == "Cancelled.":
            console.print("Cancelled.")
            raise typer.Exit(0)
        _print_error(result.error, result.error_details, plain=plain)
        raise typer.Exit(1)

    if plain:
        console.print(f"Deleted {result.chunks_deleted} chunks from {source}")
    else:
        console.print(f"[green]Deleted {result.chunks_deleted} chunks from {source}[/green]")


@app.command(name="config", rich_help_panel="Config")
def config_cmd_handler(
    config_file: str = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to config file",
    ),
) -> None:
    """Show current configuration settings."""
    result = config_cmd.config(config_path=config_file)

    if not result.success:
        _print_error(result.error, result.error_details)
        raise typer.Exit(1)

    table = Table(title="Isotope Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Source", style="dim")

    # Provider config
    table.add_row("provider", result.provider, result.provider_source)

    if result.provider == "litellm":
        table.add_row(
            "llm_model",
            result.llm_model or "(not set)",
            result.llm_model_source if result.llm_model else "not set",
        )
        table.add_row(
            "embedding_model",
            result.embedding_model or "(not set)",
            result.embedding_model_source if result.embedding_model else "not set",
        )

    # Data directory
    table.add_row("data_dir", result.data_dir, result.data_dir_source)

    # Separator
    table.add_row("", "", "")

    # Behavioral settings
    for setting in result.settings:
        table.add_row(setting.name, setting.value, setting.source)

    console.print(table)

    # Show config file location
    if result.config_path:
        console.print(f"\n[dim]Config file: {result.config_path}[/dim]")
    else:
        console.print("\n[dim]No config file found. Using env vars / defaults.[/dim]")

    console.print("\n[dim]Precedence: env var > yaml settings > default[/dim]")


@app.command(name="init", rich_help_panel="Core")
def init_cmd(
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

    def cli_prompt(request: PromptRequest) -> str:
        """Handle prompts in CLI context."""
        if request.choices:
            # Show question and numbered choices
            console.print()
            console.print(f"[bold]{request.message}[/bold]")
            for i, choice in enumerate(request.choices, 1):
                console.print(f"  [{i}] {choice}")

            default_index = "1"
            if request.default:
                for i, choice in enumerate(request.choices, 1):
                    if choice == request.default or choice.startswith(request.default):
                        default_index = str(i)
                        break

            response = typer.prompt("Choose", default=default_index)

            try:
                idx = int(response) - 1
                if 0 <= idx < len(request.choices):
                    return request.choices[idx]
            except ValueError:
                pass

            # Return first choice as fallback
            return request.choices[0]
        else:
            # Free text input
            return str(
                typer.prompt(
                    request.message,
                    default=request.default or "",
                    show_default=bool(request.default),
                    hide_input=request.is_secret,
                )
            )

    try:
        result = init.init(
            on_prompt=cli_prompt,
            provider=provider,
            llm_model=llm_model,
            embedding_model=embedding_model,
        )
    except InitCancelled:
        raise typer.Exit(0) from None

    if not result.success:
        _print_error(result.error, result.error_details)
        raise typer.Exit(1)

    console.print(f"\n[green]Created {result.config_path}[/green]")

    if result.env_path:
        console.print(f"[green]Saved API key(s) to {result.env_path}[/green]")

    if result.provider == "litellm":
        console.print()
        console.print("[bold]Ready![/bold] Try these commands:")
        console.print("  isotope ingest ./docs")
        console.print("  isotope query 'your question'")
    else:
        console.print()
        console.print("Next steps:")
        console.print("  1. Implement your custom classes")
        console.print("  2. Update the class paths in isotope.yaml")
        console.print("  3. Ingest documents: isotope ingest ./docs")


@app.command(name="questions", rich_help_panel="Database")
def questions_cmd(
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
    """Show a sample of indexed questions."""
    result = questions.sample_questions(
        n=n,
        source=source,
        data_dir=data_dir,
        config_path=config_file,
    )

    if not result.success:
        _print_error(result.error, result.error_details, plain=plain)
        raise typer.Exit(1)

    if not result.questions and not result.total:
        if source:
            if plain:
                console.print(f"No questions found for source: {source}")
            else:
                console.print(f"[yellow]No questions found for source: {source}[/yellow]")
        else:
            if plain:
                console.print("No questions indexed. Run 'isotope ingest' first.")
            else:
                console.print("[dim]No questions indexed. Run 'isotope ingest' first.[/dim]")
        raise typer.Exit(0)

    if plain:
        console.print(f"Sample Questions ({len(result.questions)} of {result.total}):")
        for i, q in enumerate(result.questions, 1):
            console.print(f"  {i}. {q.text}")
    else:
        title = f"Sample Questions ({len(result.questions)} of {result.total})"
        if source:
            title += f" from {source}"

        table = Table(title=title)
        table.add_column("#", style="dim", width=3)
        table.add_column("Question", style="cyan")

        for i, q in enumerate(result.questions, 1):
            table.add_row(str(i), q.text)

        console.print(table)

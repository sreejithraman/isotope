"""Main interactive screen with command handling."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from rich.table import Table
from textual.app import ComposeResult
from textual.css.query import NoMatches
from textual.screen import Screen
from textual.widgets import Footer

from isotope.commands import (
    ProgressUpdate,
    config_cmd,
    delete,
    ingest,
    list_cmd,
    query,
    status,
)
from isotope.commands.base import ConfirmRequest
from isotope.config import (
    DEFAULT_DATA_DIR,
    find_config_file,
    get_stores,
    load_config,
)
from isotope.tui.commands.parser import CommandParser, CommandType
from isotope.tui.screens.init import InitScreen
from isotope.tui.widgets.input_area import CommandInput, InputArea
from isotope.tui.widgets.output import OutputDisplay
from isotope.tui.widgets.status_bar import StatusBar
from isotope.tui.widgets.sticky_header import StickyHeader


class MainScreen(Screen[None]):
    """Main interactive screen with output, status bar, and input."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._parser = CommandParser()
        self._data_dir: str | None = None
        self._config: dict[str, Any] = {}
        self._current_task: asyncio.Task[None] | None = None

    def compose(self) -> ComposeResult:
        """Compose the main screen layout."""
        yield StickyHeader(id="sticky-header")
        yield OutputDisplay(id="output-area")
        yield StatusBar(id="status-bar")
        yield InputArea()
        yield Footer()

    async def on_mount(self) -> None:
        """Initialize the screen on mount."""
        # Load config and update status
        await self._load_config()
        self._update_status_bar()

        # Focus the input after DOM is fully ready
        self.call_after_refresh(self._focus_input)

    def _focus_input(self) -> None:
        """Focus the command input."""
        try:
            self.query_one(CommandInput).focus()
        except NoMatches:
            # Retry with timer if not ready
            self.set_timer(0.1, self._focus_input)

    async def _load_config(self) -> None:
        """Load isotope configuration."""
        config_path = find_config_file()
        self._config = load_config(config_path)
        self._data_dir = self._config.get("data_dir") or DEFAULT_DATA_DIR

    def _update_status_bar(self) -> None:
        """Update the status bar with current database info."""
        status_bar = self.query_one("#status-bar", StatusBar)

        if not self._data_dir or not os.path.exists(self._data_dir):
            status_bar.update_stats(questions=0, sources=0)
            return

        try:
            stores = get_stores(self._data_dir)
            sources = len(stores["chunk_store"].list_sources())
            questions = stores["embedded_question_store"].count_questions()

            # Get model name from config
            model = self._config.get("llm_model", "")
            # Shorten model name if too long
            if model and "/" in model:
                model = model.split("/")[-1]

            status_bar.update_stats(questions=questions, sources=sources, model=model)
        except Exception:
            status_bar.update_stats(questions=0, sources=0)

    async def on_command_input_submitted(self, event: CommandInput.Submitted) -> None:
        """Handle command submission."""
        output = self.query_one("#output-area", OutputDisplay)
        header = self.query_one("#sticky-header", StickyHeader)
        output.write_user_input(event.value)

        command = self._parser.parse(event.value)

        # Cancel any running task if starting a new command
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()

        # Hide sticky header
        header.hide()

        # Dispatch command
        try:
            if command.type == CommandType.QUIT:
                self.app.exit()
            elif command.type == CommandType.CLEAR:
                output.clear()
            elif command.type == CommandType.HELP:
                self._show_help(output)
            elif command.type == CommandType.STATUS:
                await self._cmd_status(output, command.flags)
            elif command.type == CommandType.LIST:
                await self._cmd_list(output)
            elif command.type == CommandType.CONFIG:
                await self._cmd_config(output)
            elif command.type == CommandType.INGEST:
                if not command.args:
                    output.write_error("Usage: ingest <path>")
                else:
                    self._current_task = asyncio.create_task(
                        self._cmd_ingest(output, header, command.args[0])
                    )
            elif command.type == CommandType.DELETE:
                if not command.args:
                    output.write_error("Usage: delete <source>")
                else:
                    await self._cmd_delete(output, command.args[0], command.flags)
            elif command.type == CommandType.INIT:
                await self._cmd_init(output)
            elif command.type == CommandType.QUERY:
                if command.args:
                    question = " ".join(command.args)
                    self._current_task = asyncio.create_task(
                        self._cmd_query(output, header, question, command.flags)
                    )
                else:
                    output.write_error("Please enter a question.")
            else:
                output.write_warning(f"Unknown command: {command.raw}")
                output.write_info("Type /help for available commands.")
        except Exception as e:
            output.write_error(f"Error: {e}")
            header.hide()

    def _show_help(self, output: OutputDisplay) -> None:
        """Show help information."""
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Command", style="#ff8700")
        table.add_column("Description", style="dim")

        commands = [
            ("init", "Initialize isotope configuration"),
            ("ingest <path>", "Index a file or directory"),
            ("query <question>", "Ask a question (or just type it)"),
            ("status", "Show database statistics"),
            ("list", "List indexed sources"),
            ("config", "Show configuration"),
            ("delete <source>", "Remove a source from index"),
            ("clear", "Clear the screen"),
            ("quit", "Exit isotope"),
        ]

        for cmd, desc in commands:
            table.add_row(cmd, desc)

        output.write_table(table)

        output.write_info("")
        output.write_info("Tips:")
        output.write_info("  Just type a question directly to query")
        output.write_info("  Use Tab for command/path completion")
        output.write_info("  Use Up/Down for command history")

    async def _cmd_status(self, output: OutputDisplay, flags: dict[str, Any]) -> None:
        """Show database status."""
        detailed = bool(flags.get("detailed") or flags.get("d"))
        result = status.status(data_dir=self._data_dir, detailed=detailed)

        if not result.success:
            output.write_error(result.error or "Unknown error")
            return

        if result.total_sources == 0:
            output.write_warning("No database found. Use 'ingest' first.")
            return

        table = Table(title="Database Status", box=None, title_style="#ff8700")
        table.add_column("Metric", style="#5fafaf")
        table.add_column("Value", style="#ff8700", justify="right")

        table.add_row("Sources", str(result.total_sources))
        table.add_row("Chunks", str(result.total_chunks))
        table.add_row("Atoms", str(result.total_atoms))
        table.add_row("Questions", str(result.total_questions))

        output.write_table(table)

        # Show detailed breakdown if requested
        if detailed and result.sources:
            detail_table = Table(title="By Source", box=None, title_style="#ff8700")
            detail_table.add_column("Source", style="#5fafaf")
            detail_table.add_column("Chunks", justify="right")
            detail_table.add_column("Questions", justify="right", style="#ff8700")

            for source_info in result.sources:
                detail_table.add_row(
                    source_info.source,
                    str(source_info.chunk_count),
                    str(source_info.question_count),
                )

            output.write_table(detail_table)

    async def _cmd_list(self, output: OutputDisplay) -> None:
        """List indexed sources."""
        result = list_cmd.list_sources(data_dir=self._data_dir)

        if not result.success:
            output.write_error(result.error or "Unknown error")
            return

        if not result.sources:
            output.write_info("No sources indexed.")
            return

        table = Table(
            title=f"Indexed Sources ({len(result.sources)})",
            box=None,
            title_style="#ff8700",
        )
        table.add_column("Source", style="#5fafaf")
        table.add_column("Chunks", justify="right")

        for source_info in result.sources:
            table.add_row(source_info.source, str(source_info.chunk_count))

        output.write_table(table)

    async def _cmd_config(self, output: OutputDisplay) -> None:
        """Show configuration."""
        result = config_cmd.config()

        if not result.success:
            output.write_error(result.error or "Unknown error")
            return

        table = Table(title="Configuration", box=None, title_style="#ff8700")
        table.add_column("Setting", style="#5fafaf")
        table.add_column("Value", style="#ff8700")
        table.add_column("Source", style="dim")

        # Provider config
        table.add_row("provider", result.provider, "yaml" if self._config else "default")

        if result.provider == "litellm":
            table.add_row(
                "llm_model",
                result.llm_model or "(not set)",
                "yaml" if self._config.get("llm_model") else "env var",
            )
            table.add_row(
                "embedding_model",
                result.embedding_model or "(not set)",
                "yaml" if self._config.get("embedding_model") else "env var",
            )

        # Data directory
        table.add_row(
            "data_dir",
            result.data_dir,
            "yaml" if self._config.get("data_dir") else "default",
        )

        table.add_row("", "", "")

        # Behavioral settings
        for setting in result.settings:
            table.add_row(setting.name, setting.value, setting.source)

        output.write_table(table)

        if result.config_path:
            output.write_info(f"Config file: {result.config_path}")

    async def _cmd_ingest(self, output: OutputDisplay, header: StickyHeader, path: str) -> None:
        """Ingest a file or directory."""
        if not os.path.exists(path):
            output.write_error(f"Path not found: {path}")
            return

        # Track file progress
        current_file_index = 0
        total_files = 0

        def on_file_start(filepath: str, index: int, total: int) -> None:
            nonlocal current_file_index, total_files
            current_file_index = index
            total_files = total
            filename = os.path.basename(filepath)
            header.show(f"Ingesting {index + 1}/{total}: {filename}")
            output.write_info(f"[{index + 1}/{total}] {filename}")

        def on_progress(update: ProgressUpdate) -> None:
            stage_name = update.stage.value
            if update.total > 1:
                output.write_progress(stage_name, update.current, update.total)
            else:
                output.write_info(f"   {stage_name}... {update.message or ''}")

        def on_file_complete(file_result: ingest.FileIngestResult) -> None:
            if file_result.skipped:
                output.write_info(f"   Skipped: {file_result.reason or 'unchanged'}")
            else:
                output.write_success(
                    f"{file_result.chunks} chunks, {file_result.questions} questions"
                )

        result = await asyncio.to_thread(
            ingest.ingest,
            path=path,
            data_dir=self._data_dir,
            on_progress=on_progress,
            on_file_start=on_file_start,
            on_file_complete=on_file_complete,
        )

        header.hide()

        if not result.success:
            output.write_error(result.error or "Unknown error")
            return

        if result.files_processed == 0 and result.files_skipped == 0:
            output.write_warning("No supported files found.")
            return

        output.write("")
        output.write_success(
            f"Done! Ingested {result.files_processed} files → {result.total_questions} questions"
        )
        if result.files_skipped:
            output.write_info(f"Skipped {result.files_skipped} unchanged files")

        self._update_status_bar()

    async def _cmd_delete(self, output: OutputDisplay, source: str, flags: dict[str, Any]) -> None:
        """Delete a source from the database."""
        force = flags.get("force") or flags.get("f")

        def tui_confirm(request: ConfirmRequest) -> bool:
            """TUI confirmation callback - shows warning and requires --force."""
            if request.details:
                output.write_warning(request.details)
            output.write_info("Use 'delete --force <source>' to confirm.")
            return False  # TUI requires explicit --force flag

        # Use callback only if not --force
        on_confirm = None if force else tui_confirm

        result = delete.delete(
            source=source,
            data_dir=self._data_dir,
            on_confirm=on_confirm,
        )

        if not result.success:
            # Handle cancellation gracefully (not an error)
            if result.error == "Cancelled.":
                return
            output.write_error(result.error or "Unknown error")
            return

        output.write_success(f"Deleted {result.chunks_deleted} chunks from {source}")
        self._update_status_bar()

    async def _cmd_init(self, output: OutputDisplay) -> None:
        """Launch the init screen."""
        # Check if config already exists
        config_path = Path("isotope.yaml")
        if config_path.exists():
            output.write_warning("isotope.yaml already exists.")
            output.write_info("Use 'init' to overwrite, or edit the file directly.")

        def handle_init_result(result: bool | None) -> None:
            """Handle the result from init screen."""
            if result:
                output.write_success("Created isotope.yaml")
                output.write_info("Ready! Try: ingest <path>")
                # Reload config
                asyncio.create_task(self._load_config())
                self._update_status_bar()
            else:
                output.write_info("Init cancelled.")

        self.app.push_screen(InitScreen(), handle_init_result)

    async def _cmd_query(
        self, output: OutputDisplay, header: StickyHeader, question: str, flags: dict[str, Any]
    ) -> None:
        """Query the knowledge base."""
        header.show("Searching...")
        output.write_info("Searching...")

        # Parse flags
        raw = bool(flags.get("raw") or flags.get("r"))
        k = int(flags.get("k", 5)) if flags.get("k") else None
        show_questions = bool(flags.get("show-matched-questions") or flags.get("q"))

        result = await asyncio.to_thread(
            query.query,
            question=question,
            data_dir=self._data_dir,
            k=k,
            raw=raw,
            show_matched_questions=show_questions,
        )

        header.hide()

        if not result.success:
            error_msg = result.error or "Unknown error"
            if "Data directory not found" in error_msg:
                output.write_warning("No database found. Use 'ingest' first.")
            else:
                output.write_error(error_msg)
            return

        if not result.results:
            output.write_warning("No results found.")
            return

        # Show synthesized answer
        if result.answer:
            output.write_markdown(result.answer, title="Answer")

        # Show sources
        sources = [(r.source, r.score) for r in result.results]
        output.write_sources(sources)

        # Show matched questions if requested
        if show_questions:
            output.write("")
            output.write_info("Matched questions:")
            for i, search_result in enumerate(result.results, 1):
                if search_result.matched_question:
                    output.write_info(f"  {i}. {search_result.matched_question}")

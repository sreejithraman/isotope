"""Output display widget for conversation-style output."""

from __future__ import annotations

from typing import Any

from rich.console import RenderableType
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.widgets import RichLog


class OutputDisplay(RichLog):
    """Scrollable output area with rich formatting support."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            highlight=True,
            markup=True,
            wrap=True,
            **kwargs,
        )

    def write_user_input(self, text: str) -> None:
        """Display user input with prompt prefix."""
        prompt = Text()
        prompt.append("> ", style="bold #ff8700")
        prompt.append(text, style="#ffaf5f")
        self.write(prompt)

    def write_response(self, content: RenderableType) -> None:
        """Display a response."""
        self.write(content)
        self.write("")  # Add spacing

    def write_markdown(self, content: str, title: str | None = None) -> None:
        """Display markdown content in a styled panel."""
        md = Markdown(content)
        if title:
            panel = Panel(
                md,
                title=f"[bold]{title}[/bold]",
                title_align="left",
                border_style="#87d787",
                padding=(0, 1),
            )
            self.write(panel)
        else:
            self.write(md)
        self.write("")

    def write_table(self, table: Table) -> None:
        """Display a table."""
        self.write(table)
        self.write("")

    def write_info(self, message: str) -> None:
        """Display an info message."""
        self.write(Text(message, style="dim"))

    def write_success(self, message: str) -> None:
        """Display a success message."""
        text = Text()
        text.append(" ", style="#87d787")
        text.append(message, style="#87d787")
        self.write(text)

    def write_warning(self, message: str) -> None:
        """Display a warning message."""
        text = Text()
        text.append(" ", style="#ffaf5f")
        text.append(message, style="#ffaf5f")
        self.write(text)

    def write_error(self, message: str) -> None:
        """Display an error message."""
        text = Text()
        text.append(" ", style="#ff8787")
        text.append(message, style="#ff8787")
        self.write(text)

    def write_progress(self, stage: str, current: int, total: int, detail: str = "") -> None:
        """Display progress information with a styled bar."""
        if total > 0:
            pct = current / total
            bar_width = 20
            filled = int(bar_width * pct)

            text = Text()
            text.append(f"  {stage:>12} ", style="#ff8700")
            text.append("[", style="dim")
            text.append("\u2588" * filled, style="#ff8700")
            text.append("\u2591" * (bar_width - filled), style="dim")
            text.append("]", style="dim")
            text.append(f" {current}/{total}", style="dim")
            if detail:
                text.append(f" {detail}", style="dim")
            self.write(text)
        else:
            text = Text()
            text.append(f"   {stage} ", style="#ff8700")
            if detail:
                text.append(detail, style="dim")
            self.write(text)

    def write_sources(self, sources: list[tuple[str, float]]) -> None:
        """Display source references with scores."""
        text = Text()
        text.append("\n", style="")
        text.append("\u2500" * 50, style="dim")
        text.append("\n", style="")
        text.append(" Sources: ", style="dim")

        for i, (src, score) in enumerate(sources):
            if i > 0:
                text.append("  ", style="dim")
            text.append(src, style="#5fafaf")
            # Show score as percentage with color intensity based on confidence
            score_pct = int(score * 100)
            if score >= 0.8:
                score_style = "bold #87d787"
            elif score >= 0.5:
                score_style = "#ffaf5f"
            else:
                score_style = "dim #ff8787"
            text.append(f" ({score_pct}%)", style=score_style)

        self.write(text)

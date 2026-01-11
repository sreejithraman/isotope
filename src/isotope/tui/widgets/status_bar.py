"""Status bar widget showing database stats."""

from __future__ import annotations

from typing import Any

from rich.text import Text
from textual.reactive import reactive
from textual.widgets import Static


class StatusBar(Static):
    """Status bar showing indexed questions, sources, and model info."""

    questions: reactive[int] = reactive(0)
    sources: reactive[int] = reactive(0)
    model: reactive[str] = reactive("")

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def render(self) -> Text:
        """Render the status bar content."""
        text = Text()

        if self.questions == 0 and self.sources == 0:
            text.append(" No database ", style="dim")
            text.append("| ", style="dim")
            text.append("Run ", style="dim")
            text.append("ingest <path>", style="#ff8700")
            text.append(" to get started", style="dim")
        else:
            # Atom symbol
            text.append(" ", style="#ff8700")
            text.append(f"{self.questions:,}", style="bold #ff8700")
            text.append(" questions", style="dim")
            text.append("  |  ", style="dim")
            text.append(f"{self.sources}", style="bold #ff8700")
            text.append(" sources", style="dim")

            if self.model:
                text.append("  |  ", style="dim")
                text.append(self.model, style="#5fafaf dim")

        return text

    def update_stats(
        self, questions: int | None = None, sources: int | None = None, model: str | None = None
    ) -> None:
        """Update the status bar statistics."""
        if questions is not None:
            self.questions = questions
        if sources is not None:
            self.sources = sources
        if model is not None:
            self.model = model

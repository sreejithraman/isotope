"""Tips panel for first-launch guidance."""

from __future__ import annotations

from rich.panel import Panel
from rich.text import Text
from textual.widgets import Static

TIPS = [
    ("Ingest documents", "ingest ./docs", "Index files or directories"),
    ("Ask questions", "how does auth work?", "Natural language queries"),
    ("View status", "status", "See what's indexed"),
    ("Get help", "/help", "See all commands"),
]


class TipsPanel(Static):
    """Panel showing getting started tips."""

    def render(self) -> Panel:
        """Render the tips panel."""
        content = Text()
        content.append(" Getting Started\n\n", style="bold #ff8700")

        for _tip, example, description in TIPS:
            content.append("  ", style="")
            content.append(example, style="#5fafaf")
            content.append("\n", style="")
            content.append(f"    {description}\n\n", style="dim")

        content.append("  Press ", style="dim")
        content.append("Tab", style="bold #ff8700")
        content.append(" for completion, ", style="dim")
        content.append("Up/Down", style="bold #ff8700")
        content.append(" for history\n", style="dim")

        return Panel(
            content,
            border_style="#ff8700 dim",
            padding=(0, 2),
            expand=False,
        )

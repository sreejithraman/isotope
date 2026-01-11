"""Sticky header widget for showing context during operations."""

from __future__ import annotations

from rich.text import Text
from textual.reactive import reactive
from textual.widgets import Static


class StickyHeader(Static):
    """Context header shown during long-running operations."""

    message: reactive[str] = reactive("")
    is_visible: reactive[bool] = reactive(False)

    def render(self) -> Text:
        """Render the header content."""
        text = Text()
        if self.message:
            text.append(" ", style="#ff8700")
            text.append(self.message, style="#ffaf5f")
        return text

    def watch_is_visible(self, is_visible: bool) -> None:
        """React to visibility changes."""
        self.set_class(is_visible, "-visible")

    def show(self, message: str) -> None:
        """Show the header with a message."""
        self.message = message
        self.is_visible = True

    def hide(self) -> None:
        """Hide the header."""
        self.is_visible = False
        self.message = ""

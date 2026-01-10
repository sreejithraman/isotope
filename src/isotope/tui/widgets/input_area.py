"""Anchored input area with command history and tab completion."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Container
from textual.message import Message
from textual.widgets import Input

if TYPE_CHECKING:
    from textual.events import Key


class CommandInput(Input):
    """Input widget with command history and tab completion."""

    BINDINGS = [
        ("up", "history_prev", "Previous command"),
        ("down", "history_next", "Next command"),
        ("tab", "complete", "Complete"),
    ]

    class Submitted(Message):
        """Posted when command is submitted."""

        def __init__(self, value: str) -> None:
            self.value = value
            super().__init__()

    def __init__(
        self,
        placeholder: str = "Ask a question or type /help...",
        *,
        id: str | None = None,
    ) -> None:
        super().__init__(placeholder=placeholder, id=id)
        self._history: list[str] = []
        self._history_index: int = -1
        self._current_input: str = ""
        self._commands = [
            "ingest",
            "query",
            "status",
            "list",
            "config",
            "delete",
            "help",
            "quit",
            "clear",
        ]

    def action_history_prev(self) -> None:
        """Navigate to previous command in history."""
        if not self._history:
            return

        if self._history_index == -1:
            self._current_input = self.value
            self._history_index = len(self._history) - 1
        elif self._history_index > 0:
            self._history_index -= 1

        self.value = self._history[self._history_index]
        self.cursor_position = len(self.value)

    def action_history_next(self) -> None:
        """Navigate to next command in history."""
        if self._history_index == -1:
            return

        if self._history_index < len(self._history) - 1:
            self._history_index += 1
            self.value = self._history[self._history_index]
        else:
            self._history_index = -1
            self.value = self._current_input

        self.cursor_position = len(self.value)

    def action_complete(self) -> None:
        """Tab completion for commands and file paths."""
        text = self.value.strip()

        if not text:
            return

        parts = text.split(maxsplit=1)
        cmd = parts[0].lower()

        # Complete command names
        if len(parts) == 1 and not text.endswith(" "):
            matches = [c for c in self._commands if c.startswith(cmd)]
            if len(matches) == 1:
                self.value = matches[0] + " "
                self.cursor_position = len(self.value)
            return

        # Complete file paths for ingest/delete commands
        if cmd in ("ingest", "delete", "add", "rm") and len(parts) > 1:
            path_part = parts[1]
            self._complete_path(cmd, path_part)

    def _complete_path(self, cmd: str, partial: str) -> None:
        """Complete a file path."""
        if not partial:
            partial = "./"

        path = Path(partial).expanduser()

        if path.is_dir():
            parent = path
            prefix = ""
        else:
            parent = path.parent if path.parent.exists() else Path(".")
            prefix = path.name

        try:
            entries = list(parent.iterdir())
        except (PermissionError, OSError):
            return

        matches = [e for e in entries if e.name.startswith(prefix) and not e.name.startswith(".")]

        if len(matches) == 1:
            completed = matches[0]
            completed_str = str(completed) + os.sep if completed.is_dir() else str(completed)
            self.value = f"{cmd} {completed_str}"
            self.cursor_position = len(self.value)
        elif len(matches) > 1:
            # Find common prefix
            common = os.path.commonprefix([str(m) for m in matches])
            if common and len(common) > len(str(path)):
                self.value = f"{cmd} {common}"
                self.cursor_position = len(self.value)

    def add_to_history(self, command: str) -> None:
        """Add a command to history."""
        command = command.strip()
        if command and (not self._history or self._history[-1] != command):
            self._history.append(command)
        self._history_index = -1
        self._current_input = ""

    async def _on_key(self, event: Key) -> None:
        """Handle key events."""
        if event.key == "enter":
            if self.value.strip():
                self.post_message(self.Submitted(self.value))
                self.add_to_history(self.value)
                self.value = ""
            event.stop()
            event.prevent_default()


class InputArea(Container):
    """Container for the anchored input at the bottom of the screen."""

    def compose(self) -> ComposeResult:
        """Compose the input area with the command input."""
        yield CommandInput(id="command-input")

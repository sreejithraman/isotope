"""Main TUI application for Isotope."""

from __future__ import annotations

import os
from typing import Any

from textual.app import App
from textual.binding import Binding

from isotope.tui.screens.main import MainScreen
from isotope.tui.screens.welcome import WelcomeScreen


class IsotopeTUI(App[None]):
    """Isotope TUI - Beautiful terminal interface for Reverse RAG."""

    TITLE = "isotope"
    CSS_PATH = "styles.tcss"

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=True),
        Binding("ctrl+l", "clear", "Clear", show=True),
    ]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._show_welcome = True

    def on_mount(self) -> None:
        """Set up the initial screen."""
        if self._show_welcome and self._should_show_welcome():
            self.push_screen(WelcomeScreen())
            self.push_screen(MainScreen())
            self.pop_screen()  # Pop main, leaving welcome on top
        else:
            self.push_screen(MainScreen())

    def _should_show_welcome(self) -> bool:
        """Determine if welcome screen should be shown."""
        # Show welcome if no database exists yet
        from isotope.config import DEFAULT_DATA_DIR, find_config_file, load_config

        config_path = find_config_file()
        config = load_config(config_path)
        data_dir = config.get("data_dir") or DEFAULT_DATA_DIR

        return not os.path.exists(data_dir)

    async def action_quit(self) -> None:
        """Quit the application."""
        self.exit()

    def action_clear(self) -> None:
        """Clear the output area on the current screen."""
        from isotope.tui.widgets.output import OutputDisplay

        try:
            output = self.screen.query_one("#output-area", OutputDisplay)
            output.clear()
        except Exception:
            pass  # Screen might not have output area


def main() -> None:
    """Entry point for the TUI."""
    app = IsotopeTUI()
    app.run()


if __name__ == "__main__":
    main()

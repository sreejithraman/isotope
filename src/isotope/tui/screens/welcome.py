"""Welcome screen with ASCII art logo and tips."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll
from textual.screen import Screen

from isotope.tui.widgets.ascii_logo import ASCIILogo
from isotope.tui.widgets.input_area import CommandInput, InputArea
from isotope.tui.widgets.tips import TipsPanel


class WelcomeScreen(Screen):
    """Welcome screen displayed on first launch."""

    def compose(self) -> ComposeResult:
        """Compose the welcome screen layout."""
        with VerticalScroll(id="welcome-content"):
            yield Container(id="welcome-spacer")
            yield ASCIILogo()
            yield TipsPanel()
        yield InputArea()

    def on_mount(self) -> None:
        """Focus the input when mounted."""
        self.call_after_refresh(self._focus_input)

    def _focus_input(self) -> None:
        """Focus the command input."""
        try:
            self.query_one(CommandInput).focus()
        except Exception:
            # Retry with timer if not ready
            self.set_timer(0.1, self._focus_input)

    async def on_command_input_submitted(self, event: CommandInput.Submitted) -> None:
        """Handle command submission - switch to main screen."""
        # Dismiss welcome screen and go to main with the command
        self.app.pop_screen()
        # Re-emit the event so main screen can handle it
        self.app.screen.post_message(event)

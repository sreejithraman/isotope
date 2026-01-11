"""Tests for TUI widgets."""

from isotope.tui.widgets.input_area import CommandInput
from isotope.tui.widgets.status_bar import StatusBar
from isotope.tui.widgets.sticky_header import StickyHeader


class TestCommandInput:
    """Tests for CommandInput widget."""

    def test_history_empty(self) -> None:
        """Test history navigation with no history."""
        widget = CommandInput()
        # Should not raise
        widget.action_history_prev()
        widget.action_history_next()

    def test_add_to_history(self) -> None:
        """Test adding commands to history."""
        widget = CommandInput()
        widget.add_to_history("status")
        widget.add_to_history("query test")

        assert len(widget._history) == 2
        assert widget._history[0] == "status"
        assert widget._history[1] == "query test"

    def test_history_no_duplicates(self) -> None:
        """Test that consecutive duplicates are not added."""
        widget = CommandInput()
        widget.add_to_history("status")
        widget.add_to_history("status")

        assert len(widget._history) == 1

    def test_history_allows_non_consecutive_duplicates(self) -> None:
        """Test that non-consecutive duplicates are allowed."""
        widget = CommandInput()
        widget.add_to_history("status")
        widget.add_to_history("query test")
        widget.add_to_history("status")

        assert len(widget._history) == 3

    def test_history_strips_whitespace(self) -> None:
        """Test that whitespace is stripped from history entries."""
        widget = CommandInput()
        widget.add_to_history("  status  ")

        assert widget._history[0] == "status"

    def test_history_skips_empty(self) -> None:
        """Test that empty commands are not added."""
        widget = CommandInput()
        widget.add_to_history("")
        widget.add_to_history("   ")

        assert len(widget._history) == 0

    def test_commands_list(self) -> None:
        """Test that default commands are set."""
        widget = CommandInput()
        assert "ingest" in widget._commands
        assert "query" in widget._commands
        assert "status" in widget._commands
        assert "list" in widget._commands
        assert "config" in widget._commands
        assert "delete" in widget._commands
        assert "help" in widget._commands
        assert "quit" in widget._commands


class TestStatusBar:
    """Tests for StatusBar widget."""

    def test_initial_state(self) -> None:
        """Test initial state of status bar."""
        bar = StatusBar()
        assert bar.questions == 0
        assert bar.sources == 0
        assert bar.model == ""

    def test_update_stats(self) -> None:
        """Test updating status bar stats."""
        bar = StatusBar()
        bar.update_stats(questions=100, sources=5, model="gpt-4o-mini")

        assert bar.questions == 100
        assert bar.sources == 5
        assert bar.model == "gpt-4o-mini"

    def test_partial_update(self) -> None:
        """Test partial updates to status bar."""
        bar = StatusBar()
        bar.update_stats(questions=50)
        assert bar.questions == 50
        assert bar.sources == 0

        bar.update_stats(sources=3)
        assert bar.questions == 50
        assert bar.sources == 3


class TestStickyHeader:
    """Tests for StickyHeader widget."""

    def test_initial_state(self) -> None:
        """Test initial state of sticky header."""
        header = StickyHeader()
        assert header.message == ""
        assert header.is_visible is False

    def test_show(self) -> None:
        """Test showing the header."""
        header = StickyHeader()
        header.show("Processing file.txt")

        assert header.message == "Processing file.txt"
        assert header.is_visible is True

    def test_hide(self) -> None:
        """Test hiding the header."""
        header = StickyHeader()
        header.show("Processing")
        header.hide()

        assert header.message == ""
        assert header.is_visible is False

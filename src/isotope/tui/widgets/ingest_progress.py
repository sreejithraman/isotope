"""Progress display widget for ingest operations."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Container
from textual.reactive import reactive
from textual.widgets import Label, ProgressBar, Static

# User-friendly stage names
STAGE_DISPLAY_NAMES = {
    "Storing": "Saving",
    "Atomizing": "Analyzing",
    "Generating": "Processing",
    "Embedding": "Indexing",
    "Filtering": "Optimizing",
    "Indexing": "Finalizing",
}


class IngestProgress(Container):
    """Progress container with file and stage progress bars."""

    DEFAULT_CSS = """
    IngestProgress {
        dock: top;
        height: auto;
        max-height: 5;
        background: $surface;
        border: round $primary 40%;
        padding: 0 1;
        margin: 0 1 1 1;
        display: none;
    }

    IngestProgress.-visible {
        display: block;
    }

    IngestProgress Label {
        width: 10;
        color: $text-muted;
    }

    IngestProgress .progress-row {
        height: 1;
        layout: horizontal;
    }

    IngestProgress ProgressBar {
        width: 1fr;
        padding: 0 1;
    }

    IngestProgress .file-name {
        width: auto;
        max-width: 30;
        color: $warning;
    }

    IngestProgress .stage-name {
        width: auto;
        max-width: 20;
        color: $text-muted;
    }
    """

    is_visible: reactive[bool] = reactive(False)
    current_file: reactive[str] = reactive("")
    current_stage: reactive[str] = reactive("")

    def compose(self) -> ComposeResult:
        with Container(classes="progress-row"):
            yield Label("Files:")
            yield ProgressBar(id="file-progress", total=100, show_eta=False)
            yield Static("", id="file-name", classes="file-name")
        with Container(classes="progress-row"):
            yield Label("Stage:")
            yield ProgressBar(id="stage-progress", total=100, show_eta=False)
            yield Static("", id="stage-name", classes="stage-name")

    def watch_is_visible(self, visible: bool) -> None:
        self.set_class(visible, "-visible")

    def watch_current_file(self, name: str) -> None:
        self.query_one("#file-name", Static).update(name)

    def watch_current_stage(self, stage: str) -> None:
        display_name = STAGE_DISPLAY_NAMES.get(stage, stage)
        self.query_one("#stage-name", Static).update(display_name)

    def show(self) -> None:
        """Show the progress container."""
        self.is_visible = True

    def hide(self) -> None:
        """Hide the progress container."""
        self.is_visible = False

    def update_file_progress(self, current: int, total: int, filename: str) -> None:
        """Update file-level progress."""
        bar = self.query_one("#file-progress", ProgressBar)
        bar.update(total=total, progress=current)
        self.current_file = filename

    def update_stage_progress(self, current: int, total: int, stage: str) -> None:
        """Update stage-level progress."""
        bar = self.query_one("#stage-progress", ProgressBar)
        if total > 0:
            bar.update(total=total, progress=current)
        else:
            bar.update(total=100, progress=0)  # Indeterminate
        self.current_stage = stage

    def reset(self) -> None:
        """Reset progress bars."""
        self.query_one("#file-progress", ProgressBar).update(progress=0)
        self.query_one("#stage-progress", ProgressBar).update(progress=0)
        self.current_file = ""
        self.current_stage = ""

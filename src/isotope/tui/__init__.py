"""Isotope TUI - Interactive terminal interface for Isotope."""

from __future__ import annotations

import sys
from typing import Any

__all__ = ["IsotopeTUI", "main"]


def __getattr__(name: str) -> Any:
    """Lazy import IsotopeTUI to avoid importing textual at module load."""
    if name == "IsotopeTUI":
        try:
            from isotope.tui.app import IsotopeTUI

            return IsotopeTUI
        except ImportError as e:
            if "textual" in str(e).lower() or "rich" in str(e).lower():
                raise ImportError(
                    "The TUI requires additional dependencies. "
                    "Install with: pip install isotope-rag[tui]"
                ) from e
            raise
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def main() -> None:
    """Entry point for isotope-tui command."""
    try:
        from isotope.tui.app import main as _main

        _main()
    except ImportError as e:
        if "textual" in str(e).lower() or "rich" in str(e).lower():
            print(
                "Error: The TUI requires additional dependencies.\n"
                "Install with: pip install isotope-rag[tui]",
                file=sys.stderr,
            )
            sys.exit(1)
        raise

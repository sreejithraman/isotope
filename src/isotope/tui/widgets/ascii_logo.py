"""Chemistry-themed ASCII art logo for Isotope."""

from __future__ import annotations

from rich.text import Text
from textual.widgets import Static

# Abstract atomic design with electron cloud
# The dots/asterisks represent electron probability clouds
# The circle represents the nucleus


def _build_logo() -> Text:
    """Build the logo text with styling."""
    text = Text()

    # Electron cloud - top
    text.append("         .  *  .\n", style="dim")
    text.append("        *  ", style="dim")
    text.append("(o)", style="bold #ff8700")
    text.append("  *\n", style="dim")
    text.append("         `  *  `\n", style="dim")
    text.append("\n")

    # Atom drawing + ISOTOPE text
    text.append("          ", style="")
    text.append(".--.", style="#d75f00")
    text.append("           ___ ____   ___  _____  ___  ____  _____\n", style="#ff8700")

    text.append("         ", style="")
    text.append("/    \\", style="#d75f00")
    text.append("         |_ _/ ___| / _ \\|_   _|/ _ \\|  _ \\| ____|\n", style="#ff8700")

    text.append("         ", style="")
    text.append("|  ", style="#d75f00")
    text.append("()", style="bold #ffaf5f")
    text.append("  |", style="#d75f00")
    text.append("          | |\\___ \\| | | | | | | | | | |_) |  _|\n", style="#ff8700")

    text.append("         ", style="")
    text.append("\\    /", style="#d75f00")
    text.append("          | | ___) | |_| | | | | |_| |  __/| |___\n", style="#ff8700")

    text.append("          ", style="")
    text.append("`--'", style="#d75f00")
    text.append("          |___|____/ \\___/  |_|  \\___/|_|   |_____|\n", style="#ff8700")

    text.append("\n")

    # Tagline
    text.append("                       ", style="")
    text.append("Reverse RAG", style="bold #ffaf5f")
    text.append(" - ", style="dim")
    text.append("Index Questions, Not Chunks\n", style="#808080")
    text.append("\n")

    return text


class ASCIILogo(Static):
    """Chemistry-themed ASCII art logo widget."""

    def render(self) -> Text:
        """Render the logo with gradient coloring."""
        return _build_logo()

"""TUI widgets for Isotope."""

from isotope.tui.widgets.ascii_logo import ASCIILogo
from isotope.tui.widgets.input_area import CommandInput, InputArea
from isotope.tui.widgets.output import OutputDisplay
from isotope.tui.widgets.status_bar import StatusBar
from isotope.tui.widgets.sticky_header import StickyHeader
from isotope.tui.widgets.tips import TipsPanel

__all__ = [
    "ASCIILogo",
    "CommandInput",
    "InputArea",
    "OutputDisplay",
    "StatusBar",
    "StickyHeader",
    "TipsPanel",
]

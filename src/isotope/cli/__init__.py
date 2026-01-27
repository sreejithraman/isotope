# src/isotope/cli/__init__.py
"""CLI package for Isotope.

This package provides the command-line interface using Typer.
The CLI is a thin wrapper around the commands layer.
"""

from isotope.cli.app import app, console

__all__ = ["app", "console"]

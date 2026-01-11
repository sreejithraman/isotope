# src/isotope/commands/__init__.py
"""UI-agnostic command layer for Isotope.

This module provides command functions that both CLI and TUI can call.
Commands return data structures, allowing UIs to render results appropriately.

Usage:
    from isotope.commands import ingest, query, status

    # Ingest files
    result = ingest.ingest("./docs", on_progress=my_callback)

    # Query the database
    result = query.query("How does authentication work?")

    # Get database status
    result = status.status()
"""

# Import command modules for easy access
from isotope.commands import config_cmd, delete, ingest, init, query, status
from isotope.commands import list as list_cmd
from isotope.commands.base import (
    CommandResult,
    CommandStage,
    ConfigResult,
    ConfirmCallback,
    ConfirmRequest,
    DeleteResult,
    FileIngestResult,
    IngestResult,
    InitPrompt,
    InitResult,
    ListResult,
    ProgressCallback,
    ProgressUpdate,
    PromptCallback,
    PromptRequest,
    QueryResult,
    SearchResult,
    SettingInfo,
    SourceInfo,
    StatusResult,
)

__all__ = [
    # Base types
    "CommandStage",
    "ProgressUpdate",
    "ProgressCallback",
    "InitPrompt",
    "PromptRequest",
    "PromptCallback",
    "ConfirmRequest",
    "ConfirmCallback",
    "CommandResult",
    # Result types
    "IngestResult",
    "FileIngestResult",
    "QueryResult",
    "SearchResult",
    "StatusResult",
    "SourceInfo",
    "ListResult",
    "DeleteResult",
    "ConfigResult",
    "SettingInfo",
    "InitResult",
    # Command modules
    "ingest",
    "query",
    "status",
    "list_cmd",
    "delete",
    "config_cmd",
    "init",
]

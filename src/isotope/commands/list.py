# src/isotope/commands/list.py
"""List command - list indexed sources.

This module provides the list logic that both CLI and TUI use.
"""

from __future__ import annotations

import os
from pathlib import Path

from isotope.commands.base import ListResult, SourceInfo
from isotope.config import DEFAULT_DATA_DIR, get_stores, load_config


def list_sources(
    data_dir: str | None = None,
    config_path: str | Path | None = None,
) -> ListResult:
    """List all indexed sources.

    Args:
        data_dir: Override data directory
        config_path: Override config file path

    Returns:
        ListResult with source information
    """
    config = load_config(config_path)
    effective_data_dir = data_dir or config.get("data_dir") or DEFAULT_DATA_DIR

    if not os.path.exists(effective_data_dir):
        return ListResult(
            success=True,
            sources=[],
        )

    try:
        stores = get_stores(effective_data_dir)
    except Exception as e:
        return ListResult(
            success=False,
            error=f"Failed to access database: {e}",
        )

    chunk_store = stores["chunk_store"]
    sources = chunk_store.list_sources()

    if not sources:
        return ListResult(
            success=True,
            sources=[],
        )

    result = ListResult(success=True)

    for source in sorted(sources):
        chunks = chunk_store.get_by_source(source)
        result.sources.append(
            SourceInfo(
                source=source,
                chunk_count=len(chunks),
            )
        )

    return result

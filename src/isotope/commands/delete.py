# src/isotope/commands/delete.py
"""Delete command - remove sources from the database.

This module provides the delete logic that both CLI and TUI use.
"""

from __future__ import annotations

import os
from pathlib import Path

from isotope.commands.base import DeleteResult
from isotope.config import DEFAULT_DATA_DIR, get_stores, load_config


def delete(
    source: str,
    data_dir: str | None = None,
    config_path: str | Path | None = None,
) -> DeleteResult:
    """Delete a source from the database.

    Args:
        source: Source path/name to delete
        data_dir: Override data directory
        config_path: Override config file path

    Returns:
        DeleteResult with deletion statistics
    """
    config = load_config(config_path)
    effective_data_dir = data_dir or config.get("data_dir") or DEFAULT_DATA_DIR

    if not os.path.exists(effective_data_dir):
        return DeleteResult(
            success=False,
            source=source,
            error="No database found.",
        )

    try:
        stores = get_stores(effective_data_dir)
    except Exception as e:
        return DeleteResult(
            success=False,
            source=source,
            error=f"Failed to access database: {e}",
        )

    chunk_store = stores["chunk_store"]
    atom_store = stores["atom_store"]
    question_store = stores["embedded_question_store"]
    source_registry = stores["source_registry"]

    # Check if source exists
    sources = chunk_store.list_sources()
    if source not in sources:
        return DeleteResult(
            success=False,
            source=source,
            error=f"Source not found: {source}",
        )

    # Get chunk IDs for this source
    chunk_ids = chunk_store.get_chunk_ids_by_source(source)

    # Count items to delete
    chunks_deleted = len(chunk_ids)
    atoms_deleted = sum(len(atom_store.get_by_chunk(cid)) for cid in chunk_ids)
    questions_deleted = question_store.count_by_chunk_ids(chunk_ids)

    # Delete in order: questions -> atoms -> chunks -> source registry
    question_store.delete_by_chunk_ids(chunk_ids)
    atom_store.delete_by_chunk_ids(chunk_ids)
    chunk_store.delete_many(chunk_ids)

    # Delete from source registry
    source_registry.delete(source)

    return DeleteResult(
        success=True,
        source=source,
        chunks_deleted=chunks_deleted,
        atoms_deleted=atoms_deleted,
        questions_deleted=questions_deleted,
    )

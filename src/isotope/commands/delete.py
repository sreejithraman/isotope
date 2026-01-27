# src/isotope/commands/delete.py
"""Delete command - remove sources from the database.

This module provides the delete logic that both CLI and TUI use.
It uses callbacks for interactive confirmation, allowing each UI to
implement their own confirmation method.
"""

from __future__ import annotations

import os
from pathlib import Path

from isotope.commands.base import ConfirmCallback, ConfirmRequest, DeleteResult
from isotope.config import DEFAULT_DATA_DIR, get_stores, load_config


def delete(
    source: str,
    data_dir: str | None = None,
    config_path: str | Path | None = None,
    on_confirm: ConfirmCallback | None = None,
) -> DeleteResult:
    """Delete a source from the database.

    This function uses a callback for confirmation, allowing different UIs
    (CLI, TUI) to implement their own confirmation methods.

    Args:
        source: Source path/name to delete
        data_dir: Override data directory
        config_path: Override config file path
        on_confirm: Optional callback for confirmation. If provided, it will be
            called with details about what will be deleted. Return True to
            proceed, False to cancel. If None, deletion proceeds without
            confirmation (equivalent to --force).

    Returns:
        DeleteResult with deletion statistics, or cancelled result
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

    # Count items that will be deleted
    chunks_to_delete = len(chunk_ids)
    atoms_to_delete = atom_store.count_by_chunk_ids(chunk_ids)
    questions_to_delete = question_store.count_by_chunk_ids(chunk_ids)

    # Confirm deletion if callback provided
    if on_confirm is not None:
        confirm_request = ConfirmRequest(
            message=f"Delete {source}?",
            details=f"This will remove {chunks_to_delete} chunks from the database.",
        )
        if not on_confirm(confirm_request):
            return DeleteResult(
                success=False,
                source=source,
                error="Cancelled.",
            )

    # Delete in order: questions -> atoms -> chunks -> source registry
    question_store.delete_by_chunk_ids(chunk_ids)
    atom_store.delete_by_chunk_ids(chunk_ids)
    chunk_store.delete_many(chunk_ids)

    # Delete from source registry
    source_registry.delete(source)

    return DeleteResult(
        success=True,
        source=source,
        chunks_deleted=chunks_to_delete,
        atoms_deleted=atoms_to_delete,
        questions_deleted=questions_to_delete,
    )

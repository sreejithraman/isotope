# src/isotope/commands/inspect.py
"""Inspect command - show database statistics and source breakdown.

This module provides the inspect logic that both CLI and TUI use.
"""

from __future__ import annotations

import os
from pathlib import Path

from isotope.commands.base import SourceInfo, StatusResult
from isotope.config import DEFAULT_DATA_DIR, get_stores, load_config


def inspect(
    data_dir: str | None = None,
    config_path: str | Path | None = None,
    sources: bool = False,
    detailed: bool = False,
) -> StatusResult:
    """Get database statistics.

    Args:
        data_dir: Override data directory
        config_path: Override config file path
        sources: If True, include per-source chunk counts
        detailed: If True, include per-source breakdown with atoms and questions

    Returns:
        StatusResult with database statistics
    """
    config = load_config(config_path)
    effective_data_dir = data_dir or config.get("data_dir") or DEFAULT_DATA_DIR

    if not os.path.exists(effective_data_dir):
        return StatusResult(
            success=True,
            total_sources=0,
            total_chunks=0,
            total_atoms=0,
            total_questions=0,
        )

    try:
        stores = get_stores(effective_data_dir)
    except Exception as e:
        return StatusResult(
            success=False,
            error=f"Failed to access database: {e}",
        )

    # Get counts
    chunk_store = stores["chunk_store"]
    atom_store = stores["atom_store"]
    question_store = stores["embedded_question_store"]

    source_names = chunk_store.list_sources()
    total_sources = len(source_names)
    total_chunks = chunk_store.count_chunks()
    total_atoms = atom_store.count_atoms()
    total_questions = question_store.count_questions()

    result = StatusResult(
        success=True,
        total_sources=total_sources,
        total_chunks=total_chunks,
        total_atoms=total_atoms,
        total_questions=total_questions,
    )

    # Add per-source breakdown if sources or detailed
    if sources or detailed:
        for source in sorted(source_names):
            chunk_ids = chunk_store.get_chunk_ids_by_source(source)
            chunk_count = len(chunk_ids)

            if detailed:
                atom_count = atom_store.count_by_chunk_ids(chunk_ids)
                question_count = question_store.count_by_chunk_ids(chunk_ids)
            else:
                atom_count = 0
                question_count = 0

            result.sources.append(
                SourceInfo(
                    source=source,
                    chunk_count=chunk_count,
                    atom_count=atom_count,
                    question_count=question_count,
                )
            )

    return result

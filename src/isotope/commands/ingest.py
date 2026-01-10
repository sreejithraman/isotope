# src/isotope/commands/ingest.py
"""Ingest command - index files into the knowledge base.

This module provides the core ingest logic that both CLI and TUI use.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from isotope.commands.base import (
    CommandStage,
    FileIngestResult,
    IngestResult,
    ProgressCallback,
    ProgressUpdate,
)
from isotope.config import (
    ConfigError,
    create_isotope,
    get_isotope_config,
)

if TYPE_CHECKING:
    from isotope.isotope import Isotope


# Map internal stage names to CommandStage
STAGE_MAP = {
    "storing": CommandStage.STORING,
    "atomizing": CommandStage.ATOMIZING,
    "generating": CommandStage.GENERATING,
    "embedding": CommandStage.EMBEDDING,
    "filtering": CommandStage.FILTERING,
    "indexing": CommandStage.INDEXING,
}


def ingest(
    path: str | Path,
    data_dir: str | None = None,
    config_path: str | Path | None = None,
    on_progress: ProgressCallback | None = None,
    on_file_start: Callable | None = None,
    on_file_complete: Callable | None = None,
) -> IngestResult:
    """Ingest files or directories into the knowledge base.

    This is the core ingest function that both CLI and TUI call.
    It handles file discovery, ingestion, and result aggregation.

    Args:
        path: File or directory to ingest
        data_dir: Override data directory (uses config if not provided)
        config_path: Override config file path
        on_progress: Callback for progress updates during ingestion
        on_file_start: Callback when starting a file (receives filepath, file_index, total_files)
        on_file_complete: Callback when a file is done (receives FileIngestResult)

    Returns:
        IngestResult with aggregated statistics and per-file results
    """
    from isotope.loaders import LoaderRegistry

    path = Path(path)

    # Check path exists
    if not path.exists():
        return IngestResult(
            success=False,
            error=f"Path not found: {path}",
        )

    # Get configuration
    config = get_isotope_config(data_dir, config_path)
    if isinstance(config, ConfigError):
        return IngestResult(
            success=False,
            error=config.message,
        )

    # Create Isotope instance
    try:
        iso = create_isotope(config)
    except Exception as e:
        return IngestResult(
            success=False,
            error=f"Failed to create Isotope: {e}",
        )

    # Find files to ingest
    registry = LoaderRegistry.default()

    if path.is_file():
        files = [str(path)]
    else:
        # Directory - find all supported files
        files = []
        for root, _, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(root, filename)
                if registry.find_loader(filepath):
                    files.append(filepath)

    if not files:
        return IngestResult(
            success=True,
            error="No supported files found",
        )

    # Ingest each file
    result = IngestResult(success=True)

    for i, filepath in enumerate(files):
        if on_file_start:
            on_file_start(filepath, i, len(files))

        file_result = _ingest_file(
            iso=iso,
            filepath=filepath,
            on_progress=on_progress,
        )

        result.file_results.append(file_result)

        if file_result.skipped:
            result.files_skipped += 1
        else:
            result.files_processed += 1
            result.total_chunks += file_result.chunks
            result.total_atoms += file_result.atoms
            result.total_questions += file_result.questions
            result.total_questions_filtered += file_result.questions_filtered

        if on_file_complete:
            on_file_complete(file_result)

    # Check for failures
    if result.errors:
        result.files_failed = len(result.errors)
        if result.files_processed == 0:
            result.success = False

    return result


def _ingest_file(
    iso: Isotope,
    filepath: str,
    on_progress: ProgressCallback | None = None,
) -> FileIngestResult:
    """Ingest a single file.

    Args:
        iso: Isotope instance to use
        filepath: Path to the file
        on_progress: Optional progress callback

    Returns:
        FileIngestResult with stats for this file
    """

    def progress_adapter(event: str, current: int, total: int, message: str) -> None:
        """Adapt Isotope's progress callback to our ProgressUpdate format."""
        if on_progress:
            stage = STAGE_MAP.get(event, CommandStage.PROCESSING)
            on_progress(
                ProgressUpdate(
                    stage=stage,
                    current=current,
                    total=total,
                    message=message,
                )
            )

    try:
        result = iso.ingest_file(
            filepath,
            on_progress=progress_adapter if on_progress else None,
        )

        if result.get("skipped"):
            return FileIngestResult(
                filepath=filepath,
                skipped=True,
                reason=result.get("reason", "unchanged"),
            )

        return FileIngestResult(
            filepath=filepath,
            skipped=False,
            chunks=result.get("chunks", 0),
            atoms=result.get("atoms", 0),
            questions=result.get("questions", 0),
            questions_filtered=result.get("questions_filtered", 0),
        )

    except (OSError, ValueError, RuntimeError) as e:
        return FileIngestResult(
            filepath=filepath,
            skipped=False,
            reason=f"Error: {type(e).__name__}: {e}",
        )


def ingest_with_isotope(
    iso: Isotope,
    path: str | Path,
    on_progress: ProgressCallback | None = None,
    on_file_start: Callable | None = None,
    on_file_complete: Callable | None = None,
) -> IngestResult:
    """Ingest files using an existing Isotope instance.

    This is useful when you already have an Isotope instance configured
    and don't want to create a new one.

    Args:
        iso: Existing Isotope instance
        path: File or directory to ingest
        on_progress: Callback for progress updates
        on_file_start: Callback when starting a file
        on_file_complete: Callback when a file is done

    Returns:
        IngestResult with aggregated statistics
    """
    from isotope.loaders import LoaderRegistry

    path = Path(path)

    if not path.exists():
        return IngestResult(
            success=False,
            error=f"Path not found: {path}",
        )

    # Find files to ingest
    registry = LoaderRegistry.default()

    if path.is_file():
        files = [str(path)]
    else:
        files = []
        for root, _, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(root, filename)
                if registry.find_loader(filepath):
                    files.append(filepath)

    if not files:
        return IngestResult(
            success=True,
            error="No supported files found",
        )

    result = IngestResult(success=True)

    for i, filepath in enumerate(files):
        if on_file_start:
            on_file_start(filepath, i, len(files))

        file_result = _ingest_file(
            iso=iso,
            filepath=filepath,
            on_progress=on_progress,
        )

        result.file_results.append(file_result)

        if file_result.skipped:
            result.files_skipped += 1
        else:
            result.files_processed += 1
            result.total_chunks += file_result.chunks
            result.total_atoms += file_result.atoms
            result.total_questions += file_result.questions
            result.total_questions_filtered += file_result.questions_filtered

        if on_file_complete:
            on_file_complete(file_result)

    if result.errors:
        result.files_failed = len(result.errors)
        if result.files_processed == 0:
            result.success = False

    return result

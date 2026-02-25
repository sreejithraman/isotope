# src/isotope/commands/ingest.py
"""Ingest command - index files into the knowledge base.

This module provides the core ingest logic that both CLI and TUI use.
"""

from __future__ import annotations

import logging
import os
import traceback
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
from isotope.providers.litellm import LLMError

if TYPE_CHECKING:
    from isotope.isotope import Isotope

logger = logging.getLogger("isotope.commands.ingest")


# Map internal stage names to CommandStage
STAGE_MAP = {
    "storing": CommandStage.STORING,
    "atomizing": CommandStage.ATOMIZING,
    "generating": CommandStage.GENERATING,
    "embedding": CommandStage.EMBEDDING,
    "filtering": CommandStage.FILTERING,
    "indexing": CommandStage.INDEXING,
}


def _do_ingest(
    iso: Isotope,
    path: Path,
    on_progress: ProgressCallback | None = None,
    on_file_start: Callable | None = None,
    on_file_complete: Callable | None = None,
    force: bool = False,
) -> IngestResult:
    """Core ingest logic shared between ingest() and ingest_with_isotope()."""
    from isotope.loaders import LoaderRegistry

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
        return IngestResult(success=True, error="No supported files found")

    result = IngestResult(success=True)

    for i, filepath in enumerate(files):
        logger.info("Ingesting %s (%d/%d)", filepath, i + 1, len(files))
        if on_file_start:
            on_file_start(filepath, i, len(files))

        file_result = _ingest_file(iso=iso, filepath=filepath, on_progress=on_progress, force=force)
        result.file_results.append(file_result)

        if file_result.failed:
            result.files_failed += 1
            result.errors.append((filepath, file_result.reason or "Unknown error"))
            logger.warning("Failed %s: %s", filepath, file_result.reason)
        elif file_result.skipped:
            result.files_skipped += 1
            logger.info("Skipped %s: %s", filepath, file_result.reason)
        else:
            result.files_processed += 1
            result.total_chunks += file_result.chunks
            result.total_atoms += file_result.atoms
            result.total_questions += file_result.questions
            result.total_questions_filtered += file_result.questions_filtered
            logger.info(
                "Completed %s: %d chunks, %d atoms, %d questions",
                filepath,
                file_result.chunks,
                file_result.atoms,
                file_result.questions,
            )
            if file_result.questions == 0 and file_result.atoms > 0:
                logger.warning("0 questions for %s (%d atoms)", filepath, file_result.atoms)

        if on_file_complete:
            on_file_complete(file_result)

    if result.files_failed > 0 and result.files_processed == 0:
        result.success = False
        first_error = result.errors[0][1] if result.errors else "All files failed"
        result.error = first_error
        if len(result.errors) > 1:
            details_lines = [f"  {filepath}: {reason}" for filepath, reason in result.errors]
            result.error_details = f"All {result.files_failed} files failed:\n" + "\n".join(
                details_lines
            )

    return result


def ingest(
    path: str | Path,
    data_dir: str | None = None,
    config_path: str | Path | None = None,
    on_progress: ProgressCallback | None = None,
    on_file_start: Callable | None = None,
    on_file_complete: Callable | None = None,
    force: bool = False,
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
        force: If True, re-ingest even if content hash is unchanged

    Returns:
        IngestResult with aggregated statistics and per-file results
    """
    path = Path(path)

    if not path.exists():
        return IngestResult(success=False, error=f"Path not found: {path}")

    config = get_isotope_config(data_dir, config_path)
    if isinstance(config, ConfigError):
        return IngestResult(success=False, error=config.message)

    try:
        iso = create_isotope(config)
    except LLMError as e:
        return IngestResult(
            success=False,
            error=f"Failed to create Isotope: {e.message}",
            error_details=e.details,
        )
    except Exception as e:
        details = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        return IngestResult(
            success=False,
            error=f"Failed to create Isotope: {e}",
            error_details=details,
        )

    return _do_ingest(iso, path, on_progress, on_file_start, on_file_complete, force=force)


def _ingest_file(
    iso: Isotope,
    filepath: str,
    on_progress: ProgressCallback | None = None,
    force: bool = False,
) -> FileIngestResult:
    """Ingest a single file.

    Args:
        iso: Isotope instance to use
        filepath: Path to the file
        on_progress: Optional progress callback
        force: If True, re-ingest even if content hash is unchanged

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
            force=force,
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

    except LLMError as e:
        return FileIngestResult(
            filepath=filepath,
            skipped=False,
            failed=True,
            reason=e.message,
        )
    except Exception as e:
        return FileIngestResult(
            filepath=filepath,
            skipped=False,
            failed=True,
            reason=f"Error: {type(e).__name__}: {e}",
        )


async def _aingest_file(
    iso: Isotope,
    filepath: str,
    on_progress: ProgressCallback | None = None,
    force: bool = False,
) -> FileIngestResult:
    """Ingest a single file asynchronously.

    Args:
        iso: Isotope instance to use
        filepath: Path to the file
        on_progress: Optional progress callback
        force: If True, re-ingest even if content hash is unchanged

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
        result = await iso.aingest_file(
            filepath,
            force=force,
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

    except LLMError as e:
        return FileIngestResult(
            filepath=filepath,
            skipped=False,
            failed=True,
            reason=e.message,
        )
    except Exception as e:
        return FileIngestResult(
            filepath=filepath,
            skipped=False,
            failed=True,
            reason=f"Error: {type(e).__name__}: {e}",
        )


async def _do_ingest_async(
    iso: Isotope,
    path: Path,
    on_progress: ProgressCallback | None = None,
    on_file_start: Callable | None = None,
    on_file_complete: Callable | None = None,
    force: bool = False,
) -> IngestResult:
    """Core async ingest logic."""
    from isotope.loaders import LoaderRegistry

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
        return IngestResult(success=True, error="No supported files found")

    result = IngestResult(success=True)

    for i, filepath in enumerate(files):
        logger.info("Ingesting %s (%d/%d)", filepath, i + 1, len(files))
        if on_file_start:
            on_file_start(filepath, i, len(files))

        file_result = await _aingest_file(
            iso=iso, filepath=filepath, on_progress=on_progress, force=force
        )
        result.file_results.append(file_result)

        if file_result.failed:
            result.files_failed += 1
            result.errors.append((filepath, file_result.reason or "Unknown error"))
            logger.warning("Failed %s: %s", filepath, file_result.reason)
        elif file_result.skipped:
            result.files_skipped += 1
            logger.info("Skipped %s: %s", filepath, file_result.reason)
        else:
            result.files_processed += 1
            result.total_chunks += file_result.chunks
            result.total_atoms += file_result.atoms
            result.total_questions += file_result.questions
            result.total_questions_filtered += file_result.questions_filtered
            logger.info(
                "Completed %s: %d chunks, %d atoms, %d questions",
                filepath,
                file_result.chunks,
                file_result.atoms,
                file_result.questions,
            )
            if file_result.questions == 0 and file_result.atoms > 0:
                logger.warning("0 questions for %s (%d atoms)", filepath, file_result.atoms)

        if on_file_complete:
            on_file_complete(file_result)

    if result.files_failed > 0 and result.files_processed == 0:
        result.success = False
        first_error = result.errors[0][1] if result.errors else "All files failed"
        result.error = first_error
        if len(result.errors) > 1:
            details_lines = [f"  {filepath}: {reason}" for filepath, reason in result.errors]
            result.error_details = f"All {result.files_failed} files failed:\n" + "\n".join(
                details_lines
            )

    return result


async def aingest(
    path: str | Path,
    data_dir: str | None = None,
    config_path: str | Path | None = None,
    on_progress: ProgressCallback | None = None,
    on_file_start: Callable | None = None,
    on_file_complete: Callable | None = None,
    force: bool = False,
) -> IngestResult:
    """Ingest files or directories asynchronously.

    Async version of ingest() that uses a single event loop for all files,
    avoiding the event loop mismatch issues with litellm's LoggingWorker.

    Args:
        path: File or directory to ingest
        data_dir: Override data directory (uses config if not provided)
        config_path: Override config file path
        on_progress: Callback for progress updates during ingestion
        on_file_start: Callback when starting a file (receives filepath, file_index, total_files)
        on_file_complete: Callback when a file is done (receives FileIngestResult)
        force: If True, re-ingest even if content hash is unchanged

    Returns:
        IngestResult with aggregated statistics and per-file results
    """
    path = Path(path)

    if not path.exists():
        return IngestResult(success=False, error=f"Path not found: {path}")

    config = get_isotope_config(data_dir, config_path)
    if isinstance(config, ConfigError):
        return IngestResult(success=False, error=config.message)

    try:
        iso = create_isotope(config)
    except LLMError as e:
        return IngestResult(
            success=False,
            error=f"Failed to create Isotope: {e.message}",
            error_details=e.details,
        )
    except Exception as e:
        details = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        return IngestResult(
            success=False,
            error=f"Failed to create Isotope: {e}",
            error_details=details,
        )

    return await _do_ingest_async(
        iso, path, on_progress, on_file_start, on_file_complete, force=force
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
    path = Path(path)

    if not path.exists():
        return IngestResult(success=False, error=f"Path not found: {path}")

    return _do_ingest(iso, path, on_progress, on_file_start, on_file_complete)

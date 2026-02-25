# src/isotope/commands/questions.py
"""Questions command - show a sample of indexed questions.

This module provides the questions logic that both CLI and TUI use.
"""

from __future__ import annotations

import os
from pathlib import Path

from isotope.commands.base import QuestionInfo, QuestionsResult
from isotope.config import DEFAULT_DATA_DIR, get_stores, load_config


def sample_questions(
    n: int = 5,
    source: str | None = None,
    data_dir: str | None = None,
    config_path: str | Path | None = None,
) -> QuestionsResult:
    """Get a sample of indexed questions.

    Args:
        n: Number of questions to sample
        source: Filter by source file
        data_dir: Override data directory
        config_path: Override config file path

    Returns:
        QuestionsResult with sampled questions and total count
    """
    config = load_config(config_path)
    effective_data_dir = data_dir or config.get("data_dir") or DEFAULT_DATA_DIR

    if not os.path.exists(effective_data_dir):
        return QuestionsResult(success=True, questions=[], total=0)

    try:
        stores = get_stores(effective_data_dir)
    except Exception as e:
        return QuestionsResult(
            success=False,
            error=f"Failed to access database: {e}",
        )

    chunk_ids: list[str] | None = None
    if source:
        chunk_ids = stores["chunk_store"].get_chunk_ids_by_source(source)
        if not chunk_ids:
            return QuestionsResult(success=True, questions=[], total=0)

    questions = stores["embedded_question_store"].sample(n=n, chunk_ids=chunk_ids)

    if chunk_ids:
        total = stores["embedded_question_store"].count_by_chunk_ids(chunk_ids)
    else:
        total = stores["embedded_question_store"].count_questions()

    return QuestionsResult(
        success=True,
        questions=[QuestionInfo(text=q.text, chunk_id=q.chunk_id) for q in questions],
        total=total,
    )

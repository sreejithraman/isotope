# src/isotope/commands/query.py
"""Query command - search the knowledge base.

This module provides the core query logic that both CLI and TUI use.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from isotope.commands.base import QueryResult, SearchResult

if TYPE_CHECKING:
    from isotope import Isotope
from isotope.config import (
    DEFAULT_DATA_DIR,
    ConfigError,
    create_isotope,
    get_isotope_config,
    load_config,
)


def query(
    question: str,
    data_dir: str | None = None,
    config_path: str | Path | None = None,
    k: int | None = None,
    raw: bool = False,
    show_matched_questions: bool = False,
) -> QueryResult:
    """Query the knowledge base with a question.

    This is the core query function that both CLI and TUI call.

    Args:
        question: The question to ask
        data_dir: Override data directory
        config_path: Override config file path
        k: Number of results to return (None for default)
        raw: If True, return raw results without LLM synthesis
        show_matched_questions: If True, include matched questions in results

    Returns:
        QueryResult with answer and sources
    """
    # Load config to get data_dir
    config = load_config(config_path)
    effective_data_dir = data_dir or config.get("data_dir") or DEFAULT_DATA_DIR

    # Check data dir exists
    if not os.path.exists(effective_data_dir):
        return QueryResult(
            success=False,
            query=question,
            error=f"Data directory not found: {effective_data_dir}. Run 'isotope ingest' first.",
        )

    # Get Isotope config
    iso_config = get_isotope_config(data_dir, config_path)
    if isinstance(iso_config, ConfigError):
        return QueryResult(
            success=False,
            query=question,
            error=iso_config.message,
        )

    # Create Isotope instance
    try:
        iso = create_isotope(iso_config)
    except Exception as e:
        return QueryResult(
            success=False,
            query=question,
            error=f"Failed to create Isotope: {e}",
        )

    # Build LLM client for synthesis (if not raw mode)
    llm_client = None
    if not raw and iso_config.llm_model:
        from isotope.providers.litellm import LiteLLMClient

        llm_client = LiteLLMClient(model=iso_config.llm_model, api_key=iso_config.llm_api_key)

    # Create retriever and query
    retriever = iso.retriever(
        default_k=k,
        llm_client=llm_client,
    )

    try:
        response = retriever.get_answer(question)
    except Exception as e:
        return QueryResult(
            success=False,
            query=question,
            error=f"Query failed: {e}",
        )

    if not response.results:
        return QueryResult(
            success=True,
            query=question,
            answer=None,
            results=[],
        )

    # Convert to our result format
    results = []
    for r in response.results:
        results.append(
            SearchResult(
                source=r.chunk.source,
                content=r.chunk.content,
                score=r.score,
                matched_question=r.question.text if show_matched_questions and r.question else None,
                chunk_id=r.chunk.id,
            )
        )

    return QueryResult(
        success=True,
        query=question,
        answer=response.answer,
        results=results,
    )


def query_with_isotope(
    iso: Isotope,
    question: str,
    k: int | None = None,
    raw: bool = False,
    llm_client: Any = None,
) -> QueryResult:
    """Query using an existing Isotope instance.

    This is useful when you already have an Isotope instance configured.

    Args:
        iso: Existing Isotope instance
        question: The question to ask
        k: Number of results to return
        raw: If True, don't use LLM synthesis
        llm_client: Optional LLM client for synthesis

    Returns:
        QueryResult with answer and sources
    """
    # Create retriever and query
    retriever = iso.retriever(
        default_k=k,
        llm_client=None if raw else llm_client,
    )

    try:
        response = retriever.get_answer(question)
    except Exception as e:
        return QueryResult(
            success=False,
            query=question,
            error=f"Query failed: {e}",
        )

    if not response.results:
        return QueryResult(
            success=True,
            query=question,
            answer=None,
            results=[],
        )

    # Convert to our result format
    results = []
    for r in response.results:
        results.append(
            SearchResult(
                source=r.chunk.source,
                content=r.chunk.content,
                score=r.score,
                matched_question=r.question.text if r.question else None,
                chunk_id=r.chunk.id,
            )
        )

    return QueryResult(
        success=True,
        query=question,
        answer=response.answer,
        results=results,
    )

# src/isotope/commands/base.py
"""Base types for the commands layer.

This module defines the data structures used by all commands:
- Progress callbacks for long-running operations
- Prompt callbacks for interactive commands (like init)
- Result types for each command
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum


class CommandStage(Enum):
    """Stages of command execution for progress reporting."""

    # Ingest stages
    STORING = "Storing"
    ATOMIZING = "Atomizing"
    GENERATING = "Generating"
    EMBEDDING = "Embedding"
    FILTERING = "Filtering"
    INDEXING = "Indexing"

    # General stages
    LOADING = "Loading"
    PROCESSING = "Processing"
    COMPLETE = "Complete"


@dataclass
class ProgressUpdate:
    """Progress update for long-running operations.

    Attributes:
        stage: Current stage of the operation
        current: Current item number (1-indexed)
        total: Total number of items (0 for indeterminate)
        message: Optional status message
    """

    stage: CommandStage
    current: int
    total: int
    message: str | None = None

    @property
    def is_indeterminate(self) -> bool:
        """True if progress is indeterminate (total unknown)."""
        return self.total == 0

    @property
    def percentage(self) -> int:
        """Progress as percentage (0-100). Returns 0 if indeterminate."""
        if self.total == 0:
            return 0
        return int(100 * self.current / self.total)


# Callback type for progress updates
ProgressCallback = Callable[[ProgressUpdate], None]


class InitPrompt(Enum):
    """Types of prompts during init command."""

    LLM_MODEL = "llm_model"
    EMBEDDING_MODEL = "embedding_model"
    API_KEY_LLM = "api_key_llm"
    API_KEY_EMBEDDING = "api_key_embedding"
    API_KEY_SAME = "api_key_same"  # "Same as LLM" / "Different" / "None"
    RATE_LIMIT = "rate_limit"
    PRIORITY = "priority"
    OVERWRITE_CONFIG = "overwrite_config"


@dataclass
class PromptRequest:
    """Request for user input during interactive commands.

    Attributes:
        prompt_type: Type of prompt (for handling by UI)
        message: The question to display to the user
        choices: Available choices (None for free text input)
        default: Default value if user provides no input
        is_secret: True for sensitive input (API keys)
    """

    prompt_type: InitPrompt
    message: str
    choices: list[str] | None = None
    default: str | None = None
    is_secret: bool = False


# Callback type for interactive prompts - returns user's response
PromptCallback = Callable[[PromptRequest], str]


@dataclass
class CommandResult:
    """Base result type for commands."""

    success: bool
    error: str | None = None


@dataclass
class FileIngestResult:
    """Result for a single file ingestion."""

    filepath: str
    skipped: bool
    reason: str | None = None  # Reason if skipped
    chunks: int = 0
    atoms: int = 0
    questions: int = 0
    questions_filtered: int = 0


@dataclass
class IngestResult(CommandResult):
    """Result of the ingest command.

    Attributes:
        files_processed: Number of files successfully processed
        files_skipped: Number of files skipped (unchanged)
        files_failed: Number of files that failed
        total_chunks: Total chunks created
        total_atoms: Total atoms created
        total_questions: Total questions indexed
        total_questions_filtered: Questions filtered by diversity
        file_results: Per-file results
        errors: List of (filepath, error_message) for failed files
    """

    files_processed: int = 0
    files_skipped: int = 0
    files_failed: int = 0
    total_chunks: int = 0
    total_atoms: int = 0
    total_questions: int = 0
    total_questions_filtered: int = 0
    file_results: list[FileIngestResult] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class SearchResult:
    """A single search result."""

    source: str
    content: str
    score: float
    matched_question: str | None = None
    chunk_id: str | None = None


@dataclass
class QueryResult(CommandResult):
    """Result of the query command.

    Attributes:
        answer: Synthesized answer (None if raw mode)
        results: List of search results with sources
        query: The original query
    """

    query: str = ""
    answer: str | None = None
    results: list[SearchResult] = field(default_factory=list)


@dataclass
class SourceInfo:
    """Information about an indexed source."""

    source: str
    chunk_count: int
    atom_count: int = 0
    question_count: int = 0


@dataclass
class StatusResult(CommandResult):
    """Result of the status command.

    Attributes:
        total_sources: Number of indexed sources
        total_chunks: Total chunks in database
        total_atoms: Total atoms in database
        total_questions: Total questions indexed
        sources: Per-source breakdown (if detailed)
    """

    total_sources: int = 0
    total_chunks: int = 0
    total_atoms: int = 0
    total_questions: int = 0
    sources: list[SourceInfo] = field(default_factory=list)


@dataclass
class ListResult(CommandResult):
    """Result of the list command.

    Attributes:
        sources: List of source information
    """

    sources: list[SourceInfo] = field(default_factory=list)


@dataclass
class DeleteResult(CommandResult):
    """Result of the delete command.

    Attributes:
        source: The source that was deleted
        chunks_deleted: Number of chunks deleted
        atoms_deleted: Number of atoms deleted
        questions_deleted: Number of questions deleted
    """

    source: str = ""
    chunks_deleted: int = 0
    atoms_deleted: int = 0
    questions_deleted: int = 0


@dataclass
class SettingInfo:
    """Information about a single setting."""

    name: str
    value: str
    source: str  # "env var", "yaml", "default"


@dataclass
class ConfigResult(CommandResult):
    """Result of the config command.

    Attributes:
        provider: Provider type (litellm, custom)
        llm_model: LLM model name
        embedding_model: Embedding model name
        data_dir: Data directory path
        settings: List of behavioral settings with sources
        config_path: Path to config file (if found)
    """

    provider: str = "litellm"
    llm_model: str | None = None
    embedding_model: str | None = None
    data_dir: str = ""
    settings: list[SettingInfo] = field(default_factory=list)
    config_path: str | None = None


@dataclass
class InitResult(CommandResult):
    """Result of the init command.

    Attributes:
        config_path: Path to created config file
        env_path: Path to .env file (if API keys saved)
        provider: Provider type
        llm_model: LLM model configured
        embedding_model: Embedding model configured
    """

    config_path: str = ""
    env_path: str | None = None
    provider: str = "litellm"
    llm_model: str = ""
    embedding_model: str = ""

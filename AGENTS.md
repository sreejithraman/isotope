# AI Agent Guidelines for Isotope

## Versioning

This project is pre-1.0 (`v0.x`). Breaking changes to APIs, storage formats, and configuration are expected and acceptable. Prioritize clean, intuitive design for new users over backward compatibility with existing users.

## Project Purpose

Isotope is a **Reverse RAG** library that indexes *questions*, not chunks. Instead of hoping user queries match document chunks, we pre-generate questions each chunk can answer and match query-to-question. This gives tighter semantic alignment. Based on arXiv:2405.12363.

## Commands

```bash
# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_retriever.py

# Run a specific test
uv run pytest tests/test_retriever.py::TestRetrieverGetAnswer::test_get_answer_returns_response_with_synthesis -v

# Format code
uv run ruff format src tests

# Linting (auto-fix)
uv run ruff check --fix src tests

# Type checking
uv run mypy src

# Skip integration tests that mock LLM APIs
uv run pytest -m 'not mock_integration'
```

## Codebase Structure

```
src/isotope/
├── [Core Library]
│   ├── isotope.py          # Main Isotope class
│   ├── ingestor.py         # Ingestion pipeline
│   ├── retriever.py        # Query pipeline
│   ├── settings.py         # Behavioral settings
│   ├── stores/             # Storage layer (Chroma, SQLite)
│   ├── atomizer/           # Chunk → Atoms
│   ├── embedder/           # Text → Embeddings
│   ├── question_generator/ # Atoms → Questions
│   ├── loaders/            # File → Chunks
│   └── models/             # Data models (Chunk, Atom, Question, etc.)
│
├── config.py               # Shared configuration utilities
│
├── commands/               # UI-agnostic command layer
│   ├── base.py            # Result types, callbacks
│   ├── ingest.py          # Ingest command logic
│   ├── query.py           # Query command logic
│   ├── status.py          # Status command logic
│   ├── list.py            # List command logic
│   ├── delete.py          # Delete command logic
│   ├── config_cmd.py      # Config command logic
│   └── init.py            # Init command logic
│
└── cli/                    # Typer CLI (thin wrapper)
    ├── __init__.py
    ├── __main__.py        # python -m isotope.cli
    └── app.py             # Typer commands → Rich rendering
```

**Data flow**: Document → Chunks → Atoms → Questions → Embeddings → Index

**Pipeline classes**: `Isotope` is the facade; it creates `Ingestor` (for ingestion) and `Retriever` (for querying).

## Commands Layer

The `commands/` layer contains UI-agnostic business logic. Commands return structured result objects; UIs render them.

```python
# commands/status.py - Returns data
def status(data_dir=None, detailed=False) -> StatusResult:
    stores = get_stores(data_dir)
    return StatusResult(
        success=True,
        total_sources=len(stores["chunk_store"].list_sources()),
        total_chunks=stores["chunk_store"].count_chunks(),
    )

# cli/app.py - Renders with Rich
result = status.status(data_dir=data_dir, detailed=detailed)
table = Table(title="Database Status")
table.add_row("Sources", str(result.total_sources))
console.print(table)
```

Long-running operations use callbacks for progress updates (`ProgressCallback` in `commands/base.py`).

### Result Types

All commands return dataclass results from `commands/base.py`:

| Command | Result Type | Key Fields |
|---------|-------------|------------|
| `ingest` | `IngestResult` | `files_processed`, `total_questions`, `file_results` |
| `query` | `QueryResult` | `answer`, `results`, `query` |
| `status` | `StatusResult` | `total_sources`, `total_chunks`, `total_questions` |
| `list` | `ListResult` | `sources` (list of `SourceInfo`) |
| `delete` | `DeleteResult` | `chunks_deleted`, `source` |
| `config` | `ConfigResult` | `settings`, `provider`, `config_path` |
| `init` | `InitResult` | `config_path`, `llm_model`, `embedding_model` |

All results have `success: bool` and `error: str | None`.

### Adding a New Command

1. **Create result type** in `commands/base.py`:
   ```python
   @dataclass
   class MyCommandResult(CommandResult):
       my_field: str = ""
   ```

2. **Implement command** in `commands/my_cmd.py`:
   ```python
   def my_command(arg1, arg2=None) -> MyCommandResult:
       return MyCommandResult(success=True, my_field="value")
   ```

3. **Export** from `commands/__init__.py`

4. **Add CLI command** in `cli/app.py`:
   ```python
   @app.command()
   def my_cmd_cli(arg1: str, arg2: str = None):
       result = my_cmd.my_command(arg1, arg2)
       # Render with Rich
   ```

5. **Add tests** in `tests/commands/test_my_cmd.py`

### File Locations Quick Reference

| What | Where |
|------|-------|
| Command result types | `src/isotope/commands/base.py` |
| Command implementations | `src/isotope/commands/*.py` |
| CLI commands | `src/isotope/cli/app.py` |
| Config utilities | `src/isotope/config.py` |
| CLI tests | `tests/test_cli.py` |
| Command tests | `tests/commands/test_*.py` |

## Code Patterns

**ABCs for extensibility**: All major components have abstract base classes:
- `EmbeddedQuestionStore`, `ChunkStore`, `AtomStore`, `SourceRegistry` in `stores/base.py`
- `Atomizer` in `atomizer/base.py`
- `Embedder` in `embedder/base.py`
- `QuestionGenerator` in `question_generator/base.py`
- `LLMClient`, `EmbeddingClient` in `providers/base.py`
- `Loader` in `loaders/base.py`

**Pydantic 2.x models**: All data models use Pydantic with:
- `Field(default_factory=...)` for UUIDs
- Composition over inheritance (e.g., `EmbeddedQuestion` contains `Question`)

**LiteLLM**: Default provider clients live in `providers/litellm` for easy provider switching.

**Optional dependencies**: Use `_optional.py` pattern to handle missing optional packages gracefully.

**pytest**: Tests mirror src structure in `tests/`. Use `pytest` to run all tests.

## Extension Patterns

| To add... | Implement... | Reference |
|-----------|--------------|-----------|
| New embedded question store | `EmbeddedQuestionStore` ABC | `ChromaEmbeddedQuestionStore` |
| New chunk store | `ChunkStore` ABC | `SQLiteChunkStore` |
| New atom store | `AtomStore` ABC | `SQLiteAtomStore` |
| New atomizer | `Atomizer` ABC | `SentenceAtomizer`, `LLMAtomizer` |
| New embedder | `Embedder` ABC | `ClientEmbedder` |
| New question generator | `QuestionGenerator` ABC | `ClientQuestionGenerator` |
| New file loader | `Loader` ABC | `TextLoader`, register via `LoaderRegistry` |

## CLI Commands

```bash
uv run isotope init                  # Create an isotope.yaml config file
uv run isotope ingest <path>         # Ingest file or directory
uv run isotope query "<question>"    # Query with LLM synthesis (--raw for no synthesis)
uv run isotope inspect               # Show database statistics
uv run isotope inspect --sources     # Show per-source breakdown
uv run isotope questions             # Show sample of indexed questions
uv run isotope delete <source>       # Delete a source from the database
uv run isotope config                # Show current configuration
```

## Key Files to Read First

1. `src/isotope/models/` - Data structures (start here)
2. `src/isotope/stores/base.py` - Storage ABCs
3. `src/isotope/isotope.py` - Central facade
4. `src/isotope/settings.py` - All configuration options
5. `README.md` - Concept overview and limitations

## Before PR

### Pre-commit hooks (recommended)

Set up once, runs ruff format + lint on every commit:

```bash
make dev-setup   # or: uv sync --extra dev --extra all
```

### CI checks (automatic)

GitHub Actions runs on every PR:
- `ruff format --check` - code formatting
- `ruff check` - linting
- `mypy src` - type checking
- `pytest` - tests on Python 3.11, 3.12, and 3.13

### Manual checks

```bash
uv run ruff format src tests       # Format code
uv run ruff check --fix src tests  # Auto-fix import sorting, etc.
uv run mypy src                    # Must pass with no errors
uv run pytest                      # All tests must pass
```

## Common Tasks

- **Add new setting**: Edit `settings.py`, add field to `Settings` class
- **Update exports**: Edit `src/isotope/__init__.py` and module `__init__.py` files

## Model Names (LiteLLM)

When referencing LLM/embedding models for the LiteLLM provider, use constants from `src/isotope/providers/litellm/models.py`. This file is the **single source of truth** for model names.

**Do NOT hardcode model strings elsewhere in the codebase.** Import from `models.py`:

```python
from isotope.providers.litellm.models import ChatModels, EmbeddingModels

# Use constants
model = ChatModels.GPT_5_MINI
embedding = EmbeddingModels.TEXT_3_SMALL
```

Current models (check `models.py` for the latest):
- **Chat**: `ChatModels.GPT_5_MINI`, `ChatModels.CLAUDE_SONNET_45`, `ChatModels.GEMINI_3_FLASH`
- **Embedding**: `EmbeddingModels.TEXT_3_SMALL`, `EmbeddingModels.GEMINI_EMBEDDING_001`

**For custom providers**: Users specify their own model identifiers via class paths. The `models.py` constants only apply to LiteLLM.

**In documentation**: When showing example model names, use the string values from `models.py` (e.g., `openai/gpt-5-mini-2025-08-07`) to stay consistent.

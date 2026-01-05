# AI Agent Guidelines for Isotope

## Project Purpose

Isotope is a **Reverse RAG** library that indexes *questions*, not chunks. Instead of hoping user queries match document chunks, we pre-generate questions each chunk can answer and match query-to-question. This gives tighter semantic alignment. Based on arXiv:2405.12363.

## Commands

```bash
# Run all tests
pytest

# Run a single test file
pytest tests/test_retriever.py

# Run a specific test
pytest tests/test_retriever.py::TestRetrieverGetAnswer::test_get_answer_returns_response_with_synthesis -v

# Format code
ruff format src tests

# Linting (auto-fix)
ruff check --fix src tests

# Type checking
mypy src

# Skip integration tests that mock LLM APIs
pytest -m 'not mock_integration'
```

## Codebase Structure

```
src/isotope/
├── models/          # Pydantic data models (Chunk, Atom, Question, etc.)
├── stores/          # Storage ABCs + implementations (EmbeddedQuestionStore, ChunkStore, AtomStore, SourceRegistry)
├── atomizer/        # Break chunks into atomic facts (SentenceAtomizer, LLMAtomizer)
├── embedder/        # Embedding wrapper (ClientEmbedder)
├── configuration/   # Provider/storage configuration objects (LiteLLMProvider, LocalStorage)
├── question_generator/  # Question generation + diversity filtering
├── loaders/         # File loaders (text, PDF, HTML) with registry pattern
├── providers/       # LLM/embedding provider clients (LiteLLMClient, LiteLLMEmbeddingClient)
├── isotope.py       # Central configuration facade
├── ingestor.py      # Ingestion pipeline
├── retriever.py     # Query pipeline with LLM synthesis
├── cli.py           # Typer CLI (isotope init/config/ingest/query/list/status/delete)
├── settings.py      # Settings (Pydantic BaseModel; library does not read env vars)
└── _optional.py     # Optional dependency handling
```

**Data flow**: Document → Chunks → Atoms → Questions → Embeddings → Index

**Pipeline classes**: `Isotope` is the facade; it creates `Ingestor` (for ingestion) and `Retriever` (for querying).

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
isotope config              # Show current configuration
isotope ingest <path>       # Ingest file or directory
isotope query "<question>"  # Query with LLM synthesis (--raw for no synthesis)
isotope list                # List indexed sources
isotope status              # Show database statistics
isotope delete <source>     # Delete a source from the database
isotope init                # Create an isotope.yaml config file
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
pip install pre-commit
pre-commit install
```

### CI checks (automatic)

GitHub Actions runs on every PR:
- `ruff format --check` - code formatting
- `ruff check` - linting
- `mypy src` - type checking
- `pytest` - tests on Python 3.11 and 3.12

### Manual checks

```bash
ruff format src tests       # Format code
ruff check --fix src tests  # Auto-fix import sorting, etc.
mypy src                    # Must pass with no errors
pytest                      # All tests must pass
```

## Common Tasks

- **Add new setting**: Edit `settings.py`, add field to `Settings` class
- **Update exports**: Edit `src/isotope/__init__.py` and module `__init__.py` files

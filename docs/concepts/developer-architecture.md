# Developer Architecture

This document explains the internal code structure of Isotope for contributors. For user-facing architecture (Isotope class, Ingestor, Retriever), see [Architecture](./architecture.md).

## Code Structure Overview

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
├── cli/                    # Typer CLI (thin wrapper)
│   ├── __init__.py
│   ├── __main__.py        # python -m isotope.cli
│   └── app.py             # Typer commands → Rich rendering
│
└── tui/                    # Textual TUI
    ├── app.py             # Main TUI app
    ├── screens/           # TUI screens
    │   ├── main.py        # Command handling → output widgets
    │   ├── init.py        # Interactive init wizard
    │   └── welcome.py     # Welcome screen
    ├── widgets/           # Custom Textual widgets
    └── commands/          # TUI command parsing
```

## Design Principles

### 1. Commands Return Data, UIs Render

The `commands/` layer contains UI-agnostic business logic. Commands return structured result objects that UIs can render however they want.

```python
# commands/status.py - Returns data
def status(data_dir=None, detailed=False) -> StatusResult:
    stores = get_stores(data_dir)
    return StatusResult(
        success=True,
        total_sources=len(stores["chunk_store"].list_sources()),
        total_chunks=stores["chunk_store"].count_chunks(),
        # ...
    )

# cli/app.py - Renders with Rich
result = status.status(data_dir=data_dir, detailed=detailed)
table = Table(title="Database Status")
table.add_row("Sources", str(result.total_sources))
console.print(table)

# tui/screens/main.py - Renders with Textual
result = status.status(data_dir=self._data_dir, detailed=detailed)
table = Table(title="Database Status")
table.add_row("Sources", str(result.total_sources))
output.write_table(table)
```

### 2. Callbacks for Progress and Interaction

Long-running operations use callbacks for progress updates. Interactive commands (like `init`) use prompt callbacks.

```python
# commands/base.py
@dataclass
class ProgressUpdate:
    stage: CommandStage  # STORING, ATOMIZING, GENERATING, etc.
    current: int
    total: int
    message: str | None

ProgressCallback = Callable[[ProgressUpdate], None]

# commands/ingest.py
def ingest(path, on_progress=None, on_file_start=None, on_file_complete=None):
    for filepath in files:
        if on_file_start:
            on_file_start(filepath, i, len(files))
        # ... ingest file with on_progress callback
        if on_file_complete:
            on_file_complete(file_result)
```

### 3. Shared Configuration

The `config.py` module provides utilities used by both CLI and TUI:

```python
from isotope.config import (
    find_config_file,     # Locate isotope.yaml
    load_config,          # Parse YAML config
    get_stores,           # Get store instances for read operations
    get_isotope_config,   # Get full Isotope configuration
    create_isotope,       # Create Isotope instance from config
    build_settings,       # Build Settings from YAML + env vars
)
```

## Layer Responsibilities

### Core Library (`isotope.py`, `ingestor.py`, `retriever.py`)

The main library API. Users import from here:

```python
from isotope import Isotope, Chunk, LiteLLMProvider, LocalStorage
```

No CLI/TUI awareness. Pure library code.

### Commands Layer (`commands/`)

UI-agnostic command implementations. Each command:

1. Loads configuration
2. Performs the operation
3. Returns a structured result

```python
# Example: commands/query.py
def query(question, data_dir=None, k=None, raw=False) -> QueryResult:
    config = load_config()
    iso_config = get_isotope_config(data_dir)
    iso = create_isotope(iso_config)

    retriever = iso.retriever(default_k=k, llm_client=llm_client)
    response = retriever.get_answer(question)

    return QueryResult(
        success=True,
        query=question,
        answer=response.answer,
        results=[...],
    )
```

### CLI (`cli/`)

Thin Typer wrapper. Each command is ~50-100 lines:

1. Parse CLI arguments
2. Create Rich progress callback (if needed)
3. Call `commands.X()`
4. Render result with Rich

```python
# cli/app.py
@app.command()
def query_cmd(question: str, k: int = 5, raw: bool = False):
    result = query.query(question=question, k=k, raw=raw)

    if not result.success:
        console.print(f"[red]Error:[/red] {result.error}")
        raise typer.Exit(1)

    if result.answer:
        console.print(Panel(Markdown(result.answer), title="Answer"))
    # ... render sources
```

### TUI (`tui/`)

Textual-based interactive interface. The `MainScreen` handles commands:

1. Parse user input with `CommandParser`
2. Call appropriate `_cmd_*` method
3. Method calls `commands.X()` and renders to `OutputDisplay`

```python
# tui/screens/main.py
async def _cmd_query(self, output, header, question, flags):
    result = query.query(
        question=question,
        data_dir=self._data_dir,
        k=flags.get("k"),
        raw=flags.get("raw"),
    )

    if result.answer:
        output.write_markdown(result.answer, title="Answer")
    # ... render sources
```

## Result Types

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

All results have:
- `success: bool` - Whether the command succeeded
- `error: str | None` - Error message if failed

## Adding a New Command

1. **Create result type** in `commands/base.py`:
   ```python
   @dataclass
   class MyCommandResult(CommandResult):
       my_field: str = ""
   ```

2. **Implement command** in `commands/my_cmd.py`:
   ```python
   def my_command(arg1, arg2=None) -> MyCommandResult:
       # Load config, do work, return result
       return MyCommandResult(success=True, my_field="value")
   ```

3. **Export** from `commands/__init__.py`:
   ```python
   from isotope.commands import my_cmd
   ```

4. **Add CLI command** in `cli/app.py`:
   ```python
   @app.command()
   def my_cmd_cli(arg1: str, arg2: str = None):
       result = my_cmd.my_command(arg1, arg2)
       # Render with Rich
   ```

5. **Add TUI handler** in `tui/screens/main.py`:
   ```python
   async def _cmd_my_command(self, output, flags):
       result = my_cmd.my_command(...)
       # Render to output widget
   ```

6. **Add to TUI parser** in `tui/commands/parser.py`:
   ```python
   class CommandType(Enum):
       MY_COMMAND = auto()

   ALIASES = {
       "mycommand": CommandType.MY_COMMAND,
       "mc": CommandType.MY_COMMAND,  # alias
   }
   ```

7. **Add tests** in `tests/commands/test_my_cmd.py`

## Testing

### Unit Tests

Test commands directly without UI:

```python
# tests/commands/test_status.py
def test_status_no_database():
    result = status.status(data_dir="/nonexistent")
    assert result.success is True
    assert result.total_sources == 0
```

### CLI Integration Tests

Test CLI via Typer's `CliRunner`:

```python
# tests/test_cli.py
def test_status_command(runner):
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
```

### TUI Tests

Test parser and widgets:

```python
# tests/tui/test_parser.py
def test_parse_status(parser):
    result = parser.parse("status --detailed")
    assert result.type == CommandType.STATUS
    assert result.flags.get("detailed") is True
```

## File Locations Quick Reference

| What | Where |
|------|-------|
| Command result types | `src/isotope/commands/base.py` |
| Command implementations | `src/isotope/commands/*.py` |
| CLI commands | `src/isotope/cli/app.py` |
| TUI command handlers | `src/isotope/tui/screens/main.py` |
| TUI command parser | `src/isotope/tui/commands/parser.py` |
| Config utilities | `src/isotope/config.py` |
| CLI tests | `tests/test_cli.py` |
| Command tests | `tests/commands/test_*.py` |
| TUI parser tests | `tests/tui/test_parser.py` |

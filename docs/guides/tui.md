# TUI Reference

Isotope includes an interactive Terminal User Interface (TUI) for a conversational experience.

## Installation

The TUI requires both the interface and a provider/storage backend:

```bash
pip install isotope-rag[all]

# Or pick specific extras:
pip install "isotope-rag[tui,litellm,chroma]"
```

## Launching

```bash
isotope-tui
```

On first launch (before any data is indexed), you'll see a welcome screen guiding you through setup. After that, you'll go directly to the main interactive shell.

## Interface Overview

The TUI has four main areas:

```
┌─────────────────────────────────────────────────────────┐
│  isotope                                    [Header]    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  [Output Area - command results, answers, tables]       │
│                                                         │
├─────────────────────────────────────────────────────────┤
│  3 sources | 127 questions | gpt-4o-mini   [Status Bar] │
├─────────────────────────────────────────────────────────┤
│  > _                                       [Input Area] │
└─────────────────────────────────────────────────────────┘
```

- **Header**: Shows operation in progress (e.g., "Ingesting 1/3: guide.md")
- **Output Area**: Displays command results, answers, tables, and progress
- **Status Bar**: Shows database statistics and current model
- **Input Area**: Type commands or questions here

## Commands

The TUI supports most CLI commands. Type `/help` or just `help` to see them:

> **Note:** The `questions sample` command is CLI-only and not available in the TUI.

| Command | Description |
|---------|-------------|
| `init` | Initialize isotope configuration |
| `ingest <path>` | Index a file or directory |
| `query <question>` | Ask a question (or just type it directly) |
| `status` | Show database statistics |
| `list` | List indexed sources |
| `config` | Show current configuration |
| `delete <source>` | Remove a source from the index |
| `clear` | Clear the output area |
| `quit` | Exit isotope |

### Natural Language Queries

You don't need a command prefix for queries. Just type your question:

```
> How does authentication work?
```

This is equivalent to `query How does authentication work?`

### Command Examples

```
> ingest ./docs
> status
> list
> delete /path/to/old-file.md --force
> config
> clear
> quit
```

## Query Options

The `query` command supports these flags:

| Flag | Description |
|------|-------------|
| `--raw`, `-r` | Return raw chunks without LLM synthesis |
| `--k N` | Return N results (default: 5) |
| `--show-matched-questions`, `-q` | Show which generated questions matched |

**Examples:**

```
> query "What is the API rate limit?" --raw
> query authentication -k 10
> query "How do I authenticate?" -q
```

## Status Options

```
> status            # Basic statistics
> status --detailed # Include per-source breakdown
> status -d         # Short form
```

## Delete Options

In the TUI, delete requires the `--force` flag (no interactive confirmation prompt):

```
> delete /path/to/file.md          # Shows warning, requires --force
> delete /path/to/file.md --force  # Performs the delete
> delete /path/to/file.md -f       # Short form
```

> **Note:** Unlike the CLI which has an interactive confirmation prompt, the TUI requires `--force` to confirm deletions.

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Tab` | Auto-complete commands and file paths |
| `Up` / `Down` | Navigate command history |
| `Ctrl+L` | Clear the output area |
| `Ctrl+C` | Quit the application |
| `Enter` | Submit command |

## Tab Completion

The TUI supports tab completion for:

- **Commands**: Type `in` then `Tab` to complete to `ingest`
- **File paths**: Type `ingest ./do` then `Tab` to complete to `./docs/`

## Example Session

```
> init
Created isotope.yaml
Ready! Try: ingest <path>

> ingest ./docs
[1/3] getting-started.md
   ATOMIZING...
   GENERATING...
   EMBEDDING...
   45 chunks, 127 questions
[2/3] api.md
   82 chunks, 234 questions
[3/3] auth.md
   23 chunks, 67 questions

Done! Ingested 3 files -> 428 questions

> How does authentication work?
Searching...

Authentication in Isotope uses JWT tokens stored in httpOnly cookies.
The tokens are validated on each request...

Sources:
  docs/auth.md (0.94)
  docs/api.md (0.87)

> status
┌──────────────────────────────────────┐
│          Database Status             │
├─────────────────────┬────────────────┤
│ Metric              │ Value          │
├─────────────────────┼────────────────┤
│ Sources             │ 3              │
│ Chunks              │ 150            │
│ Atoms               │ 312            │
│ Questions           │ 428            │
└─────────────────────┴────────────────┘

> list
┌──────────────────────────────────────────┐
│     Indexed Sources (3)                  │
├────────────────────────────────┬─────────┤
│ Source                         │ Chunks  │
├────────────────────────────────┼─────────┤
│ /path/to/docs/getting-started.md │ 45    │
│ /path/to/docs/api.md             │ 82    │
│ /path/to/docs/auth.md            │ 23    │
└────────────────────────────────┴─────────┘

> quit
```

## Configuration

The TUI uses the same configuration as the CLI:

1. Looks for `isotope.yaml`, `isotope.yml`, or `.isotoperc` in the current directory or parent directories
2. Falls back to environment variables if no config file is found
3. Use `init` command to create a new configuration interactively

See [Configuration Guide](./configuration.md) for full details.

## Tips

- **Just type questions directly** - no need for the `query` command prefix
- **Use Tab for file paths** - saves typing and avoids typos
- **Use Up/Down for history** - quickly re-run previous commands
- **Use Ctrl+L to clear** - keeps the output area clean
- **Use `--raw` for debugging** - see matched chunks without LLM synthesis
- **Use `-q` to see matched questions** - understand why certain results were returned

## Comparison with CLI

Both interfaces share the same underlying command layer:

| Feature | CLI | TUI |
|---------|-----|-----|
| Core commands | Yes | Yes |
| `questions sample` | Yes | No |
| Delete confirmation | Interactive prompt | Requires `--force` |
| Progress bars | Yes | Yes (inline) |
| Markdown rendering | Basic | Rich |
| Command history | Shell-based | Built-in (Up/Down) |
| Tab completion | Shell-based | Built-in |
| Interactive prompts | Yes | Via `init` screen |
| Scripting | Yes | No |

Use the CLI for scripting and automation. Use the TUI for interactive exploration.

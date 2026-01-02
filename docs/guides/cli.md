# CLI Reference

Isotope includes a command-line interface for quick ingestion and querying workflows.

## Installation

The CLI is included when you install `isotopedb`:

```bash
pip install isotopedb
```

Verify it's working:

```bash
isotope --version
```

## Commands

### `isotope ingest`

Ingest files or directories into the database.

```bash
isotope ingest <path> [options]
```

**Arguments:**
- `path` - File or directory to ingest

**Options:**
- `--data-dir, -d` - Data directory (default: `ISOTOPE_DATA_DIR` or `./isotope_data`)
- `--plain` - Plain text output without colors

**Examples:**

```bash
# Ingest a single file
isotope ingest docs/guide.md

# Ingest a directory (recursively finds .txt and .md files)
isotope ingest docs/

# Use a custom data directory
isotope ingest docs/ --data-dir ./my_data
```

**What happens:**
1. Loads supported files (`.txt`, `.md`, `.markdown`)
2. Breaks content into chunks
3. Atomizes chunks into facts
4. Generates questions for each atom
5. Embeds and indexes questions
6. Shows statistics when complete

**Output:**
```
Ingested 3 chunks
Created 12 atoms
Generated 89 questions
Filtered 15 similar questions
```

---

### `isotope query`

Query the database with a question.

```bash
isotope query "<question>" [options]
```

**Arguments:**
- `question` - Your question (use quotes if it contains spaces)

**Options:**
- `--data-dir, -d` - Data directory
- `--k, -k` - Number of results to return (default: 5)
- `--raw, -r` - Return raw chunks without LLM synthesis
- `--plain` - Plain text output

**Examples:**

```bash
# Ask a question (with LLM-synthesized answer)
isotope query "How do I authenticate?"

# Get more results
isotope query "authentication" -k 10

# Raw mode (no LLM, just matching chunks)
isotope query "authentication" --raw

# Plain output for scripting
isotope query "Who created Python?" --plain
```

**With LLM synthesis (default):**
```
┌─────────────────────────────────────────────────────────┐
│ Answer                                                   │
├─────────────────────────────────────────────────────────┤
│ Python was created by Guido van Rossum and was first   │
│ released in 1991.                                        │
└─────────────────────────────────────────────────────────┘

Sources:
  [1] docs/python.md (score: 0.923)
      Python is a high-level programming language created by Guido...
```

**Raw mode (`--raw`):**
```
Sources:
  [1] docs/python.md (score: 0.923)
      Who created Python?
           → Python is a high-level programming language created by Guido...
```

---

### `isotope status`

Show database statistics.

```bash
isotope status [options]
```

**Options:**
- `--data-dir, -d` - Data directory
- `--plain` - Plain text output

**Example:**

```bash
isotope status
```

**Output:**
```
┌─────────────────────────────────────┐
│         Database Status             │
├─────────────────────┬───────────────┤
│ Metric              │ Value         │
├─────────────────────┼───────────────┤
│ Data directory      │ ./isotope_data│
│ Sources             │ 5             │
│ Chunks              │ 23            │
│ Atoms               │ 89            │
│ Indexed questions   │ 23            │
└─────────────────────┴───────────────┘
```

---

### `isotope list`

List all indexed sources.

```bash
isotope list [options]
```

**Options:**
- `--data-dir, -d` - Data directory
- `--plain` - Plain text output

**Example:**

```bash
isotope list
```

**Output:**
```
┌─────────────────────────────────────┐
│     Indexed Sources (5)             │
├─────────────────────────────┬───────┤
│ Source                      │ Chunks│
├─────────────────────────────┼───────┤
│ docs/api.md                 │ 3     │
│ docs/authentication.md      │ 2     │
│ docs/getting-started.md     │ 5     │
│ docs/quickstart.md          │ 1     │
│ README.md                   │ 12    │
└─────────────────────────────┴───────┘
```

---

### `isotope delete`

Delete a source and all its chunks from the database.

```bash
isotope delete <source> [options]
```

**Arguments:**
- `source` - Source path to delete (as shown in `isotope list`)

**Options:**
- `--data-dir, -d` - Data directory
- `--force, -f` - Skip confirmation prompt
- `--plain` - Plain text output

**Examples:**

```bash
# Delete with confirmation
isotope delete docs/old.md

# Force delete (no confirmation)
isotope delete docs/old.md --force
```

**Output:**
```
About to delete 3 chunks from docs/old.md
Continue? [y/N]: y
Deleted 3 chunks from docs/old.md
```

---

### `isotope config`

Show current configuration settings.

```bash
isotope config
```

**Output:**
```
┌───────────────────────────────────────────────────────┐
│                 Isotope Configuration                  │
├─────────────────────────────┬─────────────────────────┤
│ Setting                     │ Value                   │
├─────────────────────────────┼─────────────────────────┤
│ llm_model                   │ gemini/gemini-2.0-flash │
│ embedding_model             │ gemini/text-embedding-004│
│ atomizer                    │ sentence                │
│ questions_per_atom          │ 15                      │
│ question_diversity_threshold│ 0.85                    │
│ data_dir                    │ ./isotope_data          │
│ vector_store                │ chroma                  │
│ doc_store                   │ sqlite                  │
│ dedup_strategy              │ source_aware            │
│ default_k                   │ 5                       │
└─────────────────────────────┴─────────────────────────┘
```

## Global Options

These options work with any command:

- `--version, -v` - Show version and exit
- `--help` - Show help for any command

```bash
isotope --version
isotope --help
isotope ingest --help
```

## Configuration

The CLI reads configuration from environment variables. See [Configuration Guide](./configuration.md) for details.

Key variables:
- `ISOTOPE_DATA_DIR` - Default data directory
- `ISOTOPE_LLM_MODEL` - Model for question generation and synthesis
- `ISOTOPE_EMBEDDING_MODEL` - Model for embeddings
- `GOOGLE_API_KEY` / `OPENAI_API_KEY` - Provider API keys

## Scripting

Use `--plain` for scriptable output:

```bash
# Check if database exists
if isotope status --plain 2>/dev/null; then
    echo "Database exists"
fi

# Get answer without formatting
isotope query "What is the API endpoint?" --plain | grep "Answer:"
```

## Common Workflows

### Initial Setup

```bash
# Set your API key
export GOOGLE_API_KEY="your-api-key"

# Ingest your docs
isotope ingest docs/

# Verify
isotope status
```

### Re-ingesting Updated Files

Isotope uses source-aware deduplication by default. Re-ingesting a file replaces the old version:

```bash
# Edit docs/guide.md...

# Re-ingest (automatically replaces old version)
isotope ingest docs/guide.md
```

### Querying Workflow

```bash
# Quick answer
isotope query "How do I authenticate?"

# More context
isotope query "authentication" -k 10

# Just the chunks (no LLM)
isotope query "authentication" --raw
```

### Cleanup

```bash
# See what's indexed
isotope list

# Remove outdated file
isotope delete docs/deprecated.md

# Verify
isotope status
```

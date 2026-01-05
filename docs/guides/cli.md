# CLI Reference

Isotope includes a command-line interface for quick ingestion and querying workflows.

## Installation

The CLI is included when you install Isotope with CLI extras:

```bash
pip install isotope-rag[all]

# Or pick specific extras:
pip install isotope-rag[cli]              # CLI only
pip install "isotope-rag[cli,litellm]"    # CLI + LiteLLM
pip install "isotope-rag[cli,chroma]"     # CLI + ChromaDB
```

Note: `isotope-rag[cli]` installs the CLI only. To ingest/query you also need a provider +
storage backend (e.g. `isotope-rag[cli,litellm,chroma]` or `isotope-rag[all]`). `--raw` disables
answer synthesis but still requires embeddings.

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
- `--data-dir, -d` - Data directory (default: `data_dir` in config or `./isotope_data`)
- `--config, -c` - Path to config file (overrides auto-discovery)
- `--plain` - Plain text output without colors

**Examples:**

```bash
# Ingest a single file
isotope ingest docs/guide.md

# Ingest a directory (recursively finds supported files)
isotope ingest docs/

# Use a custom data directory
isotope ingest docs/ --data-dir ./my_data

# Use a specific config file
isotope ingest docs/ --config ./isotope.yaml
```

**What happens:**
1. Loads supported files (`.txt`, `.md`, `.markdown`, plus PDFs/HTML if loader extras are installed)
2. Breaks content into chunks
3. Atomizes chunks into facts
4. Generates questions for each atom
5. Embeds and indexes questions
6. Shows statistics when complete

**Output:**
```
Ingested 2 files (3 chunks)
Created 12 atoms
Indexed 89 questions
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
- `--config, -c` - Path to config file (overrides auto-discovery)
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
      Python is a high-level programming language created by Guido...
```

---

### `isotope status`

Show database statistics.

```bash
isotope status [options]
```

**Options:**
- `--data-dir, -d` - Data directory
- `--config, -c` - Path to config file (overrides auto-discovery)
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
- `--config, -c` - Path to config file (overrides auto-discovery)
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
│ /Users/you/project/docs/api.md            │ 3     │
│ /Users/you/project/docs/authentication.md │ 2     │
│ /Users/you/project/docs/getting-started.md│ 5     │
│ /Users/you/project/docs/quickstart.md     │ 1     │
│ /Users/you/project/README.md              │ 12    │
└─────────────────────────────┴───────┘
```

Sources are stored as absolute paths by default, so your output will likely show full paths.

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
- `--config, -c` - Path to config file (overrides auto-discovery)
- `--force, -f` - Skip confirmation prompt
- `--plain` - Plain text output

**Examples:**

```bash
# Delete with confirmation
isotope delete /Users/you/project/docs/old.md

# Force delete (no confirmation)
isotope delete /Users/you/project/docs/old.md --force
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

**Options:**
- `--config, -c` - Path to config file (overrides auto-discovery)

**Output:**
```
┌──────────────────────────────┬─────────────────────────┬─────────────┐
│ Setting                      │ Value                   │ Source      │
├──────────────────────────────┼─────────────────────────┼─────────────┤
│ provider                     │ litellm                 │ config file │
│ llm_model                    │ openai/gpt-4o           │ config file │
│ embedding_model              │ openai/text-embedding-3-small │ config file │
│                              │                         │             │
│ questions_per_atom           │ 15                      │ env var     │
│ question_diversity_threshold │ 0.85                    │ env var     │
│ diversity_scope              │ global                  │ env var     │
│ default_k                    │ 5                       │ env var     │
└──────────────────────────────┴─────────────────────────┴─────────────┘
```

---

### `isotope init`

Initialize a new `isotope.yaml` configuration file.

```bash
isotope init [options]
```

**Options:**
- `--provider, -p` - Provider to use (`litellm` or `custom`)
- `--llm-model` - LLM model (for `litellm` provider)
- `--embedding-model` - Embedding model (for `litellm` provider)

**Example:**

```bash
isotope init --provider litellm --llm-model openai/gpt-4o --embedding-model openai/text-embedding-3-small
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

The CLI reads provider configuration from a YAML file and behavioral settings from `ISOTOPE_*` env vars. See [Configuration Guide](./configuration.md) for details.

**Config file discovery**:
- Looks for `isotope.yaml`, `isotope.yml`, or `.isotoperc` in the current directory or parents
- Use `--config` to point to a specific file

Key variables:
- `ISOTOPE_LITELLM_LLM_MODEL` - LiteLLM model (CLI fallback if no config file)
- `ISOTOPE_LITELLM_EMBEDDING_MODEL` - LiteLLM embedding model (CLI fallback)
- `ISOTOPE_QUESTIONS_PER_ATOM` - Question generation count per atom
- `ISOTOPE_QUESTION_DIVERSITY_THRESHOLD` - Diversity filter threshold
- `ISOTOPE_DIVERSITY_SCOPE` - Diversity scope (`global`, `per_chunk`, `per_atom`)
- `ISOTOPE_DEFAULT_K` - Default top-k
- Provider API keys (`OPENAI_API_KEY`, `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`, etc.)

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

Isotope uses content hashes to skip unchanged files. Re-ingesting a file with changes replaces the old version:

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

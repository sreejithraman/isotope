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
- `--no-progress` - Disable progress bars

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
1. Loads supported files (`.txt`, `.text`, `.md`, `.markdown`, plus PDFs/HTML if loader extras are installed)
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

When running in a terminal, `isotope ingest` shows progress bars by default.
Use `--plain` or `--no-progress` to disable them.

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
- `--show-matched-questions, -q` - Show which generated question matched each result
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

# Show matched questions
isotope query "authentication" --show-matched-questions
```

Returns a synthesized answer with source citations. Use `--raw` to skip synthesis and get matched chunks only.

---

### `isotope status`

Show database statistics.

```bash
isotope status [options]
```

**Options:**
- `--data-dir, -d` - Data directory
- `--config, -c` - Path to config file (overrides auto-discovery)
- `--detailed` - Show question distribution per source
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

Use `--detailed` to see question distribution per source.

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

Lists all indexed sources with chunk counts. Sources are stored as absolute paths.

---

### `isotope questions sample`

Show a random sample of generated questions.

```bash
isotope questions sample [options]
```

**Options:**
- `--source, -s` - Filter by source file
- `-n` - Number of questions to sample (default: 5)
- `--data-dir, -d` - Data directory
- `--config, -c` - Path to config file (overrides auto-discovery)
- `--plain` - Plain text output

**Examples:**

```bash
# Sample 5 questions
isotope questions sample

# Sample 10 questions
isotope questions sample -n 10

# Sample from a specific source
isotope questions sample --source /Users/you/project/docs/api.md
```

Shows a table of randomly sampled questions from the index.

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

Show all effective configuration settings and their sources.

```bash
isotope config
```

**Options:**
- `--config, -c` - Path to config file (overrides auto-discovery)

Shows all effective settings with their values and source (`yaml`, `env var`, or `default`).

---

### `isotope init`

Initialize a new `isotope.yaml` configuration file with an interactive wizard.

```bash
isotope init [options]
```

**Options:**
- `--provider, -p` - Provider to use (`litellm` or `custom`)
- `--llm-model` - LLM model (for `litellm` provider)
- `--embedding-model` - Embedding model (for `litellm` provider)

**Interactive mode** (no arguments):

```bash
isotope init
```

The wizard walks you through selecting LLM/embedding models, API keys, rate-limit preferences, and quality priorities. It generates `isotope.yaml` and saves API keys to `.env`. For local models (e.g., `ollama/llama3.2`), API key prompts are skipped automatically.

**Non-interactive mode:**

```bash
isotope init --provider litellm --llm-model openai/gpt-5-mini-2025-08-07 --embedding-model openai/text-embedding-3-small
```

**Generated config:**

The wizard generates a config with only non-default values active, plus commented defaults for discoverability:

```yaml
# isotope.yaml (generated by isotope init)
provider: litellm
llm_model: openai/gpt-5-mini-2025-08-07
embedding_model: openai/text-embedding-3-small

settings:
  max_concurrent_llm_calls: 2  # rate-limit friendly

# Uncomment to customize (showing defaults):
#   atomization_granularity: fine  # coarse=fast, medium=balanced, fine=quality
#   questions_per_atom: 5          # more = better recall, higher cost
#   diversity_scope: global        # global | per_chunk | per_atom
#   max_concurrent_llm_calls: 10   # parallel LLM requests
#
# Advanced settings:
#   num_retries: 5
#   diversity_threshold: 0.85
#   default_k: 5
```

API keys entered during initialization are saved to `.env` for automatic loading.

## Global Options

These options work with any command:

- `--version, -v` - Show version and exit
- `--help` - Show help for any command

```bash
isotope --version
isotope --help
isotope ingest --help
```

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
export OPENAI_API_KEY="your-api-key"

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

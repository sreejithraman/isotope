# Configuration Guide

IsotopeDB uses [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) for configuration. All settings can be configured via environment variables (prefixed with `ISOTOPE_`) or programmatically.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ISOTOPE_LLM_MODEL` | `gemini/gemini-3-flash-preview` | LLM for question generation |
| `ISOTOPE_EMBEDDING_MODEL` | `gemini/text-embedding-004` | Embedding model |
| `ISOTOPE_ATOMIZER` | `sentence` | Atomization strategy: `sentence` or `llm` |
| `ISOTOPE_QUESTIONS_PER_ATOM` | `15` | Questions to generate per atom |
| `ISOTOPE_QUESTION_PROMPT` | (default prompt) | Custom prompt template |
| `ISOTOPE_QUESTION_DIVERSITY_THRESHOLD` | `0.85` | Similarity threshold for dedup (empty = disable) |
| `ISOTOPE_DATA_DIR` | `./isotope_data` | Storage directory |
| `ISOTOPE_VECTOR_STORE` | `chroma` | Vector store backend |
| `ISOTOPE_DOC_STORE` | `sqlite` | Document store backend |
| `ISOTOPE_DEDUP_STRATEGY` | `source_aware` | Re-ingestion strategy: `none` or `source_aware` |
| `ISOTOPE_DEFAULT_K` | `5` | Default number of results to return |

## Programmatic Configuration

```python
from isotopedb import Settings

# Create custom settings
settings = Settings(
    llm_model="openai/gpt-4",
    embedding_model="openai/text-embedding-3-small",
    atomizer="llm",
    questions_per_atom=10,
    question_diversity_threshold=0.9,
)
```

## Provider API Keys

IsotopeDB uses [LiteLLM](https://docs.litellm.ai/) for LLM and embedding calls. Set the appropriate API key for your provider:

### Gemini (default)

```bash
export GOOGLE_API_KEY="your-gemini-api-key"
```

### OpenAI

```bash
export OPENAI_API_KEY="your-openai-api-key"
export ISOTOPE_LLM_MODEL="openai/gpt-4"
export ISOTOPE_EMBEDDING_MODEL="openai/text-embedding-3-small"
```

### Anthropic

```bash
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export ISOTOPE_LLM_MODEL="anthropic/claude-sonnet-4-5-20250929"
# Note: Anthropic doesn't provide embeddings, use a different provider
```

### Azure OpenAI

```bash
export AZURE_API_KEY="your-azure-api-key"
export AZURE_API_BASE="https://your-resource.openai.azure.com"
export AZURE_API_VERSION="2024-02-15-preview"
export ISOTOPE_LLM_MODEL="azure/your-deployment-name"
```

See the [LiteLLM provider list](https://docs.litellm.ai/docs/providers) for more options.

## Example .env File

```bash
# Provider keys
GOOGLE_API_KEY=your-gemini-api-key

# LLM settings
ISOTOPE_LLM_MODEL=gemini/gemini-3-flash-preview
ISOTOPE_EMBEDDING_MODEL=gemini/text-embedding-004

# Atomization
ISOTOPE_ATOMIZER=sentence
ISOTOPE_QUESTIONS_PER_ATOM=15

# Question diversity (set empty to disable)
ISOTOPE_QUESTION_DIVERSITY_THRESHOLD=0.85

# Storage
ISOTOPE_DATA_DIR=./isotope_data

# Re-ingestion
ISOTOPE_DEDUP_STRATEGY=source_aware
```

## Configuration Details

### LLM Model

The `llm_model` setting uses LiteLLM format: `provider/model-name`.

Examples:
- `gemini/gemini-3-flash-preview` (default, fast and cheap)
- `openai/gpt-4` (high quality)
- `anthropic/claude-sonnet-4-5-20250929` (balanced)
- `anthropic/claude-haiku-4-5-20251001` (fast)

### Embedding Model

The `embedding_model` setting also uses LiteLLM format.

Examples:
- `gemini/text-embedding-004` (default, 768 dimensions)
- `openai/text-embedding-3-small` (1536 dimensions)
- `openai/text-embedding-3-large` (3072 dimensions)

### Atomizer

Choose between two atomization strategies:

| Value | Strategy | Best For |
|-------|----------|----------|
| `sentence` | Split by sentence (pysbd) | Structured docs, speed |
| `llm` | LLM extraction | Semantic extraction, accuracy |

See [Atomization Guide](./atomization.md) for detailed comparison.

### Question Diversity Threshold

Controls how aggressively duplicate questions are removed:

- `0.85` (default): Remove questions with >85% similarity
- `0.95`: Keep more similar questions
- `0.70`: More aggressive deduplication
- Empty string or `None`: Disable deduplication

### Diversity Filter Scope

Controls how diversity filtering is applied during question generation:

| Value | Description | Performance | Trade-off |
|-------|-------------|-------------|-----------|
| `global` (default) | Filter across all questions | O(N²) complexity | Best retrieval quality (research-validated) |
| `per_chunk` | Filter within each chunk only | ~100x faster | May retain similar questions from different chunks |
| `per_atom` | Filter within each atom only | ~1000x faster | Only deduplicates within each atom's questions |

**Default**: `global` (research-validated for maximum retrieval performance)

**When to use non-default scopes**:
- Large corpora (>10,000 questions) where global filtering is slow
- Performance-critical ingestion pipelines
- When you've verified that cross-chunk/cross-atom duplicates are acceptable

**Programmatic configuration**:
```python
from isotopedb import Isotope

# Default: global filtering (best quality, slower for large corpora)
ingestor = iso.ingestor()

# Performance optimization: filter within chunks (~100x faster)
ingestor = iso.ingestor(diversity_scope="per_chunk")

# Maximum speed: filter within atoms only (~1000x faster)
ingestor = iso.ingestor(diversity_scope="per_atom")
```

**How it works**:
- `global`: Compares every question against every other question (paper-validated)
- `per_chunk`: Groups questions by chunk_id, filters each group separately
- `per_atom`: Groups questions by atom_id, filters each group separately

With `per_chunk` or `per_atom`, duplicate questions may remain if they come from different groups. Global filtering catches all duplicates but requires more comparisons.

### Question Generation Concurrency

Controls how many LLM calls can run concurrently during question generation:

**Default**: `1` (sequential, same as current behavior)

**When to increase**:
- You want faster ingestion and can handle concurrent LLM API calls
- Your LLM provider supports high rate limits
- You're ingesting large documents with many atoms

**Programmatic configuration**:
```python
from isotopedb import Isotope

# Default: sequential generation (current behavior)
ingestor = iso.ingestor()

# Concurrent generation: 5 LLM calls at once
ingestor = iso.ingestor(max_concurrent_generations=5)

# Maximum concurrency (be mindful of rate limits)
ingestor = iso.ingestor(max_concurrent_generations=10)
```

**Note**: This setting only affects *concurrency*, not the quality of question generation. Each atom still gets its own LLM call with full chunk context (research-validated approach). Higher values mean faster ingestion at the same cost, but may hit API rate limits.

**Implementation**: Uses `asyncio.Semaphore` to limit concurrent `litellm.acompletion()` calls.

### Deduplication Strategy

Controls what happens when you re-ingest documents:

| Value | Behavior |
|-------|----------|
| `source_aware` | Delete existing chunks from same source before adding new ones |
| `none` | Never delete existing data (may create duplicates) |

## Accessing Settings in Code

```python
from isotopedb import Settings

# Load from environment
settings = Settings()

# Access values
print(settings.llm_model)
print(settings.embedding_model)
print(settings.atomizer)
```

Settings are validated on construction. Invalid values raise `ValidationError`.

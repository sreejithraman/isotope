# Configuration Guide

Isotope separates configuration into three parts:

- **Provider configuration**: Which LLM/embedding service to use (LiteLLM, custom, etc.)
- **Storage configuration**: Where data lives (local Chroma + SQLite or custom stores)
- **Behavioral settings**: How the system operates (questions per atom, diversity threshold, etc.)

## Quick Start

### Using LiteLLM (Recommended)

```python
from isotope import Isotope, LiteLLMProvider, LocalStorage

# Simple setup with LiteLLM + local storage
iso = Isotope(
    provider=LiteLLMProvider(
        llm="openai/gpt-5-mini-2025-08-07",
        embedding="openai/text-embedding-3-small",
    ),
    storage=LocalStorage("./my_data"),
)

# Create ingestor and retriever
ingestor = iso.ingestor()
retriever = iso.retriever()

# For synthesized answers, pass llm_model to the retriever:
# retriever = iso.retriever(llm_model="openai/gpt-5-mini-2025-08-07")
```

### Using Explicit Stores (Enterprise)

```python
from isotope import Isotope, LiteLLMProvider
from isotope.stores import (
    ChromaEmbeddedQuestionStore,
    SQLiteChunkStore,
    SQLiteAtomStore,
    SQLiteSourceRegistry,
)

iso = Isotope(
    provider=LiteLLMProvider(
        llm="openai/gpt-5-mini-2025-08-07",
        embedding="openai/text-embedding-3-small",
    ),
    embedded_question_store=ChromaEmbeddedQuestionStore("./data/chroma"),
    chunk_store=SQLiteChunkStore("./data/chunks.db"),
    atom_store=SQLiteAtomStore("./data/atoms.db"),
    source_registry=SQLiteSourceRegistry("./data/sources.db"),
)
```

If you need custom embedders or generators, implement a custom `ProviderConfig`
and pass it to `Isotope` (see below).

## Configuration Objects

Isotope uses configuration objects that know how to build components.
You pass a `ProviderConfig` (builds embedder/atomizer/generator) and either
a `StorageConfig` (builds stores) or explicit store instances.

### Path 1: LiteLLMProvider + LocalStorage (Recommended)

```python
from isotope import Isotope, LiteLLMProvider, LocalStorage, Settings

iso = Isotope(
    provider=LiteLLMProvider(
        llm="openai/gpt-5-mini-2025-08-07",            # LiteLLM model format
        embedding="openai/text-embedding-3-small",
    ),
    storage=LocalStorage("./isotope_data"),
    # Optional: customize behavior
    settings=Settings(
        atomization_granularity="fine",  # coarse=fast, medium=balanced, fine=quality (default)
        questions_per_atom=5,
    ),
)
```

This creates:
- Local stores (ChromaEmbeddedQuestionStore, SQLiteChunkStore, SQLiteAtomStore, SQLiteSourceRegistry)
- LiteLLM-backed components via `LiteLLMProvider`

Requires extras: `isotope-rag[chroma]` for `LocalStorage`, `isotope-rag[litellm]` for `LiteLLMProvider`.

### Path 2: Explicit Stores

For enterprise deployments or custom storage implementations:

```python
from isotope import Isotope, LiteLLMProvider
from isotope.stores import (
    ChromaEmbeddedQuestionStore,
    SQLiteChunkStore,
    SQLiteAtomStore,
    SQLiteSourceRegistry,
)

iso = Isotope(
    provider=LiteLLMProvider(
        llm="openai/gpt-5-mini-2025-08-07",
        embedding="openai/text-embedding-3-small",
    ),
    embedded_question_store=ChromaEmbeddedQuestionStore("./data/chroma"),
    chunk_store=SQLiteChunkStore("./data/chunks.db"),
    atom_store=SQLiteAtomStore("./data/atoms.db"),
    source_registry=SQLiteSourceRegistry("./data/sources.db"),
)
```

If you implement a custom `EmbeddedQuestionStore`, it must also implement:
- `sample(n, chunk_ids=None)` for `isotope questions sample`
- `count_by_chunk_ids(chunk_ids)` for `isotope status --detailed`

### Path 3: Custom ProviderConfig (Bring Your Own Components)

Implement `ProviderConfig` to plug in custom embedders/atomizers/generators:

```python
from dataclasses import dataclass
from isotope import Isotope, LocalStorage, Settings

@dataclass(frozen=True)
class BedrockProvider:
    def build_embedder(self, settings: Settings):
        return BedrockEmbedder(model="amazon.titan-embed-text-v1")

    def build_atomizer(self, settings: Settings):
        return MyAtomizer()

    def build_question_generator(self, settings: Settings):
        return MyQuestionGenerator(num_questions=settings.questions_per_atom)

    # Optional: used for LLM synthesis in Retriever.get_answer()
    # def build_llm_client(self, settings: Settings | None = None):
    #     return MyLLMClient(model="bedrock/...")

iso = Isotope(
    provider=BedrockProvider(),
    storage=LocalStorage("./isotope_data"),
)
```

## Storage

Rough estimates per 1,000 chunks:

| Component | Size |
|-----------|------|
| SQLite metadata | 1-5 MB |
| Chroma embeddings | 50-100 MB |
| Total | ~100 MB |

For 100,000 chunks, plan for ~10 GB storage. SSD recommended for Chroma queries; avoid network mounts (NFS, SMB) for the data directory.

## CLI Configuration

The CLI uses a config file (`isotope.yaml`, `isotope.yml`, or `.isotoperc`) for provider configuration.
It searches the current directory and parent directories (up to 10 levels). Use `--config` to point
to a specific file if needed.

### Creating a Config File

```bash
# Create config for LiteLLM
isotope init --provider litellm --llm-model openai/gpt-5-mini-2025-08-07 --embedding-model openai/text-embedding-3-small

# Or create manually
```

### Config File Format

```yaml
# isotope.yaml

# LiteLLM provider
provider: litellm
llm_model: openai/gpt-5-mini-2025-08-07
embedding_model: openai/text-embedding-3-small

# Optional: Data directory
data_dir: ./isotope_data

# Behavioral settings (all optional)
settings:
  atomization_granularity: fine  # coarse=fast (3-5 facts), medium=balanced, fine=quality (default)
  questions_per_atom: 5          # questions per atomic fact
  diversity_scope: global        # global | per_chunk | per_atom
  max_concurrent_llm_calls: 10   # parallel LLM requests
  num_retries: 5                 # retry count on failures
  rate_limit_profile: aggressive # aggressive | conservative
  synthesis_temperature: 0.3     # answer synthesis temperature
  generation_preset: cloud       # cloud | local (auto-detects if omitted)
  batch_size: 1                  # atoms per LLM prompt (overrides preset)
```

### Configuration Precedence

Settings are loaded in this order (highest to lowest precedence):
1. **Environment variables** - for CI/CD override
2. **YAML config file** - `settings:` section
3. **Built-in defaults**

This means environment variables always override YAML settings.

### Custom Provider

```yaml
# isotope.yaml

provider: custom

# Python import paths for your classes
embedder: my_package.BedrockEmbedder
question_generator: my_package.BedrockGenerator
atomizer: my_package.BedrockAtomizer

# Optional kwargs for each class
embedder_kwargs:
  region: us-east-1
question_generator_kwargs:
  temperature: 0.7
atomizer_kwargs: {}
```

### CLI Environment Variable Fallback

If no config file is found, the CLI falls back to LiteLLM environment variables:

```bash
export ISOTOPE_LITELLM_LLM_MODEL=openai/gpt-5-mini-2025-08-07
export ISOTOPE_LITELLM_EMBEDDING_MODEL=openai/text-embedding-3-small
```

## Behavioral Settings (Settings + CLI Env Vars)

These settings apply regardless of which provider you use. In Python, pass a
`Settings` object to `Isotope` (or use `Settings.with_profile(...)` for rate-limit presets).
The CLI supports a subset via `ISOTOPE_*` env vars and YAML `settings:`.

### Common Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `ISOTOPE_ATOMIZATION_GRANULARITY` | `fine` | Atomization detail: `coarse` (fast, 3-5 facts), `medium`, `fine` (default, all facts) |
| `ISOTOPE_QUESTIONS_PER_ATOM` | `5` | Questions to generate per atom |
| `ISOTOPE_DIVERSITY_SCOPE` | `global` | Scope for diversity filter: `global`, `per_chunk`, `per_atom` |
| `ISOTOPE_MAX_CONCURRENT_LLM_CALLS` | `10` | Maximum concurrent async LLM requests |
| `ISOTOPE_RATE_LIMIT_PROFILE` | (none) | Apply `aggressive` or `conservative` preset (sets concurrency + retries, CLI only) |

### Advanced Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `ISOTOPE_NUM_RETRIES` | `5` | Number of retries on LLM failures |
| `ISOTOPE_QUESTION_DIVERSITY_THRESHOLD` | `0.85` | Similarity threshold for dedup (empty = disable) |
| `ISOTOPE_DEFAULT_K` | `5` | Default number of results to return |
| `ISOTOPE_QUESTION_GENERATOR_PROMPT` | (default prompt) | Custom question generation prompt template |
| `ISOTOPE_ATOMIZER_PROMPT` | (default prompt) | Custom atomization prompt template |
| `ISOTOPE_SYNTHESIS_PROMPT` | (default prompt) | Custom answer synthesis prompt template |

### YAML-only Settings (no env var support)

These settings can be configured in `isotope.yaml` under the `settings:` section, but have no corresponding `ISOTOPE_*` environment variables:

| YAML Key | Default | Description |
|----------|---------|-------------|
| `generation_preset` | `None` | `cloud` or `local` preset for batching (auto-detects from model if omitted) |
| `batch_size` | `None` | Atoms per LLM prompt (overrides preset when set) |
| `synthesis_temperature` | `0.3` | Temperature for answer synthesis |

**Rate-limit profiles (Python):**
```python
from isotope import Settings

# Conservative defaults for free tiers / strict rate limits
settings = Settings.with_profile("conservative", questions_per_atom=5)
```

## Provider API Keys

Isotope uses [LiteLLM](https://docs.litellm.ai/) for LLM and embedding calls. Set the appropriate API key for your provider.

> **Can I use multiple providers?** Yes, but not simultaneously in one Isotope instance. You can use different providers for different projects (separate `isotope.yaml`) or switch providers by updating your config.

| Provider | Env Var | LLM Model | Embedding Model |
|----------|---------|-----------|-----------------|
| **OpenAI** | `OPENAI_API_KEY` | `openai/gpt-5-mini-2025-08-07` | `openai/text-embedding-3-small` |
| **Gemini** | `GEMINI_API_KEY` | `gemini/gemini-3-flash-preview` | `gemini/gemini-embedding-001` |
| **Anthropic** | `ANTHROPIC_API_KEY` | `anthropic/claude-sonnet-4-5-20250929` | Use OpenAI or Gemini embeddings |
| **Azure OpenAI** | `AZURE_API_KEY` + `AZURE_API_BASE` + `AZURE_API_VERSION` | `azure/your-deployment-name` | `azure/your-embedding-deployment` |

See the [LiteLLM provider list](https://docs.litellm.ai/docs/providers) for more options.

## Configuration Details

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
# Default: global filtering (best quality, slower for large corpora)
ingestor = iso.ingestor()

# Performance optimization: filter within chunks (~100x faster)
ingestor = iso.ingestor(diversity_scope="per_chunk")

# Maximum speed: filter within atoms only (~1000x faster)
ingestor = iso.ingestor(diversity_scope="per_atom")
```

### Re-ingestion Behavior

When using `Isotope.ingest_file()`, the system automatically handles re-ingestion via
the `SourceRegistry`. It tracks content hashes to detect changed files and cascades
deletion of old data before adding new content. This is handled automatically—no
configuration needed.

### Async Ingestion

For large documents, use async methods to parallelize question generation:

```python
import asyncio
from isotope.question_generator.base import BatchConfig

# Async file ingestion (10-50x faster for large docs)
result = asyncio.run(iso.aingest_file("large-report.pdf"))

# Or with explicit ingestor and custom batching
ingestor = iso.ingestor(
    batch_config=BatchConfig(batch_size=5, max_concurrent=2)
)
result = asyncio.run(ingestor.aingest_chunks(chunks))
```

`BatchConfig` controls both the number of atoms per prompt (`batch_size`) and the
maximum concurrent LLM calls (`max_concurrent`). Higher values = faster ingestion
but may hit rate limits.

Isotope will auto-detect a batching preset from the model name (cloud vs local).
Override with `generation_preset`, `batch_size`, or `max_concurrent_llm_calls` in `Settings`
if you need tighter rate limits or larger batches.

**Programmatic configuration**:
```python
settings = Settings(
    questions_per_atom=10,
    max_concurrent_llm_calls=20,  # Increase for faster ingestion
    generation_preset="local",   # Or "cloud" to force single-atom prompts
    batch_size=5,                # Override preset batch size
)

iso = Isotope(
    provider=LiteLLMProvider(...),
    storage=LocalStorage("./data"),
    settings=settings,
)
```

### Prompt Customization

You can customize the prompts used by Isotope for atomization, question generation, and answer synthesis.
In Python, set these on `Settings`; in the CLI, use the `ISOTOPE_*` env vars shown below.

#### Atomizer Prompt (`ISOTOPE_ATOMIZER_PROMPT`)

Used when breaking chunks into atomic facts with `LLMAtomizer`. Your prompt must include `{content}`.

```bash
export ISOTOPE_ATOMIZER_PROMPT="Extract key facts from this text as a JSON array of strings:\n\n{content}"
```

#### Question Generator Prompt (`ISOTOPE_QUESTION_GENERATOR_PROMPT`)

Used when generating questions for each atom. Available variables:
- `{num_questions}` - Number of questions to generate
- `{atom_content}` - The atomic fact text
- `{chunk_content}` - The parent chunk content (may be empty)

```bash
export ISOTOPE_QUESTION_GENERATOR_PROMPT="Generate {num_questions} search queries for: {atom_content}"
```

Note: Custom prompts apply to single-atom generation (batch_size=1). When batching
multiple atoms into one prompt, Isotope uses a built-in multi-atom template.

#### Synthesis Prompt (`ISOTOPE_SYNTHESIS_PROMPT`)

Used when synthesizing answers from retrieved context. Available variables:
- `{context}` - The retrieved chunks/context
- `{query}` - The user's question

```bash
export ISOTOPE_SYNTHESIS_PROMPT="Answer based on context:\n\n{context}\n\nQuestion: {query}"
```

## Example .env File (CLI)

```bash
# Provider API key
OPENAI_API_KEY=your-openai-api-key

# Behavioral settings (all optional, shown with defaults)
ISOTOPE_ATOMIZATION_GRANULARITY=fine
ISOTOPE_QUESTIONS_PER_ATOM=5
ISOTOPE_DIVERSITY_SCOPE=global
ISOTOPE_MAX_CONCURRENT_LLM_CALLS=10
ISOTOPE_RATE_LIMIT_PROFILE=aggressive
ISOTOPE_NUM_RETRIES=5
ISOTOPE_QUESTION_DIVERSITY_THRESHOLD=0.85
ISOTOPE_DEFAULT_K=5
```

Note: In Python, pass LLM/embedding models directly to `LiteLLMProvider` (or your custom
provider config). The CLI can also read `ISOTOPE_LITELLM_LLM_MODEL` and
`ISOTOPE_LITELLM_EMBEDDING_MODEL` if no config file is present.

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
        llm="openai/gpt-4o",
        embedding="openai/text-embedding-3-small",
    ),
    storage=LocalStorage("./my_data"),
)

# Create ingestor and retriever
ingestor = iso.ingestor()
retriever = iso.retriever()

# For synthesized answers, pass llm_model to the retriever:
# retriever = iso.retriever(llm_model="openai/gpt-4o")
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
        llm="openai/gpt-4o",
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
from isotope import Isotope, LiteLLMProvider, LocalStorage

iso = Isotope(
    provider=LiteLLMProvider(
        llm="openai/gpt-4o",            # LiteLLM model format
        embedding="openai/text-embedding-3-small",
        atomizer_type="llm",            # or "sentence"
    ),
    storage=LocalStorage("./isotope_data"),
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
        llm="openai/gpt-4o",
        embedding="openai/text-embedding-3-small",
    ),
    embedded_question_store=ChromaEmbeddedQuestionStore("./data/chroma"),
    chunk_store=SQLiteChunkStore("./data/chunks.db"),
    atom_store=SQLiteAtomStore("./data/atoms.db"),
    source_registry=SQLiteSourceRegistry("./data/sources.db"),
)
```

### Path 3: Custom ProviderConfig (Bring Your Own Components)

Implement `ProviderConfig` to plug in custom embedders/atomizers/generators:

```python
from dataclasses import dataclass
from isotope import Isotope, LocalStorage, Settings

@dataclass(frozen=True)
class BedrockProvider:
    def build_embedder(self):
        return BedrockEmbedder(model="amazon.titan-embed-text-v1")

    def build_atomizer(self, settings: Settings):
        return MyAtomizer()

    def build_question_generator(self, settings: Settings):
        return MyQuestionGenerator(num_questions=settings.questions_per_atom)

iso = Isotope(
    provider=BedrockProvider(),
    storage=LocalStorage("./isotope_data"),
)
```

## CLI Configuration

The CLI uses a config file (`isotope.yaml`, `isotope.yml`, or `.isotoperc`) for provider configuration.
It searches the current directory and parent directories (up to 10 levels). Use `--config` to point
to a specific file if needed.

### Creating a Config File

```bash
# Create config for LiteLLM
isotope init --provider litellm --llm-model openai/gpt-4o --embedding-model openai/text-embedding-3-small

# Or create manually
```

### Config File Format

```yaml
# isotope.yaml

# LiteLLM provider
provider: litellm
llm_model: openai/gpt-4o
embedding_model: openai/text-embedding-3-small

# Optional settings
data_dir: ./isotope_data
use_sentence_atomizer: false
```

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
export ISOTOPE_LITELLM_LLM_MODEL=openai/gpt-4o
export ISOTOPE_LITELLM_EMBEDDING_MODEL=openai/text-embedding-3-small
```

## Behavioral Settings (Settings + CLI Env Vars)

These settings apply regardless of which provider you use. In Python, pass a
`Settings` object to `Isotope`. The CLI reads equivalent `ISOTOPE_*` env vars
and passes them explicitly.

| Variable | Default | Description |
|----------|---------|-------------|
| `ISOTOPE_QUESTIONS_PER_ATOM` | `15` | Questions to generate per atom |
| `ISOTOPE_QUESTION_GENERATOR_PROMPT` | (default prompt) | Custom question generation prompt template |
| `ISOTOPE_ATOMIZER_PROMPT` | (default prompt) | Custom atomization prompt template |
| `ISOTOPE_QUESTION_DIVERSITY_THRESHOLD` | `0.85` | Similarity threshold for dedup (empty = disable) |
| `ISOTOPE_DIVERSITY_SCOPE` | `global` | Scope for diversity filter: `global`, `per_chunk`, `per_atom` |
| `ISOTOPE_DEFAULT_K` | `5` | Default number of results to return |
| `ISOTOPE_SYNTHESIS_PROMPT` | (default prompt) | Custom answer synthesis prompt template |
| `ISOTOPE_MAX_CONCURRENT_QUESTIONS` | `10` | Maximum concurrent async question generation requests |

## Provider API Keys

Isotope uses [LiteLLM](https://docs.litellm.ai/) for LLM and embedding calls. Set the appropriate API key for your provider:

### OpenAI

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

```python
from isotope import Isotope, LiteLLMProvider, LocalStorage

iso = Isotope(
    provider=LiteLLMProvider(
        llm="openai/gpt-4o",
        embedding="openai/text-embedding-3-small",
    ),
    storage=LocalStorage("./isotope_data"),
)
```

### Gemini

```bash
export GOOGLE_API_KEY="your-gemini-api-key"
```

```python
from isotope import Isotope, LiteLLMProvider, LocalStorage

iso = Isotope(
    provider=LiteLLMProvider(
        llm="gemini/gemini-2.0-flash",
        embedding="gemini/text-embedding-004",
    ),
    storage=LocalStorage("./isotope_data"),
)
```

### Anthropic

```bash
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

```python
from isotope import Isotope, LiteLLMProvider, LocalStorage

iso = Isotope(
    provider=LiteLLMProvider(
        llm="anthropic/claude-sonnet-4-5-20250929",
        embedding="openai/text-embedding-3-small",  # Anthropic doesn't provide embeddings
    ),
    storage=LocalStorage("./isotope_data"),
)
```

### Azure OpenAI

```bash
export AZURE_API_KEY="your-azure-api-key"
export AZURE_API_BASE="https://your-resource.openai.azure.com"
export AZURE_API_VERSION="2024-02-15-preview"
```

```python
from isotope import Isotope, LiteLLMProvider, LocalStorage

iso = Isotope(
    provider=LiteLLMProvider(
        llm="azure/your-deployment-name",
        embedding="azure/your-embedding-deployment",
    ),
    storage=LocalStorage("./isotope_data"),
)
```

See the [LiteLLM provider list](https://docs.litellm.ai/docs/providers) for more options.

## Imports

LiteLLM provider clients and model constants live in `isotope.providers.litellm`:

```python
# Configuration objects
from isotope import Isotope, LiteLLMProvider, LocalStorage, Settings
from isotope.configuration import ProviderConfig, StorageConfig

# LiteLLM provider clients + model constants
from isotope.providers.litellm import LiteLLMClient, LiteLLMEmbeddingClient
from isotope.providers.litellm import ChatModels, EmbeddingModels

# Component wrappers (use any LLMClient / EmbeddingClient)
from isotope.atomizer import LLMAtomizer
from isotope.embedder import ClientEmbedder
from isotope.question_generator import ClientQuestionGenerator

# Abstract base classes (for custom implementations)
from isotope import Embedder, QuestionGenerator, Atomizer
from isotope.providers import LLMClient, EmbeddingClient
```

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

# Async file ingestion (10-50x faster for large docs)
result = asyncio.run(iso.aingest_file("large-report.pdf"))

# Or with explicit ingestor and custom concurrency
ingestor = iso.ingestor(max_concurrent_questions=20)
result = asyncio.run(ingestor.aingest_chunks(chunks))
```

The `max_concurrent_questions` setting controls how many LLM requests run concurrently.
Higher values = faster ingestion but may hit rate limits. Default is 10.

**Programmatic configuration**:
```python
settings = Settings(
    questions_per_atom=15,
    max_concurrent_questions=20,  # Increase for faster ingestion
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

#### Synthesis Prompt (`ISOTOPE_SYNTHESIS_PROMPT`)

Used when synthesizing answers from retrieved context. Available variables:
- `{context}` - The retrieved chunks/context
- `{query}` - The user's question

```bash
export ISOTOPE_SYNTHESIS_PROMPT="Answer based on context:\n\n{context}\n\nQuestion: {query}"
```

## Accessing Settings in Code

```python
from isotope import Isotope, LiteLLMProvider, LocalStorage, Settings

# Settings are plain Python objects (no env auto-loading)
settings = Settings(
    questions_per_atom=20,
    diversity_scope="per_chunk",
    max_concurrent_questions=20,  # For async ingestion
)

iso = Isotope(
    provider=LiteLLMProvider(
        llm="openai/gpt-4o",
        embedding="openai/text-embedding-3-small",
    ),
    storage=LocalStorage("./isotope_data"),
    settings=settings,
)

# Access values
print(settings.questions_per_atom)
print(settings.question_diversity_threshold)
print(settings.diversity_scope)
print(settings.default_k)
print(settings.max_concurrent_questions)
```

## Example .env File (CLI)

```bash
# Provider API key
OPENAI_API_KEY=your-openai-api-key

# Behavioral settings (all optional, shown with defaults)
ISOTOPE_QUESTIONS_PER_ATOM=15
ISOTOPE_QUESTION_DIVERSITY_THRESHOLD=0.85
ISOTOPE_DIVERSITY_SCOPE=global
ISOTOPE_DEFAULT_K=5
ISOTOPE_MAX_CONCURRENT_QUESTIONS=10
```

Note: In Python, pass LLM/embedding models directly to `LiteLLMProvider` (or your custom
provider config). The CLI can also read `ISOTOPE_LITELLM_LLM_MODEL` and
`ISOTOPE_LITELLM_EMBEDDING_MODEL` if no config file is present.

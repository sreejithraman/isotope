# Configuration Guide

IsotopeDB separates **provider configuration** from **behavioral settings**:

- **Provider configuration**: Which LLM/embedding service to use (LiteLLM, custom, etc.)
- **Behavioral settings**: How the system operates (questions per atom, diversity threshold, etc.)

## Quick Start

### Using LiteLLM (Recommended)

```python
from isotopedb import Isotope

# Simple setup with LiteLLM
iso = Isotope.with_litellm(
    llm_model="openai/gpt-4o",
    embedding_model="openai/text-embedding-3-small",
    data_dir="./my_data",
)

# Create ingestor and retriever
ingestor = iso.ingestor()
retriever = iso.retriever()

# For synthesized answers, pass llm_model to the retriever:
# retriever = iso.retriever(llm_model="openai/gpt-4o")
```

### Using Custom Components (Enterprise)

```python
from isotopedb import Isotope
from isotopedb.atomizer import LLMAtomizer
from isotopedb.embedder import ClientEmbedder
from isotopedb.providers.litellm import LiteLLMClient, LiteLLMEmbeddingClient
from isotopedb.question_generator import ClientQuestionGenerator

llm_client = LiteLLMClient(model="openai/gpt-4o")
embedding_client = LiteLLMEmbeddingClient(model="openai/text-embedding-3-small")

# Explicit component configuration
iso = Isotope(
    vector_store=MyPineconeStore(...),
    chunk_store=MyPostgresChunkStore(...),
    atom_store=MyPostgresAtomStore(...),
    embedder=ClientEmbedder(embedding_client=embedding_client),
    atomizer=LLMAtomizer(llm_client=llm_client),
    question_generator=ClientQuestionGenerator(llm_client=llm_client),
)
```

## Provider Configuration

### Path 1: `Isotope.with_litellm()` Factory

The simplest way to get started:

```python
from isotopedb import Isotope

iso = Isotope.with_litellm(
    llm_model="openai/gpt-4o",           # LiteLLM model format
    embedding_model="openai/text-embedding-3-small",
    data_dir="./isotope_data",            # Optional, defaults to "./isotope_data"
    use_sentence_atomizer=False,          # Optional, use sentence splitter instead of LLM
)
```

This creates:
- Local stores (ChromaVectorStore, SQLiteChunkStore, SQLiteAtomStore)
- LiteLLM clients wrapped by `ClientEmbedder`, `ClientQuestionGenerator`, and `LLMAtomizer`

### Path 2: Explicit Components

For enterprise deployments or custom implementations:

```python
from isotopedb import Isotope
from isotopedb.atomizer import LLMAtomizer
from isotopedb.embedder import ClientEmbedder
from isotopedb.providers.litellm import LiteLLMClient, LiteLLMEmbeddingClient
from isotopedb.question_generator import ClientQuestionGenerator

# Bring your own stores
from my_company.stores import PineconeStore, PostgresChunkStore, PostgresAtomStore

llm_client = LiteLLMClient(model="openai/gpt-4o")
embedding_client = LiteLLMEmbeddingClient(model="openai/text-embedding-3-small")

iso = Isotope(
    vector_store=PineconeStore(...),
    chunk_store=PostgresChunkStore(...),
    atom_store=PostgresAtomStore(...),
    embedder=ClientEmbedder(embedding_client=embedding_client),
    atomizer=LLMAtomizer(llm_client=llm_client),
    question_generator=ClientQuestionGenerator(llm_client=llm_client),
)

# All components configured - ingestor is a simple call
ingestor = iso.ingestor()
```

### Path 3: Local Stores with Custom Embedder

Mix local stores with custom embedder:

```python
from isotopedb import Isotope
from my_company.embedder import BedrockEmbedder

iso = Isotope.with_local_stores(
    embedder=BedrockEmbedder(model="amazon.titan-embed-text-v1"),
    atomizer=my_atomizer,
    question_generator=my_generator,
    data_dir="./isotope_data",
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

### Environment Variable Override

If no config file is found, the CLI falls back to LiteLLM environment variables:

```bash
export ISOTOPE_LITELLM_LLM_MODEL=openai/gpt-4o
export ISOTOPE_LITELLM_EMBEDDING_MODEL=openai/text-embedding-3-small
```

## Behavioral Settings (Environment Variables)

These settings apply regardless of which provider you use. Configure via environment variables with the `ISOTOPE_` prefix.

| Variable | Default | Description |
|----------|---------|-------------|
| `ISOTOPE_QUESTIONS_PER_ATOM` | `15` | Questions to generate per atom |
| `ISOTOPE_QUESTION_GENERATOR_PROMPT` | (default prompt) | Custom question generation prompt template |
| `ISOTOPE_ATOMIZER_PROMPT` | (default prompt) | Custom atomization prompt template |
| `ISOTOPE_QUESTION_DIVERSITY_THRESHOLD` | `0.85` | Similarity threshold for dedup (empty = disable) |
| `ISOTOPE_DIVERSITY_SCOPE` | `global` | Scope for diversity filter: `global`, `per_chunk`, `per_atom` |
| `ISOTOPE_DEFAULT_K` | `5` | Default number of results to return |
| `ISOTOPE_SYNTHESIS_PROMPT` | (default prompt) | Custom answer synthesis prompt template |

## Provider API Keys

IsotopeDB uses [LiteLLM](https://docs.litellm.ai/) for LLM and embedding calls. Set the appropriate API key for your provider:

### OpenAI

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

```python
iso = Isotope.with_litellm(
    llm_model="openai/gpt-4o",
    embedding_model="openai/text-embedding-3-small",
)
```

### Gemini

```bash
export GOOGLE_API_KEY="your-gemini-api-key"
```

```python
iso = Isotope.with_litellm(
    llm_model="gemini/gemini-2.0-flash",
    embedding_model="gemini/text-embedding-004",
)
```

### Anthropic

```bash
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

```python
iso = Isotope.with_litellm(
    llm_model="anthropic/claude-sonnet-4-5-20250929",
    embedding_model="openai/text-embedding-3-small",  # Anthropic doesn't provide embeddings
)
```

### Azure OpenAI

```bash
export AZURE_API_KEY="your-azure-api-key"
export AZURE_API_BASE="https://your-resource.openai.azure.com"
export AZURE_API_VERSION="2024-02-15-preview"
```

```python
iso = Isotope.with_litellm(
    llm_model="azure/your-deployment-name",
    embedding_model="azure/your-embedding-deployment",
)
```

See the [LiteLLM provider list](https://docs.litellm.ai/docs/providers) for more options.

## Imports

LiteLLM provider clients and model constants live in `isotopedb.providers.litellm`:

```python
# LiteLLM provider clients + model constants
from isotopedb.providers.litellm import LiteLLMClient, LiteLLMEmbeddingClient
from isotopedb.providers.litellm import ChatModels, EmbeddingModels

# Component wrappers (use any LLMClient / EmbeddingClient)
from isotopedb.atomizer import LLMAtomizer
from isotopedb.embedder import ClientEmbedder
from isotopedb.question_generator import ClientQuestionGenerator

# Abstract base classes (for custom implementations)
from isotopedb import Embedder, QuestionGenerator, Atomizer
from isotopedb.providers import LLMClient, EmbeddingClient
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

### Prompt Customization

You can customize the prompts used by IsotopeDB for atomization, question generation, and answer synthesis.

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
from isotopedb.config import Settings

# Load behavioral settings from environment
settings = Settings()

# Access values
print(settings.questions_per_atom)
print(settings.question_diversity_threshold)
print(settings.diversity_scope)
print(settings.default_k)
```

## Example .env File

```bash
# Provider API key
OPENAI_API_KEY=your-openai-api-key

# Behavioral settings (all optional, shown with defaults)
ISOTOPE_QUESTIONS_PER_ATOM=15
ISOTOPE_QUESTION_DIVERSITY_THRESHOLD=0.85
ISOTOPE_DIVERSITY_SCOPE=global
ISOTOPE_DEFAULT_K=5
```

Note: In Python, pass LLM/embedding models directly to `Isotope.with_litellm()` (or your custom
components). The CLI can also read `ISOTOPE_LITELLM_LLM_MODEL` and
`ISOTOPE_LITELLM_EMBEDDING_MODEL` if no config file is present.

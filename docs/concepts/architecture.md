# Isotope Architecture

This document explains how Isotope implements the Reverse RAG approach from [arXiv:2405.12363](./reverse-rag.md).

## System Overview

Isotope has three layers:

1. **High-level API** - The `Isotope` class for simple usage
2. **Pipelines** - `Ingestor` and `Retriever` for orchestrated workflows
3. **Components** - Individual building blocks (stores, atomizers, embedders, etc.)

```
┌─────────────────────────────────────────────────────────────────┐
│                         Isotope                                  │
│  (Central configuration - creates Ingestors and Retrievers)     │
└─────────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                                         ▼
┌─────────────────┐                     ┌─────────────────┐
│    Ingestor     │                     │    Retriever    │
│  (Ingestion     │                     │  (Query         │
│   pipeline)     │                     │   pipeline)     │
└─────────────────┘                     └─────────────────┘
         │                                         │
         ▼                                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Components                                 │
│  Stores │ Atomizers │ Embedders │ Generators │ Loaders           │
└─────────────────────────────────────────────────────────────────┘
```

## The `Isotope` Class

The `Isotope` class is the main entry point. It bundles configuration and creates pipelines:

```python
from isotope import Isotope, LiteLLMProvider, LocalStorage

# Simple setup with LiteLLM + local stores
iso = Isotope(
    provider=LiteLLMProvider(
        llm="openai/gpt-4o",
        embedding="openai/text-embedding-3-small",
    ),
    storage=LocalStorage("./my_data"),
)

# Create pipelines
ingestor = iso.ingestor()
retriever = iso.retriever(llm_model="openai/gpt-4o")
```

`Isotope` handles:
- Holding references to stores and components (built from configuration objects)
- Applying behavioral defaults from `Settings` (questions per atom, dedup, default_k)
- Providing factory methods for `Ingestor` and `Retriever`

## Pipelines

### Ingestor

The `Ingestor` orchestrates the ingestion pipeline:

```
Chunks → Store Chunks → Atomize → Store Atoms → Generate Questions
       → Embed → Filter → Index
```

Two modes available:
- `ingest_chunks()` - Synchronous, sequential question generation
- `aingest_chunks()` - Async, concurrent question generation (10-50x faster for large docs)

```python
# Sync ingestion
ingestor = iso.ingestor()
result = ingestor.ingest_chunks([chunk1, chunk2])

# Async ingestion (faster for large documents)
result = await ingestor.aingest_chunks([chunk1, chunk2])
# result = {"chunks": 2, "atoms": 8, "questions": 45, ...}
```

Key features:
- Progress callbacks for UI/logging
- Diversity filtering to remove duplicate questions
- Concurrent async question generation with configurable rate limiting

### Retriever

The `Retriever` orchestrates the query pipeline:

```
Query → Embed → Search Questions → Fetch Chunks → (Optional) Synthesize Answer
```

```python
retriever = iso.retriever(llm_model="openai/gpt-4o")

# With LLM synthesis
response = retriever.get_answer("How do I authenticate?")
print(response.answer)  # LLM-generated answer
print(response.results)  # Source chunks

# Without synthesis (raw chunks only)
results = retriever.get_context("authentication")
```

## Data Flow

```
┌──────────┐    ┌────────┐    ┌───────┐    ┌───────────┐    ┌────────────┐
│ Document │───▶│ Chunks │───▶│ Atoms │───▶│ Questions │───▶│ Embeddings │
└──────────┘    └────────┘    └───────┘    └───────────┘    └────────────┘
                    │              │             │                 │
                    ▼              ▼             ▼                 ▼
               ┌───────────┐ ┌──────────┐  ┌────────────┐   ┌─────────────┐
               │ChunkStore │ │AtomStore │  │DiversityFilt│   │ EmbeddedQuestionStore │
               └───────────┘ └──────────┘  └────────────┘   └─────────────┘
```

## Core Components

### Data Models (`models/`)

| Model | Purpose | Paper Concept |
|-------|---------|---------------|
| `Chunk` | Document fragment with content + source | Document chunk |
| `Atom` | Atomic fact extracted from chunk | Atomic unit y_i^(k,j) |
| `Question` | Synthetic question for an atom | Generated question |
| `EmbeddedQuestion` | Question + embedding vector | Indexed question |
| `SearchResult` | Query result with score | Retrieved result |
| `QueryResponse` | Full query response with answer + results | API response |

**Paper notation**: `y_i^(k,j)` means the i-th question for the j-th atom of chunk k.

### Storage Layer (`stores/`)

Four abstract base classes define storage contracts:

| ABC | Implementations | Stores |
|-----|-----------------|--------|
| `EmbeddedQuestionStore` | `ChromaEmbeddedQuestionStore` | Question embeddings |
| `ChunkStore` | `SQLiteChunkStore` | Chunk content |
| `AtomStore` | `SQLiteAtomStore` | Atom content |
| `SourceRegistry` | `SQLiteSourceRegistry` | Source content hashes |

All stores support deletion by chunk ID for re-ingestion workflows.

### Atomization (`atomizer/`)

Atomizers break chunks into atomic facts:

| Atomizer | Strategy | Use Case |
|----------|----------|----------|
| `SentenceAtomizer` | Split by sentence (pysbd) | Fast, structured docs |
| `LLMAtomizer` | LLM extraction | Semantic extraction |

See [Atomization Guide](../guides/atomization.md) for when to use each.

### Question Generation (`question_generator/`)

| Component | Purpose |
|-----------|---------|
| `QuestionGenerator` | Generate synthetic questions via LLM |
| `DiversityFilter` | Remove near-duplicate questions |
| `BatchGenerationError` | Exception for partial batch failures |

The generator creates ~15 questions per atom by default. The diversity filter (threshold 0.85) removes questions with >85% cosine similarity, keeping diverse coverage.

**Async Support**: For large ingests, use async methods for concurrent question generation:
- `agenerate(atom, chunk_content)` - Single atom, async
- `agenerate_batch(atoms, chunk_contents, max_concurrent)` - Concurrent batch with semaphore-based rate limiting

### Embeddings (`embedder/`)

`Embedder` wraps an `EmbeddingClient` (e.g., `LiteLLMEmbeddingClient`) for embedding generation:
- `embed_text(str)` → single embedding
- `embed_texts(list[str])` → list of embeddings
- `embed_question(Question)` → single `EmbeddedQuestion`
- `embed_questions(list[Question])` → list of `EmbeddedQuestion`

### Source Registry

Tracks ingested sources for change detection. The `SourceRegistry` ABC is defined in `stores/base.py` alongside other store ABCs, with `SQLiteSourceRegistry` in `stores/source_registry.py`.

When re-ingesting files via `Isotope.ingest_file()`, the system compares content hashes
to detect changes and automatically cascades deletion of old data before adding new content.

### File Loading (`loaders/`)

| Loader | File Types |
|--------|------------|
| `TextLoader` | `.txt`, `.md`, `.markdown` |
| `PyPDFLoader` | `.pdf` (via pypdf) |
| `PDFPlumberLoader` | `.pdf` (via pdfplumber) |
| `HTMLLoader` | `.html`, `.htm` |
| `LoaderRegistry` | Auto-selects loader by extension |

## Component Mapping to Paper

| Paper Concept | Isotope Component |
|---------------|---------------------|
| Document chunking | `Loader` + user chunking |
| Atomic unit extraction | `Atomizer` (sentence or LLM) |
| Question generation | `QuestionGenerator` |
| Question deduplication | `DiversityFilter` |
| Embedding | `Embedder` |
| Vector index | `EmbeddedQuestionStore` |
| Retrieval pipeline | `Retriever` |
| Answer synthesis | `Retriever.get_answer()` with `llm_model` set |

## Design Decisions

### Why Three Layers?

1. **Isotope class** - Easy for common cases (80% of users)
2. **Pipelines** - Control without wiring (15% of users)
3. **Components** - Full customization (5% of users)

Most users just need `Isotope`. Power users can customize pipelines or swap components.

### Why Separate Stores?

We split storage into three stores (EmbeddedQuestionStore, ChunkStore, AtomStore) because:
1. **Vector stores have specific needs**: Optimized for similarity search
2. **Chunks need full-text retrieval**: For returning content to users
3. **Atoms bridge the gap**: Track which questions came from which chunks

### Why ABCs?

Abstract base classes enable:
- Swapping implementations (e.g., Pinecone instead of Chroma)
- Testing with mocks
- Clear contracts for custom implementations

### Why LiteLLM?

LiteLLM provides a unified interface to 100+ LLM providers. This means:
- Switch providers by changing a string (e.g., `openai/gpt-4` → `anthropic/claude-3`)
- Same API for embeddings and completions
- No vendor lock-in

## Configuration

Behavioral settings (questions per atom, diversity, default_k, etc.) are configured via
the `Settings` class (the CLI reads `ISOTOPE_*` env vars and passes them explicitly).
Provider and storage configuration are explicit in code via configuration objects
or via the CLI config file. See
[Configuration Guide](../guides/configuration.md) for details.

```python
# LiteLLM + local stores
from isotope import Isotope, LiteLLMProvider, LocalStorage

iso = Isotope(
    provider=LiteLLMProvider(
        llm="openai/gpt-4o",
        embedding="openai/text-embedding-3-small",
    ),
    storage=LocalStorage("./isotope_data"),
)

# Explicit stores + custom provider
iso = Isotope(
    provider=my_provider_config,
    embedded_question_store=my_embedded_question_store,
    chunk_store=my_chunk_store,
    atom_store=my_atom_store,
    source_registry=my_source_registry,
)
```

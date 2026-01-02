# IsotopeDB Architecture

This document explains how IsotopeDB implements the Reverse RAG approach from [arXiv:2405.12363](./reverse-rag.md).

## System Overview

IsotopeDB has three layers:

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
│  Stores │ Atomizers │ Embedders │ Generators │ Dedup │ Loaders  │
└─────────────────────────────────────────────────────────────────┘
```

## The `Isotope` Class

The `Isotope` class is the main entry point. It bundles configuration and creates pipelines:

```python
from isotopedb import Isotope

# Simple setup with defaults from environment
iso = Isotope()

# Or with explicit configuration
iso = Isotope(
    data_dir="./my_data",
    embedding_model="openai/text-embedding-3-small",
    llm_model="openai/gpt-4",
)

# Create pipelines
ingestor = iso.ingestor()
retriever = iso.retriever()
```

`Isotope` handles:
- Reading configuration from `ISOTOPE_*` environment variables
- Creating and wiring up all stores (vector, doc, atom)
- Creating embedders with the configured model
- Providing factory methods for `Ingestor` and `Retriever`

## Pipelines

### Ingestor

The `Ingestor` orchestrates the ingestion pipeline:

```
Chunks → Dedup → Store Chunks → Atomize → Store Atoms → Generate Questions
       → Embed → Filter → Index
```

```python
ingestor = iso.ingestor()
result = ingestor.ingest_chunks([chunk1, chunk2])
# result = {"chunks": 2, "atoms": 8, "questions": 45, ...}
```

Key features:
- Progress callbacks for UI/logging
- Source-aware deduplication (re-ingest updated files)
- Diversity filtering to remove duplicate questions

### Retriever

The `Retriever` orchestrates the query pipeline:

```
Query → Embed → Search Questions → Fetch Chunks → (Optional) Synthesize Answer
```

```python
retriever = iso.retriever()

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
               ┌─────────┐   ┌──────────┐  ┌────────────┐   ┌─────────────┐
               │DocStore │   │AtomStore │  │DiversityFilt│   │ VectorStore │
               └─────────┘   └──────────┘  └────────────┘   └─────────────┘
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

Three abstract base classes define storage contracts:

| ABC | Implementations | Stores |
|-----|-----------------|--------|
| `VectorStore` | `ChromaVectorStore` | Question embeddings |
| `DocStore` | `SQLiteDocStore` | Chunk content |
| `AtomStore` | `SQLiteAtomStore` | Atom content |

All stores support deletion by chunk ID for re-ingestion workflows.

### Atomization (`atomizer/`)

Atomizers break chunks into atomic facts:

| Atomizer | Strategy | Use Case |
|----------|----------|----------|
| `SentenceAtomizer` | Split by sentence (pysbd) | Fast, structured docs |
| `LLMAtomizer` | LLM extraction | Semantic extraction |

See [Atomization Guide](../guides/atomization.md) for when to use each.

### Question Generation (`generator/`)

| Component | Purpose |
|-----------|---------|
| `QuestionGenerator` | Generate synthetic questions via LLM |
| `DiversityFilter` | Remove near-duplicate questions |

The generator creates ~15 questions per atom by default. The diversity filter (threshold 0.85) removes questions with >85% cosine similarity, keeping diverse coverage.

### Embeddings (`embedder/`)

`Embedder` wraps LiteLLM for embedding generation:
- `embed(str)` → single embedding
- `embed_batch(list[str])` → list of embeddings
- `embed_questions(list[Question])` → list of `EmbeddedQuestion`

### Deduplication (`dedup/`)

Handles re-ingestion of updated documents:

| Strategy | Behavior |
|----------|----------|
| `NoDedup` | Never delete existing data |
| `SourceAwareDedup` | Delete chunks from same source before re-ingesting |

### File Loading (`loaders/`)

| Loader | File Types |
|--------|------------|
| `TextLoader` | `.txt`, `.md`, `.markdown` |
| `PyPDFLoader` | `.pdf` (via pypdf) |
| `PDFPlumberLoader` | `.pdf` (via pdfplumber) |
| `HTMLLoader` | `.html`, `.htm` |
| `LoaderRegistry` | Auto-selects loader by extension |

## Component Mapping to Paper

| Paper Concept | IsotopeDB Component |
|---------------|---------------------|
| Document chunking | `Loader` + user chunking |
| Atomic unit extraction | `Atomizer` (sentence or LLM) |
| Question generation | `QuestionGenerator` |
| Question deduplication | `DiversityFilter` |
| Embedding | `Embedder` |
| Vector index | `VectorStore` |
| Retrieval pipeline | `Retriever` |
| Answer synthesis | `Retriever.query()` with LLM |

## Design Decisions

### Why Three Layers?

1. **Isotope class** - Easy for common cases (80% of users)
2. **Pipelines** - Control without wiring (15% of users)
3. **Components** - Full customization (5% of users)

Most users just need `Isotope`. Power users can customize pipelines or swap components.

### Why Separate Stores?

We split storage into three stores (VectorStore, DocStore, AtomStore) because:
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

All components are configurable via environment variables or the `Settings` class. See [Configuration Guide](../guides/configuration.md) for details.

The `Isotope` class reads from environment by default:

```python
# These are equivalent
iso = Isotope()  # Reads ISOTOPE_* env vars

iso = Isotope(
    data_dir=os.getenv("ISOTOPE_DATA_DIR", "./isotope_data"),
    embedding_model=os.getenv("ISOTOPE_EMBEDDING_MODEL", "gemini/text-embedding-004"),
    llm_model=os.getenv("ISOTOPE_LLM_MODEL", "gemini/gemini-3-flash-preview"),
)
```

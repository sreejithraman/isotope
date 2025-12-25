# Isotope Design Document

**Project:** Isotope (Reverse RAG Database)
**Package:** `isotopedb`
**CLI:** `isotope`
**Date:** 2025-12-25
**Updated:** 2025-12-25 (Paper review: atomic decomposition, question diversity)

## Overview

Isotope is a "Reverse RAG" system that indexes anticipated questions rather than content chunks. Traditional RAG indexes answers and hopes they match user questions. Isotope inverts this by decomposing content into atomic statements, generating questions for each atom, and indexing those questions instead—achieving higher semantic alignment during retrieval.

**Reference:** Based on "Question-Based Retrieval Using Atomic Units for Enterprise RAG" (arXiv:2405.12363v2)

## Core Concept

```
Traditional RAG:  User Question → Search Chunks → Hope for match
Isotope:          User Question → Search Questions → Exact chunk lookup
```

**Pipeline:**
```
Chunk → Atoms → Questions → Embeddings → Index
```

**Isotopes:** Generated questions that map back to the same content chunk.
**Retrieval:** Match user query against question index (question ↔ question), then fetch the corresponding content.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Isotope                              │
├─────────────────────────────────────────────────────────────┤
│  Ingestor                    │   Retriever                  │
│  ─────────                   │   ─────────                  │
│  • Load files                │   • Search questions         │
│  • Atomize chunks            │   • Fetch chunks             │
│  • Generate questions        │   • Synthesize answer        │
│  • Deduplicate questions     │                              │
│  • Embed & index             │                              │
├──────────────────────────────┴──────────────────────────────┤
│  VectorStore (ABC)           │   DocStore (ABC)             │
│  ──────────────────          │   ──────────────             │
│  • ChromaDB (MVP)            │   • SQLite (MVP)             │
├─────────────────────────────────────────────────────────────┤
│  LiteLLM                                                    │
│  • Question generation (any provider)                       │
│  • Embeddings (any provider)                                │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
isotopedb/
├── pyproject.toml
├── README.md
├── src/
│   └── isotopedb/
│       ├── __init__.py              # Public API exports
│       ├── cli.py                   # CLI entry point (typer)
│       ├── config.py                # Pydantic Settings
│       │
│       ├── models/                  # Pydantic data models
│       │   ├── __init__.py
│       │   ├── chunk.py             # Chunk, Atom, Question
│       │   └── results.py           # SearchResult, QueryResponse
│       │
│       ├── stores/                  # Storage abstractions
│       │   ├── __init__.py
│       │   ├── base.py              # VectorStore ABC, DocStore ABC
│       │   ├── chroma.py            # ChromaDB implementation
│       │   └── sqlite.py            # SQLite doc store
│       │
│       ├── atomizer/                # Atomic decomposition
│       │   ├── __init__.py
│       │   ├── base.py              # Atomizer ABC
│       │   ├── sentence.py          # Sentence-based atomization
│       │   └── llm.py               # LLM-based atomic fact extraction
│       │
│       ├── dedup/                   # Re-ingestion deduplication
│       │   ├── __init__.py
│       │   ├── base.py              # Deduplicator ABC
│       │   ├── none.py              # NoDedup
│       │   └── source.py            # SourceAwareDedup
│       │
│       ├── loaders/                 # File loading/chunking
│       │   ├── __init__.py
│       │   ├── base.py              # Loader ABC
│       │   └── unstructured.py      # Unstructured implementation
│       │
│       ├── llm/                     # LLM integration
│       │   ├── __init__.py
│       │   ├── generator.py         # Question generation + diversity dedup
│       │   └── embedder.py          # Embedding wrapper
│       │
│       ├── ingestor.py              # Ingestion pipeline
│       └── retriever.py             # Retrieval pipeline
│
└── tests/
    ├── conftest.py
    ├── test_ingestor.py
    ├── test_retriever.py
    └── ...
```

## Data Models

```python
from pydantic import BaseModel, Field
from uuid import uuid4


class Chunk(BaseModel):
    """A piece of content that can answer questions."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str
    source: str                      # file path or identifier
    metadata: dict = Field(default_factory=dict)


class Atom(BaseModel):
    """An atomic statement extracted from a chunk."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str                     # the atomic statement
    chunk_id: str                    # references Chunk.id


class Question(BaseModel):
    """An atomic question that an atom/chunk answers."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    text: str                        # the question itself
    chunk_id: str                    # references Chunk.id (for retrieval)
    atom_id: str | None = None       # references Atom.id (for tracing)
    embedding: list[float] | None = None


class SearchResult(BaseModel):
    """A single matched question + its chunk."""
    question: Question
    chunk: Chunk
    score: float


class QueryResponse(BaseModel):
    """Full response to a user query."""
    query: str
    answer: str | None               # None if --raw mode
    results: list[SearchResult]
```

**Metadata conventions by document type:**

| Type | Common metadata fields |
|------|----------------------|
| PDF | `type`, `page`, `total_pages`, `author`, `title` |
| Markdown | `type`, `title`, `heading` |
| Web | `type`, `url`, `title`, `fetched_at` |
| Plain text | `type` |

## Abstract Base Classes

```python
# stores/base.py
class VectorStore(ABC):
    @abstractmethod
    def add(self, questions: list[Question]) -> None: ...

    @abstractmethod
    def search(self, embedding: list[float], k: int = 5) -> list[tuple[Question, float]]:
        """Search for similar questions. Returns (Question, score) pairs ordered by relevance."""
        ...

    @abstractmethod
    def delete_by_chunk_ids(self, chunk_ids: list[str]) -> None: ...

    @abstractmethod
    def list_chunk_ids(self) -> set[str]: ...


class DocStore(ABC):
    @abstractmethod
    def put(self, chunk: Chunk) -> None: ...

    @abstractmethod
    def get(self, chunk_id: str) -> Chunk | None: ...

    @abstractmethod
    def get_many(self, chunk_ids: list[str]) -> list[Chunk]: ...

    @abstractmethod
    def delete(self, chunk_id: str) -> None: ...

    @abstractmethod
    def list_sources(self) -> list[str]: ...

    @abstractmethod
    def get_by_source(self, source: str) -> list[Chunk]: ...


**Storage notes:**
- VectorStore stores Question objects (embeddings + metadata including chunk_id)
- DocStore stores Chunk objects (content + source + metadata)
- Atoms are transient (used during ingestion, not persisted)


# atomizer/base.py
class Atomizer(ABC):
    @abstractmethod
    def atomize(self, chunk: Chunk) -> list[Atom]: ...


# dedup/base.py
class Deduplicator(ABC):
    """Handles re-ingestion of sources (not question diversity)."""
    @abstractmethod
    def get_chunks_to_remove(
        self,
        new_chunks: list[Chunk],
        doc_store: DocStore
    ) -> list[str]: ...


# loaders/base.py
class Loader(ABC):
    @abstractmethod
    def load(self, path: str) -> list[Chunk]: ...

    @abstractmethod
    def supports(self, path: str) -> bool: ...
```

## Configuration

```python
# config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ISOTOPE_",
        env_file=".env",
        extra="ignore",
    )

    # LLM (question generation)
    llm_model: str = "gpt-4o-mini"

    # Embeddings
    embedding_model: str = "text-embedding-3-small"

    # Atomization
    atomizer: Literal["sentence", "llm"] = "sentence"

    # Question generation
    questions_per_atom: int = 5
    question_prompt: str | None = None  # None = use default

    # Question diversity deduplication
    question_diversity_threshold: float | None = 0.85  # None = disabled

    # Storage
    data_dir: str = "./isotope_data"
    vector_store: Literal["chroma"] = "chroma"
    doc_store: Literal["sqlite"] = "sqlite"

    # Re-ingestion deduplication
    dedup_strategy: Literal["none", "source_aware"] = "source_aware"

    # Retrieval
    default_k: int = 5
```

**Environment variables:**

```bash
# Isotope config
ISOTOPE_LLM_MODEL=gemini/gemini-1.5-flash
ISOTOPE_EMBEDDING_MODEL=gemini/text-embedding-004
ISOTOPE_ATOMIZER=sentence
ISOTOPE_QUESTIONS_PER_ATOM=5
ISOTOPE_QUESTION_DIVERSITY_THRESHOLD=0.85

# Provider API keys (LiteLLM convention)
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...
ANTHROPIC_API_KEY=...
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
```

## Ingestion Pipeline

```
Load File (via Loader)
    ↓
Chunk content
    ↓
[Source-Aware Dedup] ← if re-ingesting, remove old chunks first
    ↓
Atomize (via Atomizer) ← break chunks into atomic statements
    ↓
Generate Questions (per atom)
    ↓
Embed Questions
    ↓
[Question Diversity Dedup] ← remove similar questions (cosine threshold)
    ↓
Store (VectorStore for questions, DocStore for chunks)
```

**Question Diversity Deduplication:**

Applied after embedding to remove redundant questions using cosine similarity.

```python
# llm/generator.py
def deduplicate_questions(
    questions: list[Question],
    embeddings: list[list[float]],
    threshold: float = 0.85
) -> tuple[list[Question], list[list[float]]]:
    """
    Remove questions with pairwise cosine similarity above threshold.

    Called after embedding (requires embeddings for similarity computation).
    Returns filtered questions and their corresponding embeddings.

    Paper finding: Retaining 50% of questions maintains max performance.
    Even 20% retention shows minimal degradation.
    """
    ...
```

## Retrieval Pipeline

```
User Query
    ↓
Embed Query
    ↓
Search Question Index (VectorStore)
    ↓
Get chunk_ids from matched questions
    ↓
Fetch Chunks (DocStore)
    ↓
[Optional] Synthesize Answer (LLM)
    ↓
Return QueryResponse
```

## CLI Commands

```bash
# Ingestion
isotope ingest <path>              # File or directory
isotope ingest ./docs --plain      # Plain text output

# Querying
isotope query "How do I authenticate?"
isotope query "Rate limits?" --raw --k 10

# Management
isotope list                       # Show indexed sources
isotope delete docs/old.md         # Remove a source
isotope status                     # Show stats
isotope config                     # Show settings
```

**Output modes:**
- Rich (default): Colors, tables, progress bars
- Plain (`--plain`): Simple text for scripting

**Query modes:**
- Synthesized (default): LLM-generated answer from chunks
- Raw (`--raw`): Return matched chunks only

## Dependencies

**Core (required):**
```
pydantic>=2.0
pydantic-settings>=2.0
litellm>=1.0
chromadb>=0.4
typer>=0.9
rich>=13.0
```

**Optional:**
```
[unstructured]  →  unstructured[all-docs]  # PDF, DOCX, etc.
[dev]           →  pytest, ruff, mypy      # Development
```

**Install:**
```bash
pip install isotopedb                  # Core
pip install isotopedb[unstructured]    # + file parsing
```

## Implementation Roadmap

### Phase 1: Core Foundation
- Project setup (pyproject.toml, src layout)
- Pydantic models (Chunk, Atom, Question, SearchResult, QueryResponse)
- Settings/config
- Abstract base classes

### Phase 2: Storage Implementations
- ChromaDB vector store
- SQLite doc store
- NoDedup and SourceAwareDedup

### Phase 3: Atomization
- Sentence-based atomizer (default)
- LLM-based atomizer (optional)

### Phase 4: LLM Integration
- Question generation with LiteLLM
- Question diversity deduplication
- Embedding with LiteLLM
- Default prompt template

### Phase 5: Pipelines
- Ingestor (ingest files, ingest_chunks)
- Retriever (search, query with synthesis)

### Phase 6: File Loading
- Basic text/markdown loader (built-in)
- Unstructured loader (optional dep)

### Phase 7: CLI
- All commands (ingest, query, list, delete, status, config)
- Rich output + plain mode

### Phase 8: Polish
- Tests
- README with examples
- Error handling and validation

## Known Limitations

**Single-hop queries only (MVP):**
- Isotope handles queries where the answer exists in a single chunk
- Multi-hop queries (requiring information from multiple chunks) are not optimized
- Future: Hybrid retrieval or iterative retrieval approaches

**See also:** `docs/plans/2025-12-25-isotope-future.md` for deferred considerations.

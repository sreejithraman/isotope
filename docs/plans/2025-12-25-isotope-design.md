# Isotope Design Document

**Project:** Isotope (Reverse RAG Database)
**Package:** `isotopedb`
**CLI:** `isotope`
**Date:** 2025-12-25

## Overview

Isotope is a "Reverse RAG" system that indexes anticipated questions rather than content chunks. Traditional RAG indexes answers and hopes they match user questions. Isotope inverts this by generating atomic questions for each chunk and indexing those instead, achieving higher semantic alignment during retrieval.

## Core Concept

```
Traditional RAG:  User Question → Search Chunks → Hope for match
Isotope:          User Question → Search Questions → Exact chunk lookup
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
│  • Generate questions        │   • Fetch chunks             │
│  • Embed & index             │   • Synthesize answer        │
├──────────────────────────────┴──────────────────────────────┤
│  VectorStore (ABC)           │   DocStore (ABC)             │
│  ──────────────────          │   ──────────────             │
│  • ChromaDB (MVP)            │   • SQLite (MVP)             │
│  • Vespa, OpenSearch (future)│   • S3, DynamoDB (future)    │
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
│       │   ├── chunk.py             # Chunk, Question
│       │   └── results.py           # SearchResult, QueryResponse
│       │
│       ├── stores/                  # Storage abstractions
│       │   ├── __init__.py
│       │   ├── base.py              # VectorStore ABC, DocStore ABC
│       │   ├── chroma.py            # ChromaDB implementation
│       │   └── sqlite.py            # SQLite doc store
│       │
│       ├── dedup/                   # Deduplication strategies
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
│       │   ├── generator.py         # Question generation
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


class Question(BaseModel):
    """An atomic question that a chunk answers."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    text: str                        # the question itself
    chunk_id: str                    # references Chunk.id
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
    def search(self, embedding: list[float], k: int = 5) -> list[tuple[str, float]]: ...

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


# dedup/base.py
class Deduplicator(ABC):
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

    # Question generation
    questions_per_chunk: int = 5
    question_prompt: str | None = None  # None = use default

    # Storage
    data_dir: str = "./isotope_data"
    vector_store: Literal["chroma"] = "chroma"
    doc_store: Literal["sqlite"] = "sqlite"

    # Deduplication
    dedup_strategy: Literal["none", "source_aware"] = "source_aware"

    # Retrieval
    default_k: int = 5
```

**Environment variables:**

```bash
# Isotope config
ISOTOPE_LLM_MODEL=gemini/gemini-1.5-flash
ISOTOPE_EMBEDDING_MODEL=gemini/text-embedding-004
ISOTOPE_QUESTIONS_PER_CHUNK=7

# Provider API keys (LiteLLM convention)
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...
ANTHROPIC_API_KEY=...
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
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
- Pydantic models (Chunk, Question, SearchResult, QueryResponse)
- Settings/config
- Abstract base classes

### Phase 2: Storage Implementations
- ChromaDB vector store
- SQLite doc store
- NoDedup and SourceAwareDedup

### Phase 3: LLM Integration
- Question generation with LiteLLM
- Embedding with LiteLLM
- Default prompt template

### Phase 4: Pipelines
- Ingestor (ingest files, ingest_chunks)
- Retriever (search, query with synthesis)

### Phase 5: File Loading
- Basic text/markdown loader (built-in)
- Unstructured loader (optional dep)

### Phase 6: CLI
- All commands (ingest, query, list, delete, status, config)
- Rich output + plain mode

### Phase 7: Polish
- Tests
- README with examples
- Error handling and validation

## Future Considerations

**Additional vector stores:**
- Vespa, OpenSearch, PGVector, FAISS, Pinecone

**Additional doc stores:**
- S3, DynamoDB, Redis

**Additional dedup strategies:**
- Hash-based exact dedup
- Semantic dedup (vector similarity on chunks)

**Make ChromaDB optional:**
- When multiple backends exist, let users choose

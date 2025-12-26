# Isotope Phases 1-4 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the core foundation, storage layer, atomization, and LLM integration for the Isotope reverse-RAG database.

**Architecture:** A modular Python package using abstract base classes for storage and processing components. Pydantic for data models, LiteLLM for provider-agnostic LLM access, ChromaDB for vectors, SQLite for documents.

**Tech Stack:** Python 3.11+, Pydantic 2.x, pydantic-settings, LiteLLM, ChromaDB, SQLite, pytest

---

## Phase 1: Core Foundation

### Task 1.1: Project Setup

**Files:**
- Create: `pyproject.toml`
- Create: `src/isotopedb/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

**Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "isotopedb"
version = "0.1.0"
description = "Reverse RAG database - index questions, not chunks"
readme = "README.md"
requires-python = ">=3.11"
license = "MIT"
authors = [{ name = "Your Name", email = "you@example.com" }]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]
dependencies = [
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "litellm>=1.0",
    "chromadb>=0.4",
    "typer>=0.9",
    "rich>=13.0",
]

[project.optional-dependencies]
unstructured = ["unstructured[all-docs]"]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "ruff>=0.4",
    "mypy>=1.10",
]

[project.scripts]
isotope = "isotopedb.cli:app"

[tool.hatch.build.targets.wheel]
packages = ["src/isotopedb"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]

[tool.mypy]
python_version = "3.11"
strict = true

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

**Step 2: Create src/isotopedb/__init__.py**

```python
"""Isotope - Reverse RAG database."""

__version__ = "0.1.0"
```

**Step 3: Create tests/__init__.py**

```python
"""Isotope tests."""
```

**Step 4: Create tests/conftest.py**

```python
"""Shared pytest fixtures."""

import pytest
```

**Step 5: Create virtual environment and install**

Run:
```bash
cd /Users/sree/conductor/workspaces/isotope/pattaya-v1
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Expected: Package installs successfully with all dependencies.

**Step 6: Verify installation**

Run: `python -c "import isotopedb; print(isotopedb.__version__)"`
Expected: `0.1.0`

**Step 7: Commit**

```bash
git add pyproject.toml src/ tests/
git commit -m "chore: project setup with pyproject.toml"
```

---

### Task 1.2: Data Models - Chunk and Atom

**Files:**
- Create: `src/isotopedb/models/__init__.py`
- Create: `src/isotopedb/models/chunk.py`
- Create: `tests/models/__init__.py`
- Create: `tests/models/test_chunk.py`

**Step 1: Write failing test for Chunk model**

```python
# tests/models/test_chunk.py
"""Tests for Chunk and Atom models."""

import pytest
from isotopedb.models.chunk import Chunk, Atom


class TestChunk:
    def test_create_chunk_with_defaults(self):
        chunk = Chunk(content="Hello world", source="test.md")
        assert chunk.content == "Hello world"
        assert chunk.source == "test.md"
        assert chunk.id  # auto-generated
        assert chunk.metadata == {}

    def test_create_chunk_with_metadata(self):
        chunk = Chunk(
            content="Hello",
            source="test.pdf",
            metadata={"page": 1, "type": "pdf"},
        )
        assert chunk.metadata["page"] == 1
        assert chunk.metadata["type"] == "pdf"

    def test_chunk_id_is_unique(self):
        c1 = Chunk(content="a", source="x")
        c2 = Chunk(content="a", source="x")
        assert c1.id != c2.id


class TestAtom:
    def test_create_atom(self):
        atom = Atom(content="Python is a programming language.", chunk_id="chunk-123")
        assert atom.content == "Python is a programming language."
        assert atom.chunk_id == "chunk-123"
        assert atom.id  # auto-generated

    def test_atom_id_is_unique(self):
        a1 = Atom(content="fact", chunk_id="c1")
        a2 = Atom(content="fact", chunk_id="c1")
        assert a1.id != a2.id
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/models/test_chunk.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'isotopedb.models'`

**Step 3: Create models/__init__.py**

```python
# src/isotopedb/models/__init__.py
"""Data models for Isotope."""

from isotopedb.models.chunk import Atom, Chunk

__all__ = ["Chunk", "Atom"]
```

**Step 4: Create chunk.py with Chunk and Atom models**

```python
# src/isotopedb/models/chunk.py
"""Chunk and Atom data models."""

from uuid import uuid4

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A piece of content that can answer questions."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str
    source: str
    metadata: dict = Field(default_factory=dict)


class Atom(BaseModel):
    """An atomic statement extracted from a chunk."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str
    chunk_id: str
```

**Step 5: Create tests/models/__init__.py**

```python
"""Model tests."""
```

**Step 6: Run tests to verify they pass**

Run: `pytest tests/models/test_chunk.py -v`
Expected: All 4 tests PASS

**Step 7: Commit**

```bash
git add src/isotopedb/models/ tests/models/
git commit -m "feat: add Chunk and Atom data models"
```

---

### Task 1.3: Data Models - Question

**Files:**
- Modify: `src/isotopedb/models/chunk.py`
- Modify: `src/isotopedb/models/__init__.py`
- Create: `tests/models/test_question.py`

**Step 1: Write failing test for Question model**

```python
# tests/models/test_question.py
"""Tests for Question model."""

import pytest
from isotopedb.models.chunk import Question


class TestQuestion:
    def test_create_question_minimal(self):
        q = Question(text="What is Python?", chunk_id="chunk-123")
        assert q.text == "What is Python?"
        assert q.chunk_id == "chunk-123"
        assert q.id  # auto-generated
        assert q.atom_id is None
        assert q.embedding is None

    def test_create_question_with_atom_id(self):
        q = Question(
            text="What is Python?",
            chunk_id="chunk-123",
            atom_id="atom-456",
        )
        assert q.atom_id == "atom-456"

    def test_create_question_with_embedding(self):
        embedding = [0.1, 0.2, 0.3]
        q = Question(
            text="What is Python?",
            chunk_id="chunk-123",
            embedding=embedding,
        )
        assert q.embedding == embedding

    def test_question_id_is_unique(self):
        q1 = Question(text="Q?", chunk_id="c1")
        q2 = Question(text="Q?", chunk_id="c1")
        assert q1.id != q2.id
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/models/test_question.py -v`
Expected: FAIL with `ImportError: cannot import name 'Question'`

**Step 3: Add Question model to chunk.py**

```python
# Add to src/isotopedb/models/chunk.py after Atom class

class Question(BaseModel):
    """An atomic question that an atom/chunk answers."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    text: str
    chunk_id: str
    atom_id: str | None = None
    embedding: list[float] | None = None
```

**Step 4: Update models/__init__.py**

```python
# src/isotopedb/models/__init__.py
"""Data models for Isotope."""

from isotopedb.models.chunk import Atom, Chunk, Question

__all__ = ["Chunk", "Atom", "Question"]
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/models/test_question.py -v`
Expected: All 4 tests PASS

**Step 6: Commit**

```bash
git add src/isotopedb/models/
git commit -m "feat: add Question data model"
```

---

### Task 1.4: Data Models - Results

**Files:**
- Create: `src/isotopedb/models/results.py`
- Modify: `src/isotopedb/models/__init__.py`
- Create: `tests/models/test_results.py`

**Step 1: Write failing test for result models**

```python
# tests/models/test_results.py
"""Tests for SearchResult and QueryResponse models."""

import pytest
from isotopedb.models.chunk import Chunk, Question
from isotopedb.models.results import QueryResponse, SearchResult


class TestSearchResult:
    def test_create_search_result(self):
        chunk = Chunk(content="Python is great", source="test.md")
        question = Question(text="What is Python?", chunk_id=chunk.id)
        result = SearchResult(question=question, chunk=chunk, score=0.95)

        assert result.question == question
        assert result.chunk == chunk
        assert result.score == 0.95


class TestQueryResponse:
    def test_create_query_response_with_answer(self):
        chunk = Chunk(content="Python is great", source="test.md")
        question = Question(text="What is Python?", chunk_id=chunk.id)
        result = SearchResult(question=question, chunk=chunk, score=0.95)

        response = QueryResponse(
            query="Tell me about Python",
            answer="Python is a programming language.",
            results=[result],
        )
        assert response.query == "Tell me about Python"
        assert response.answer == "Python is a programming language."
        assert len(response.results) == 1

    def test_create_query_response_raw_mode(self):
        response = QueryResponse(
            query="Tell me about Python",
            answer=None,
            results=[],
        )
        assert response.answer is None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/models/test_results.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'isotopedb.models.results'`

**Step 3: Create results.py**

```python
# src/isotopedb/models/results.py
"""Result data models for Isotope queries."""

from pydantic import BaseModel

from isotopedb.models.chunk import Chunk, Question


class SearchResult(BaseModel):
    """A single matched question + its chunk."""

    question: Question
    chunk: Chunk
    score: float


class QueryResponse(BaseModel):
    """Full response to a user query."""

    query: str
    answer: str | None
    results: list[SearchResult]
```

**Step 4: Update models/__init__.py**

```python
# src/isotopedb/models/__init__.py
"""Data models for Isotope."""

from isotopedb.models.chunk import Atom, Chunk, Question
from isotopedb.models.results import QueryResponse, SearchResult

__all__ = ["Chunk", "Atom", "Question", "SearchResult", "QueryResponse"]
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/models/test_results.py -v`
Expected: All 3 tests PASS

**Step 6: Commit**

```bash
git add src/isotopedb/models/
git commit -m "feat: add SearchResult and QueryResponse models"
```

---

### Task 1.5: Configuration with pydantic-settings

**Files:**
- Create: `src/isotopedb/config.py`
- Create: `tests/test_config.py`

**Step 1: Write failing test for Settings**

```python
# tests/test_config.py
"""Tests for configuration."""

import os
import pytest
from isotopedb.config import Settings


class TestSettings:
    def test_default_settings(self):
        settings = Settings()
        assert settings.llm_model == "gpt-4o-mini"
        assert settings.embedding_model == "text-embedding-3-small"
        assert settings.atomizer == "sentence"
        assert settings.questions_per_atom == 5
        assert settings.question_diversity_threshold == 0.85
        assert settings.data_dir == "./isotope_data"
        assert settings.vector_store == "chroma"
        assert settings.doc_store == "sqlite"
        assert settings.dedup_strategy == "source_aware"
        assert settings.default_k == 5

    def test_settings_from_env(self, monkeypatch):
        monkeypatch.setenv("ISOTOPE_LLM_MODEL", "gemini/gemini-1.5-flash")
        monkeypatch.setenv("ISOTOPE_QUESTIONS_PER_ATOM", "10")
        monkeypatch.setenv("ISOTOPE_ATOMIZER", "llm")

        settings = Settings()
        assert settings.llm_model == "gemini/gemini-1.5-flash"
        assert settings.questions_per_atom == 10
        assert settings.atomizer == "llm"

    def test_question_diversity_threshold_none(self, monkeypatch):
        monkeypatch.setenv("ISOTOPE_QUESTION_DIVERSITY_THRESHOLD", "")
        settings = Settings()
        # Empty string should be treated as None/disabled
        # We'll handle this with a validator

    def test_custom_question_prompt(self, monkeypatch):
        custom_prompt = "Generate questions about: {atom}"
        monkeypatch.setenv("ISOTOPE_QUESTION_PROMPT", custom_prompt)
        settings = Settings()
        assert settings.question_prompt == custom_prompt
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'isotopedb.config'`

**Step 3: Create config.py**

```python
# src/isotopedb/config.py
"""Configuration management for Isotope."""

from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Isotope configuration settings."""

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
    question_prompt: str | None = None

    # Question diversity deduplication
    question_diversity_threshold: float | None = 0.85

    # Storage
    data_dir: str = "./isotope_data"
    vector_store: Literal["chroma"] = "chroma"
    doc_store: Literal["sqlite"] = "sqlite"

    # Re-ingestion deduplication
    dedup_strategy: Literal["none", "source_aware"] = "source_aware"

    # Retrieval
    default_k: int = 5

    @field_validator("question_diversity_threshold", mode="before")
    @classmethod
    def parse_threshold(cls, v):
        if v == "" or v is None:
            return None
        return float(v)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_config.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/isotopedb/config.py tests/test_config.py
git commit -m "feat: add Settings configuration with pydantic-settings"
```

---

### Task 1.6: Abstract Base Classes - Storage

**Files:**
- Create: `src/isotopedb/stores/__init__.py`
- Create: `src/isotopedb/stores/base.py`
- Create: `tests/stores/__init__.py`
- Create: `tests/stores/test_base.py`

**Step 1: Write test to verify ABCs are correctly defined**

```python
# tests/stores/test_base.py
"""Tests for storage abstract base classes."""

import pytest
from abc import ABC

from isotopedb.stores.base import DocStore, VectorStore
from isotopedb.models import Chunk, Question


class TestVectorStoreABC:
    def test_is_abstract(self):
        assert issubclass(VectorStore, ABC)

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            VectorStore()

    def test_has_required_methods(self):
        assert hasattr(VectorStore, "add")
        assert hasattr(VectorStore, "search")
        assert hasattr(VectorStore, "delete_by_chunk_ids")
        assert hasattr(VectorStore, "list_chunk_ids")


class TestDocStoreABC:
    def test_is_abstract(self):
        assert issubclass(DocStore, ABC)

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            DocStore()

    def test_has_required_methods(self):
        assert hasattr(DocStore, "put")
        assert hasattr(DocStore, "get")
        assert hasattr(DocStore, "get_many")
        assert hasattr(DocStore, "delete")
        assert hasattr(DocStore, "list_sources")
        assert hasattr(DocStore, "get_by_source")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/stores/test_base.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Create stores/__init__.py**

```python
# src/isotopedb/stores/__init__.py
"""Storage abstractions for Isotope."""

from isotopedb.stores.base import DocStore, VectorStore

__all__ = ["VectorStore", "DocStore"]
```

**Step 4: Create stores/base.py with ABCs**

```python
# src/isotopedb/stores/base.py
"""Abstract base classes for storage."""

from abc import ABC, abstractmethod

from isotopedb.models import Chunk, Question


class VectorStore(ABC):
    """Abstract base class for vector storage."""

    @abstractmethod
    def add(self, questions: list[Question]) -> None:
        """Add questions with embeddings to the store."""
        ...

    @abstractmethod
    def search(self, embedding: list[float], k: int = 5) -> list[tuple[Question, float]]:
        """Search for similar questions. Returns (Question, score) pairs ordered by relevance."""
        ...

    @abstractmethod
    def delete_by_chunk_ids(self, chunk_ids: list[str]) -> None:
        """Delete all questions associated with the given chunk IDs."""
        ...

    @abstractmethod
    def list_chunk_ids(self) -> set[str]:
        """List all unique chunk IDs in the store."""
        ...


class DocStore(ABC):
    """Abstract base class for document storage."""

    @abstractmethod
    def put(self, chunk: Chunk) -> None:
        """Store a chunk."""
        ...

    @abstractmethod
    def get(self, chunk_id: str) -> Chunk | None:
        """Retrieve a chunk by ID. Returns None if not found."""
        ...

    @abstractmethod
    def get_many(self, chunk_ids: list[str]) -> list[Chunk]:
        """Retrieve multiple chunks by ID. Skips missing chunks."""
        ...

    @abstractmethod
    def delete(self, chunk_id: str) -> None:
        """Delete a chunk by ID."""
        ...

    @abstractmethod
    def list_sources(self) -> list[str]:
        """List all unique sources in the store."""
        ...

    @abstractmethod
    def get_by_source(self, source: str) -> list[Chunk]:
        """Get all chunks from a specific source."""
        ...
```

**Step 5: Create tests/stores/__init__.py**

```python
"""Store tests."""
```

**Step 6: Run tests to verify they pass**

Run: `pytest tests/stores/test_base.py -v`
Expected: All tests PASS

**Step 7: Commit**

```bash
git add src/isotopedb/stores/ tests/stores/
git commit -m "feat: add VectorStore and DocStore abstract base classes"
```

---

### Task 1.7: Abstract Base Classes - Atomizer, Deduplicator, Loader

**Files:**
- Create: `src/isotopedb/atomizer/__init__.py`
- Create: `src/isotopedb/atomizer/base.py`
- Create: `src/isotopedb/dedup/__init__.py`
- Create: `src/isotopedb/dedup/base.py`
- Create: `src/isotopedb/loaders/__init__.py`
- Create: `src/isotopedb/loaders/base.py`
- Create: `tests/test_abcs.py`

**Step 1: Write test for all remaining ABCs**

```python
# tests/test_abcs.py
"""Tests for remaining abstract base classes."""

import pytest
from abc import ABC

from isotopedb.atomizer.base import Atomizer
from isotopedb.dedup.base import Deduplicator
from isotopedb.loaders.base import Loader


class TestAtomizerABC:
    def test_is_abstract(self):
        assert issubclass(Atomizer, ABC)

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            Atomizer()

    def test_has_atomize_method(self):
        assert hasattr(Atomizer, "atomize")


class TestDeduplicatorABC:
    def test_is_abstract(self):
        assert issubclass(Deduplicator, ABC)

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            Deduplicator()

    def test_has_get_chunks_to_remove_method(self):
        assert hasattr(Deduplicator, "get_chunks_to_remove")


class TestLoaderABC:
    def test_is_abstract(self):
        assert issubclass(Loader, ABC)

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            Loader()

    def test_has_required_methods(self):
        assert hasattr(Loader, "load")
        assert hasattr(Loader, "supports")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_abcs.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Create atomizer/base.py**

```python
# src/isotopedb/atomizer/base.py
"""Atomizer abstract base class."""

from abc import ABC, abstractmethod

from isotopedb.models import Atom, Chunk


class Atomizer(ABC):
    """Abstract base class for atomization."""

    @abstractmethod
    def atomize(self, chunk: Chunk) -> list[Atom]:
        """Break a chunk into atomic statements."""
        ...
```

**Step 4: Create atomizer/__init__.py**

```python
# src/isotopedb/atomizer/__init__.py
"""Atomization for Isotope."""

from isotopedb.atomizer.base import Atomizer

__all__ = ["Atomizer"]
```

**Step 5: Create dedup/base.py**

```python
# src/isotopedb/dedup/base.py
"""Deduplicator abstract base class."""

from abc import ABC, abstractmethod

from isotopedb.models import Chunk
from isotopedb.stores.base import DocStore


class Deduplicator(ABC):
    """Abstract base class for re-ingestion deduplication."""

    @abstractmethod
    def get_chunks_to_remove(
        self,
        new_chunks: list[Chunk],
        doc_store: DocStore,
    ) -> list[str]:
        """
        Determine which existing chunks should be removed before ingesting new chunks.

        Returns chunk IDs to remove.
        """
        ...
```

**Step 6: Create dedup/__init__.py**

```python
# src/isotopedb/dedup/__init__.py
"""Deduplication for Isotope."""

from isotopedb.dedup.base import Deduplicator

__all__ = ["Deduplicator"]
```

**Step 7: Create loaders/base.py**

```python
# src/isotopedb/loaders/base.py
"""Loader abstract base class."""

from abc import ABC, abstractmethod

from isotopedb.models import Chunk


class Loader(ABC):
    """Abstract base class for file loading."""

    @abstractmethod
    def load(self, path: str) -> list[Chunk]:
        """Load a file and return chunks."""
        ...

    @abstractmethod
    def supports(self, path: str) -> bool:
        """Check if this loader supports the given path."""
        ...
```

**Step 8: Create loaders/__init__.py**

```python
# src/isotopedb/loaders/__init__.py
"""File loaders for Isotope."""

from isotopedb.loaders.base import Loader

__all__ = ["Loader"]
```

**Step 9: Run tests to verify they pass**

Run: `pytest tests/test_abcs.py -v`
Expected: All tests PASS

**Step 10: Commit**

```bash
git add src/isotopedb/atomizer/ src/isotopedb/dedup/ src/isotopedb/loaders/ tests/test_abcs.py
git commit -m "feat: add Atomizer, Deduplicator, and Loader ABCs"
```

---

## Phase 2: Storage Implementations

### Task 2.1: SQLite DocStore

**Files:**
- Create: `src/isotopedb/stores/sqlite.py`
- Modify: `src/isotopedb/stores/__init__.py`
- Create: `tests/stores/test_sqlite.py`

**Step 1: Write failing tests for SQLiteDocStore**

```python
# tests/stores/test_sqlite.py
"""Tests for SQLite document store."""

import pytest
import tempfile
import os
from pathlib import Path

from isotopedb.stores.sqlite import SQLiteDocStore
from isotopedb.stores.base import DocStore
from isotopedb.models import Chunk


@pytest.fixture
def temp_db():
    """Create a temporary database file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield os.path.join(tmpdir, "test.db")


@pytest.fixture
def doc_store(temp_db):
    """Create a SQLiteDocStore instance."""
    return SQLiteDocStore(temp_db)


class TestSQLiteDocStore:
    def test_is_docstore(self, doc_store):
        assert isinstance(doc_store, DocStore)

    def test_put_and_get(self, doc_store):
        chunk = Chunk(content="Hello world", source="test.md")
        doc_store.put(chunk)

        retrieved = doc_store.get(chunk.id)
        assert retrieved is not None
        assert retrieved.id == chunk.id
        assert retrieved.content == "Hello world"
        assert retrieved.source == "test.md"

    def test_get_nonexistent(self, doc_store):
        result = doc_store.get("nonexistent-id")
        assert result is None

    def test_get_many(self, doc_store):
        chunks = [
            Chunk(content="One", source="a.md"),
            Chunk(content="Two", source="b.md"),
            Chunk(content="Three", source="c.md"),
        ]
        for c in chunks:
            doc_store.put(c)

        retrieved = doc_store.get_many([chunks[0].id, chunks[2].id])
        assert len(retrieved) == 2
        contents = {c.content for c in retrieved}
        assert contents == {"One", "Three"}

    def test_get_many_skips_missing(self, doc_store):
        chunk = Chunk(content="Exists", source="x.md")
        doc_store.put(chunk)

        retrieved = doc_store.get_many([chunk.id, "missing-id"])
        assert len(retrieved) == 1
        assert retrieved[0].id == chunk.id

    def test_delete(self, doc_store):
        chunk = Chunk(content="To delete", source="d.md")
        doc_store.put(chunk)
        assert doc_store.get(chunk.id) is not None

        doc_store.delete(chunk.id)
        assert doc_store.get(chunk.id) is None

    def test_list_sources(self, doc_store):
        doc_store.put(Chunk(content="A", source="file1.md"))
        doc_store.put(Chunk(content="B", source="file2.md"))
        doc_store.put(Chunk(content="C", source="file1.md"))

        sources = doc_store.list_sources()
        assert set(sources) == {"file1.md", "file2.md"}

    def test_get_by_source(self, doc_store):
        doc_store.put(Chunk(content="A", source="target.md"))
        doc_store.put(Chunk(content="B", source="other.md"))
        doc_store.put(Chunk(content="C", source="target.md"))

        results = doc_store.get_by_source("target.md")
        assert len(results) == 2
        assert all(c.source == "target.md" for c in results)

    def test_metadata_preserved(self, doc_store):
        chunk = Chunk(
            content="PDF content",
            source="doc.pdf",
            metadata={"page": 5, "type": "pdf", "author": "Test"},
        )
        doc_store.put(chunk)

        retrieved = doc_store.get(chunk.id)
        assert retrieved.metadata["page"] == 5
        assert retrieved.metadata["type"] == "pdf"
        assert retrieved.metadata["author"] == "Test"

    def test_put_overwrites_existing(self, doc_store):
        chunk = Chunk(id="same-id", content="Original", source="x.md")
        doc_store.put(chunk)

        updated = Chunk(id="same-id", content="Updated", source="x.md")
        doc_store.put(updated)

        retrieved = doc_store.get("same-id")
        assert retrieved.content == "Updated"
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/stores/test_sqlite.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Implement SQLiteDocStore**

```python
# src/isotopedb/stores/sqlite.py
"""SQLite document store implementation."""

import json
import sqlite3
from pathlib import Path

from isotopedb.models import Chunk
from isotopedb.stores.base import DocStore


class SQLiteDocStore(DocStore):
    """SQLite-based document store."""

    def __init__(self, db_path: str) -> None:
        """Initialize the SQLite store."""
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Create tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    source TEXT NOT NULL,
                    metadata TEXT NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_source ON chunks(source)")
            conn.commit()

    def put(self, chunk: Chunk) -> None:
        """Store a chunk, overwriting if exists."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO chunks (id, content, source, metadata)
                VALUES (?, ?, ?, ?)
                """,
                (chunk.id, chunk.content, chunk.source, json.dumps(chunk.metadata)),
            )
            conn.commit()

    def get(self, chunk_id: str) -> Chunk | None:
        """Retrieve a chunk by ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT id, content, source, metadata FROM chunks WHERE id = ?",
                (chunk_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return Chunk(
                id=row[0],
                content=row[1],
                source=row[2],
                metadata=json.loads(row[3]),
            )

    def get_many(self, chunk_ids: list[str]) -> list[Chunk]:
        """Retrieve multiple chunks by ID."""
        if not chunk_ids:
            return []
        placeholders = ",".join("?" * len(chunk_ids))
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                f"SELECT id, content, source, metadata FROM chunks WHERE id IN ({placeholders})",
                chunk_ids,
            )
            return [
                Chunk(id=row[0], content=row[1], source=row[2], metadata=json.loads(row[3]))
                for row in cursor.fetchall()
            ]

    def delete(self, chunk_id: str) -> None:
        """Delete a chunk by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM chunks WHERE id = ?", (chunk_id,))
            conn.commit()

    def list_sources(self) -> list[str]:
        """List all unique sources."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT DISTINCT source FROM chunks")
            return [row[0] for row in cursor.fetchall()]

    def get_by_source(self, source: str) -> list[Chunk]:
        """Get all chunks from a specific source."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT id, content, source, metadata FROM chunks WHERE source = ?",
                (source,),
            )
            return [
                Chunk(id=row[0], content=row[1], source=row[2], metadata=json.loads(row[3]))
                for row in cursor.fetchall()
            ]
```

**Step 4: Update stores/__init__.py**

```python
# src/isotopedb/stores/__init__.py
"""Storage abstractions for Isotope."""

from isotopedb.stores.base import DocStore, VectorStore
from isotopedb.stores.sqlite import SQLiteDocStore

__all__ = ["VectorStore", "DocStore", "SQLiteDocStore"]
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/stores/test_sqlite.py -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add src/isotopedb/stores/
git commit -m "feat: add SQLiteDocStore implementation"
```

---

### Task 2.2: ChromaDB VectorStore

**Files:**
- Create: `src/isotopedb/stores/chroma.py`
- Modify: `src/isotopedb/stores/__init__.py`
- Create: `tests/stores/test_chroma.py`

**Step 1: Write failing tests for ChromaVectorStore**

```python
# tests/stores/test_chroma.py
"""Tests for ChromaDB vector store."""

import pytest
import tempfile

from isotopedb.stores.chroma import ChromaVectorStore
from isotopedb.stores.base import VectorStore
from isotopedb.models import Question


@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def vector_store(temp_dir):
    """Create a ChromaVectorStore instance."""
    return ChromaVectorStore(temp_dir)


class TestChromaVectorStore:
    def test_is_vectorstore(self, vector_store):
        assert isinstance(vector_store, VectorStore)

    def test_add_and_search(self, vector_store):
        # Create questions with embeddings
        q1 = Question(
            text="What is Python?",
            chunk_id="chunk-1",
            embedding=[1.0, 0.0, 0.0],
        )
        q2 = Question(
            text="What is JavaScript?",
            chunk_id="chunk-2",
            embedding=[0.0, 1.0, 0.0],
        )
        vector_store.add([q1, q2])

        # Search with embedding similar to q1
        results = vector_store.search([0.9, 0.1, 0.0], k=2)
        assert len(results) == 2
        # First result should be closest to search vector
        assert results[0][0].text == "What is Python?"
        assert results[0][1] > results[1][1]  # Higher score = closer match

    def test_search_returns_question_objects(self, vector_store):
        q = Question(
            text="Test question?",
            chunk_id="c1",
            atom_id="a1",
            embedding=[1.0, 0.0, 0.0],
        )
        vector_store.add([q])

        results = vector_store.search([1.0, 0.0, 0.0], k=1)
        assert len(results) == 1
        question, score = results[0]
        assert question.text == "Test question?"
        assert question.chunk_id == "c1"
        assert question.atom_id == "a1"

    def test_delete_by_chunk_ids(self, vector_store):
        questions = [
            Question(text="Q1", chunk_id="c1", embedding=[1.0, 0.0, 0.0]),
            Question(text="Q2", chunk_id="c1", embedding=[0.0, 1.0, 0.0]),
            Question(text="Q3", chunk_id="c2", embedding=[0.0, 0.0, 1.0]),
        ]
        vector_store.add(questions)

        vector_store.delete_by_chunk_ids(["c1"])

        # Only c2 should remain
        results = vector_store.search([0.0, 0.0, 1.0], k=10)
        assert len(results) == 1
        assert results[0][0].chunk_id == "c2"

    def test_list_chunk_ids(self, vector_store):
        questions = [
            Question(text="Q1", chunk_id="c1", embedding=[1.0, 0.0, 0.0]),
            Question(text="Q2", chunk_id="c2", embedding=[0.0, 1.0, 0.0]),
            Question(text="Q3", chunk_id="c1", embedding=[0.0, 0.0, 1.0]),
        ]
        vector_store.add(questions)

        chunk_ids = vector_store.list_chunk_ids()
        assert chunk_ids == {"c1", "c2"}

    def test_search_empty_store(self, vector_store):
        results = vector_store.search([1.0, 0.0, 0.0], k=5)
        assert results == []

    def test_add_requires_embeddings(self, vector_store):
        q = Question(text="No embedding", chunk_id="c1")
        with pytest.raises(ValueError, match="embedding"):
            vector_store.add([q])
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/stores/test_chroma.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Implement ChromaVectorStore**

```python
# src/isotopedb/stores/chroma.py
"""ChromaDB vector store implementation."""

from pathlib import Path

import chromadb

from isotopedb.models import Question
from isotopedb.stores.base import VectorStore


class ChromaVectorStore(VectorStore):
    """ChromaDB-based vector store."""

    def __init__(self, persist_dir: str, collection_name: str = "isotope") -> None:
        """Initialize the ChromaDB store."""
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, questions: list[Question]) -> None:
        """Add questions with embeddings to the store."""
        if not questions:
            return

        for q in questions:
            if q.embedding is None:
                raise ValueError(f"Question {q.id} has no embedding")

        self._collection.add(
            ids=[q.id for q in questions],
            embeddings=[q.embedding for q in questions],
            metadatas=[
                {
                    "text": q.text,
                    "chunk_id": q.chunk_id,
                    "atom_id": q.atom_id or "",
                }
                for q in questions
            ],
        )

    def search(self, embedding: list[float], k: int = 5) -> list[tuple[Question, float]]:
        """Search for similar questions."""
        if self._collection.count() == 0:
            return []

        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(k, self._collection.count()),
            include=["metadatas", "distances", "embeddings"],
        )

        questions_with_scores = []
        ids = results["ids"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        embeddings = results["embeddings"][0] if results["embeddings"] else [None] * len(ids)

        for qid, meta, dist, emb in zip(ids, metadatas, distances, embeddings):
            question = Question(
                id=qid,
                text=meta["text"],
                chunk_id=meta["chunk_id"],
                atom_id=meta["atom_id"] if meta["atom_id"] else None,
                embedding=emb,
            )
            # ChromaDB returns distance; convert to similarity score
            # For cosine distance: similarity = 1 - distance
            score = 1.0 - dist
            questions_with_scores.append((question, score))

        return questions_with_scores

    def delete_by_chunk_ids(self, chunk_ids: list[str]) -> None:
        """Delete all questions associated with the given chunk IDs."""
        if not chunk_ids:
            return

        self._collection.delete(where={"chunk_id": {"$in": chunk_ids}})

    def list_chunk_ids(self) -> set[str]:
        """List all unique chunk IDs in the store."""
        if self._collection.count() == 0:
            return set()

        results = self._collection.get(include=["metadatas"])
        return {meta["chunk_id"] for meta in results["metadatas"]}
```

**Step 4: Update stores/__init__.py**

```python
# src/isotopedb/stores/__init__.py
"""Storage abstractions for Isotope."""

from isotopedb.stores.base import DocStore, VectorStore
from isotopedb.stores.chroma import ChromaVectorStore
from isotopedb.stores.sqlite import SQLiteDocStore

__all__ = ["VectorStore", "DocStore", "SQLiteDocStore", "ChromaVectorStore"]
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/stores/test_chroma.py -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add src/isotopedb/stores/
git commit -m "feat: add ChromaVectorStore implementation"
```

---

### Task 2.3: Deduplication Strategies

**Files:**
- Create: `src/isotopedb/dedup/none.py`
- Create: `src/isotopedb/dedup/source.py`
- Modify: `src/isotopedb/dedup/__init__.py`
- Create: `tests/dedup/__init__.py`
- Create: `tests/dedup/test_dedup.py`

**Step 1: Write failing tests for deduplication strategies**

```python
# tests/dedup/test_dedup.py
"""Tests for deduplication strategies."""

import pytest
from unittest.mock import Mock

from isotopedb.dedup.none import NoDedup
from isotopedb.dedup.source import SourceAwareDedup
from isotopedb.dedup.base import Deduplicator
from isotopedb.models import Chunk


class TestNoDedup:
    def test_is_deduplicator(self):
        assert isinstance(NoDedup(), Deduplicator)

    def test_returns_empty_list(self):
        dedup = NoDedup()
        chunks = [Chunk(content="A", source="x.md")]
        mock_store = Mock()

        result = dedup.get_chunks_to_remove(chunks, mock_store)
        assert result == []


class TestSourceAwareDedup:
    def test_is_deduplicator(self):
        assert isinstance(SourceAwareDedup(), Deduplicator)

    def test_returns_existing_chunk_ids_for_same_source(self):
        dedup = SourceAwareDedup()

        # New chunks being ingested
        new_chunks = [
            Chunk(id="new-1", content="New A", source="file.md"),
            Chunk(id="new-2", content="New B", source="file.md"),
        ]

        # Existing chunks in store
        existing_chunks = [
            Chunk(id="old-1", content="Old A", source="file.md"),
            Chunk(id="old-2", content="Old B", source="file.md"),
        ]

        mock_store = Mock()
        mock_store.get_by_source.return_value = existing_chunks

        result = dedup.get_chunks_to_remove(new_chunks, mock_store)
        assert set(result) == {"old-1", "old-2"}
        mock_store.get_by_source.assert_called_once_with("file.md")

    def test_handles_multiple_sources(self):
        dedup = SourceAwareDedup()

        new_chunks = [
            Chunk(id="new-1", content="A", source="a.md"),
            Chunk(id="new-2", content="B", source="b.md"),
        ]

        def mock_get_by_source(source):
            if source == "a.md":
                return [Chunk(id="old-a", content="Old A", source="a.md")]
            return []

        mock_store = Mock()
        mock_store.get_by_source.side_effect = mock_get_by_source

        result = dedup.get_chunks_to_remove(new_chunks, mock_store)
        assert result == ["old-a"]

    def test_returns_empty_for_new_source(self):
        dedup = SourceAwareDedup()

        new_chunks = [Chunk(id="new-1", content="A", source="new.md")]

        mock_store = Mock()
        mock_store.get_by_source.return_value = []

        result = dedup.get_chunks_to_remove(new_chunks, mock_store)
        assert result == []
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/dedup/test_dedup.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Create tests/dedup/__init__.py**

```python
"""Dedup tests."""
```

**Step 4: Create none.py**

```python
# src/isotopedb/dedup/none.py
"""No-op deduplication strategy."""

from isotopedb.dedup.base import Deduplicator
from isotopedb.models import Chunk
from isotopedb.stores.base import DocStore


class NoDedup(Deduplicator):
    """No deduplication - always add new chunks without removing old ones."""

    def get_chunks_to_remove(
        self,
        new_chunks: list[Chunk],
        doc_store: DocStore,
    ) -> list[str]:
        """Return empty list - don't remove anything."""
        return []
```

**Step 5: Create source.py**

```python
# src/isotopedb/dedup/source.py
"""Source-aware deduplication strategy."""

from isotopedb.dedup.base import Deduplicator
from isotopedb.models import Chunk
from isotopedb.stores.base import DocStore


class SourceAwareDedup(Deduplicator):
    """Remove all existing chunks from the same source before re-ingesting."""

    def get_chunks_to_remove(
        self,
        new_chunks: list[Chunk],
        doc_store: DocStore,
    ) -> list[str]:
        """Get IDs of existing chunks from the same sources."""
        sources = {chunk.source for chunk in new_chunks}
        chunk_ids_to_remove = []

        for source in sources:
            existing = doc_store.get_by_source(source)
            chunk_ids_to_remove.extend(chunk.id for chunk in existing)

        return chunk_ids_to_remove
```

**Step 6: Update dedup/__init__.py**

```python
# src/isotopedb/dedup/__init__.py
"""Deduplication for Isotope."""

from isotopedb.dedup.base import Deduplicator
from isotopedb.dedup.none import NoDedup
from isotopedb.dedup.source import SourceAwareDedup

__all__ = ["Deduplicator", "NoDedup", "SourceAwareDedup"]
```

**Step 7: Run tests to verify they pass**

Run: `pytest tests/dedup/test_dedup.py -v`
Expected: All tests PASS

**Step 8: Commit**

```bash
git add src/isotopedb/dedup/ tests/dedup/
git commit -m "feat: add NoDedup and SourceAwareDedup strategies"
```

---

## Phase 3: Atomization

### Task 3.1: Sentence-Based Atomizer

**Files:**
- Create: `src/isotopedb/atomizer/sentence.py`
- Modify: `src/isotopedb/atomizer/__init__.py`
- Create: `tests/atomizer/__init__.py`
- Create: `tests/atomizer/test_sentence.py`

**Step 1: Write failing tests for SentenceAtomizer**

```python
# tests/atomizer/test_sentence.py
"""Tests for sentence-based atomizer."""

import pytest

from isotopedb.atomizer.sentence import SentenceAtomizer
from isotopedb.atomizer.base import Atomizer
from isotopedb.models import Chunk


class TestSentenceAtomizer:
    def test_is_atomizer(self):
        assert isinstance(SentenceAtomizer(), Atomizer)

    def test_atomize_single_sentence(self):
        atomizer = SentenceAtomizer()
        chunk = Chunk(content="Python is a programming language.", source="test.md")

        atoms = atomizer.atomize(chunk)

        assert len(atoms) == 1
        assert atoms[0].content == "Python is a programming language."
        assert atoms[0].chunk_id == chunk.id

    def test_atomize_multiple_sentences(self):
        atomizer = SentenceAtomizer()
        chunk = Chunk(
            content="Python is interpreted. It supports multiple paradigms. Python is popular.",
            source="test.md",
        )

        atoms = atomizer.atomize(chunk)

        assert len(atoms) == 3
        assert atoms[0].content == "Python is interpreted."
        assert atoms[1].content == "It supports multiple paradigms."
        assert atoms[2].content == "Python is popular."
        assert all(a.chunk_id == chunk.id for a in atoms)

    def test_handles_abbreviations(self):
        atomizer = SentenceAtomizer()
        chunk = Chunk(
            content="Dr. Smith works at Inc. Corp. He is great.",
            source="test.md",
        )

        atoms = atomizer.atomize(chunk)
        # Should handle common abbreviations
        assert len(atoms) == 2

    def test_handles_newlines(self):
        atomizer = SentenceAtomizer()
        chunk = Chunk(
            content="First sentence.\n\nSecond sentence.",
            source="test.md",
        )

        atoms = atomizer.atomize(chunk)
        assert len(atoms) == 2

    def test_unique_atom_ids(self):
        atomizer = SentenceAtomizer()
        chunk = Chunk(content="One. Two. Three.", source="test.md")

        atoms = atomizer.atomize(chunk)
        ids = [a.id for a in atoms]
        assert len(ids) == len(set(ids))  # All unique

    def test_empty_content(self):
        atomizer = SentenceAtomizer()
        chunk = Chunk(content="", source="test.md")

        atoms = atomizer.atomize(chunk)
        assert atoms == []

    def test_whitespace_only(self):
        atomizer = SentenceAtomizer()
        chunk = Chunk(content="   \n\n   ", source="test.md")

        atoms = atomizer.atomize(chunk)
        assert atoms == []
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/atomizer/test_sentence.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Create tests/atomizer/__init__.py**

```python
"""Atomizer tests."""
```

**Step 4: Implement SentenceAtomizer**

```python
# src/isotopedb/atomizer/sentence.py
"""Sentence-based atomizer implementation."""

import re

from isotopedb.atomizer.base import Atomizer
from isotopedb.models import Atom, Chunk


class SentenceAtomizer(Atomizer):
    """Break chunks into sentences as atoms."""

    # Common abbreviations that shouldn't end sentences
    ABBREVIATIONS = {
        "mr", "mrs", "ms", "dr", "prof", "sr", "jr",
        "inc", "corp", "ltd", "co", "vs", "etc", "eg", "ie",
        "st", "ave", "blvd", "rd",
    }

    def atomize(self, chunk: Chunk) -> list[Atom]:
        """Break a chunk into sentence-based atoms."""
        content = chunk.content.strip()
        if not content:
            return []

        sentences = self._split_sentences(content)
        return [
            Atom(content=sentence, chunk_id=chunk.id)
            for sentence in sentences
            if sentence.strip()
        ]

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences, handling abbreviations."""
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)

        # Protect abbreviations by temporarily replacing periods
        for abbr in self.ABBREVIATIONS:
            # Case insensitive replacement
            pattern = rf"\b({abbr})\."
            text = re.sub(pattern, r"\1<PERIOD>", text, flags=re.IGNORECASE)

        # Split on sentence-ending punctuation
        # Matches ., !, ? followed by space or end of string
        sentences = re.split(r"(?<=[.!?])\s+", text)

        # Restore periods in abbreviations
        sentences = [s.replace("<PERIOD>", ".") for s in sentences]

        # Clean up
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences
```

**Step 5: Update atomizer/__init__.py**

```python
# src/isotopedb/atomizer/__init__.py
"""Atomization for Isotope."""

from isotopedb.atomizer.base import Atomizer
from isotopedb.atomizer.sentence import SentenceAtomizer

__all__ = ["Atomizer", "SentenceAtomizer"]
```

**Step 6: Run tests to verify they pass**

Run: `pytest tests/atomizer/test_sentence.py -v`
Expected: All tests PASS

**Step 7: Commit**

```bash
git add src/isotopedb/atomizer/ tests/atomizer/
git commit -m "feat: add SentenceAtomizer for sentence-based atomization"
```

---

### Task 3.2: LLM-Based Atomizer

**Files:**
- Create: `src/isotopedb/atomizer/llm.py`
- Modify: `src/isotopedb/atomizer/__init__.py`
- Create: `tests/atomizer/test_llm_atomizer.py`

**Step 1: Write failing tests for LLMAtomizer**

```python
# tests/atomizer/test_llm_atomizer.py
"""Tests for LLM-based atomizer."""

import pytest
from unittest.mock import Mock, patch

from isotopedb.atomizer.llm import LLMAtomizer
from isotopedb.atomizer.base import Atomizer
from isotopedb.models import Chunk


class TestLLMAtomizer:
    def test_is_atomizer(self):
        atomizer = LLMAtomizer(model="gpt-4o-mini")
        assert isinstance(atomizer, Atomizer)

    @patch("isotopedb.atomizer.llm.completion")
    def test_atomize_calls_llm(self, mock_completion):
        mock_completion.return_value = Mock(
            choices=[
                Mock(
                    message=Mock(
                        content="- Python is a programming language.\n- Python is interpreted."
                    )
                )
            ]
        )

        atomizer = LLMAtomizer(model="gpt-4o-mini")
        chunk = Chunk(content="Python is an interpreted programming language.", source="test.md")

        atoms = atomizer.atomize(chunk)

        assert len(atoms) == 2
        assert atoms[0].content == "Python is a programming language."
        assert atoms[1].content == "Python is interpreted."
        assert all(a.chunk_id == chunk.id for a in atoms)
        mock_completion.assert_called_once()

    @patch("isotopedb.atomizer.llm.completion")
    def test_atomize_parses_numbered_list(self, mock_completion):
        mock_completion.return_value = Mock(
            choices=[
                Mock(
                    message=Mock(
                        content="1. First fact.\n2. Second fact.\n3. Third fact."
                    )
                )
            ]
        )

        atomizer = LLMAtomizer(model="gpt-4o-mini")
        chunk = Chunk(content="Some content", source="test.md")

        atoms = atomizer.atomize(chunk)

        assert len(atoms) == 3
        assert atoms[0].content == "First fact."
        assert atoms[1].content == "Second fact."
        assert atoms[2].content == "Third fact."

    @patch("isotopedb.atomizer.llm.completion")
    def test_atomize_handles_empty_response(self, mock_completion):
        mock_completion.return_value = Mock(
            choices=[Mock(message=Mock(content=""))]
        )

        atomizer = LLMAtomizer(model="gpt-4o-mini")
        chunk = Chunk(content="Some content", source="test.md")

        atoms = atomizer.atomize(chunk)
        assert atoms == []

    def test_custom_prompt(self):
        custom_prompt = "Extract facts from: {content}"
        atomizer = LLMAtomizer(model="gpt-4o-mini", prompt_template=custom_prompt)
        assert atomizer.prompt_template == custom_prompt
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/atomizer/test_llm_atomizer.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Implement LLMAtomizer**

```python
# src/isotopedb/atomizer/llm.py
"""LLM-based atomizer implementation."""

import re

from litellm import completion

from isotopedb.atomizer.base import Atomizer
from isotopedb.models import Atom, Chunk


DEFAULT_ATOMIZE_PROMPT = """Extract all atomic facts from the following text. Each fact should be:
- A single, self-contained statement
- Independently true without needing context from other facts
- As specific as possible

Return the facts as a bulleted list, one fact per line, starting each line with "- ".

Text:
{content}"""


class LLMAtomizer(Atomizer):
    """Use an LLM to extract atomic facts from chunks."""

    def __init__(
        self,
        model: str,
        prompt_template: str | None = None,
    ) -> None:
        """Initialize the LLM atomizer."""
        self.model = model
        self.prompt_template = prompt_template or DEFAULT_ATOMIZE_PROMPT

    def atomize(self, chunk: Chunk) -> list[Atom]:
        """Extract atomic facts from a chunk using an LLM."""
        prompt = self.prompt_template.format(content=chunk.content)

        response = completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )

        content = response.choices[0].message.content
        if not content:
            return []

        facts = self._parse_list(content)
        return [Atom(content=fact, chunk_id=chunk.id) for fact in facts if fact]

    def _parse_list(self, text: str) -> list[str]:
        """Parse a bulleted or numbered list from LLM response."""
        lines = text.strip().split("\n")
        facts = []

        for line in lines:
            line = line.strip()
            # Remove bullet points (-, *, •)
            line = re.sub(r"^[-*•]\s*", "", line)
            # Remove numbered prefixes (1., 2., etc.)
            line = re.sub(r"^\d+\.\s*", "", line)
            if line:
                facts.append(line)

        return facts
```

**Step 4: Update atomizer/__init__.py**

```python
# src/isotopedb/atomizer/__init__.py
"""Atomization for Isotope."""

from isotopedb.atomizer.base import Atomizer
from isotopedb.atomizer.llm import LLMAtomizer
from isotopedb.atomizer.sentence import SentenceAtomizer

__all__ = ["Atomizer", "SentenceAtomizer", "LLMAtomizer"]
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/atomizer/test_llm_atomizer.py -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add src/isotopedb/atomizer/
git commit -m "feat: add LLMAtomizer for LLM-based atomic fact extraction"
```

---

## Phase 4: LLM Integration

### Task 4.1: Embedder Wrapper

**Files:**
- Create: `src/isotopedb/llm/__init__.py`
- Create: `src/isotopedb/llm/embedder.py`
- Create: `tests/llm/__init__.py`
- Create: `tests/llm/test_embedder.py`

**Step 1: Write failing tests for Embedder**

```python
# tests/llm/test_embedder.py
"""Tests for embedding wrapper."""

import pytest
from unittest.mock import Mock, patch

from isotopedb.llm.embedder import Embedder


class TestEmbedder:
    def test_init_with_model(self):
        embedder = Embedder(model="text-embedding-3-small")
        assert embedder.model == "text-embedding-3-small"

    @patch("isotopedb.llm.embedder.embedding")
    def test_embed_single_text(self, mock_embedding):
        mock_embedding.return_value = Mock(
            data=[Mock(embedding=[0.1, 0.2, 0.3])]
        )

        embedder = Embedder(model="text-embedding-3-small")
        result = embedder.embed("Hello world")

        assert result == [0.1, 0.2, 0.3]
        mock_embedding.assert_called_once()

    @patch("isotopedb.llm.embedder.embedding")
    def test_embed_batch(self, mock_embedding):
        mock_embedding.return_value = Mock(
            data=[
                Mock(embedding=[0.1, 0.2, 0.3]),
                Mock(embedding=[0.4, 0.5, 0.6]),
            ]
        )

        embedder = Embedder(model="text-embedding-3-small")
        results = embedder.embed_batch(["Hello", "World"])

        assert len(results) == 2
        assert results[0] == [0.1, 0.2, 0.3]
        assert results[1] == [0.4, 0.5, 0.6]

    @patch("isotopedb.llm.embedder.embedding")
    def test_embed_empty_batch(self, mock_embedding):
        embedder = Embedder(model="text-embedding-3-small")
        results = embedder.embed_batch([])

        assert results == []
        mock_embedding.assert_not_called()
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/llm/test_embedder.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Create tests/llm/__init__.py**

```python
"""LLM tests."""
```

**Step 4: Create llm/__init__.py**

```python
# src/isotopedb/llm/__init__.py
"""LLM integration for Isotope."""

from isotopedb.llm.embedder import Embedder

__all__ = ["Embedder"]
```

**Step 5: Implement Embedder**

```python
# src/isotopedb/llm/embedder.py
"""Embedding wrapper using LiteLLM."""

from litellm import embedding


class Embedder:
    """Wrapper for generating embeddings via LiteLLM."""

    def __init__(self, model: str) -> None:
        """Initialize the embedder."""
        self.model = model

    def embed(self, text: str) -> list[float]:
        """Embed a single text."""
        result = embedding(model=self.model, input=[text])
        return result.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in a batch."""
        if not texts:
            return []
        result = embedding(model=self.model, input=texts)
        return [item.embedding for item in result.data]
```

**Step 6: Run tests to verify they pass**

Run: `pytest tests/llm/test_embedder.py -v`
Expected: All tests PASS

**Step 7: Commit**

```bash
git add src/isotopedb/llm/ tests/llm/
git commit -m "feat: add Embedder wrapper for LiteLLM embeddings"
```

---

### Task 4.2: Question Generator

**Files:**
- Create: `src/isotopedb/llm/generator.py`
- Modify: `src/isotopedb/llm/__init__.py`
- Create: `tests/llm/test_generator.py`

**Step 1: Write failing tests for QuestionGenerator**

```python
# tests/llm/test_generator.py
"""Tests for question generator."""

import pytest
from unittest.mock import Mock, patch

from isotopedb.llm.generator import QuestionGenerator
from isotopedb.models import Atom, Question


class TestQuestionGenerator:
    def test_init_with_model(self):
        gen = QuestionGenerator(model="gpt-4o-mini", questions_per_atom=5)
        assert gen.model == "gpt-4o-mini"
        assert gen.questions_per_atom == 5

    @patch("isotopedb.llm.generator.completion")
    def test_generate_questions(self, mock_completion):
        mock_completion.return_value = Mock(
            choices=[
                Mock(
                    message=Mock(
                        content="- What is Python?\n- Is Python interpreted?\n- What paradigms does Python support?"
                    )
                )
            ]
        )

        gen = QuestionGenerator(model="gpt-4o-mini", questions_per_atom=3)
        atom = Atom(content="Python is an interpreted, multi-paradigm language.", chunk_id="c1")

        questions = gen.generate(atom)

        assert len(questions) == 3
        assert questions[0].text == "What is Python?"
        assert all(q.chunk_id == "c1" for q in questions)
        assert all(q.atom_id == atom.id for q in questions)

    @patch("isotopedb.llm.generator.completion")
    def test_generate_batch(self, mock_completion):
        mock_completion.side_effect = [
            Mock(choices=[Mock(message=Mock(content="- Q1?\n- Q2?"))]),
            Mock(choices=[Mock(message=Mock(content="- Q3?\n- Q4?"))]),
        ]

        gen = QuestionGenerator(model="gpt-4o-mini", questions_per_atom=2)
        atoms = [
            Atom(content="Fact one.", chunk_id="c1"),
            Atom(content="Fact two.", chunk_id="c1"),
        ]

        questions = gen.generate_batch(atoms)

        assert len(questions) == 4
        assert questions[0].text == "Q1?"
        assert questions[2].text == "Q3?"

    @patch("isotopedb.llm.generator.completion")
    def test_custom_prompt_template(self, mock_completion):
        mock_completion.return_value = Mock(
            choices=[Mock(message=Mock(content="- Custom Q?"))]
        )

        custom_prompt = "Generate {n} questions for: {atom}"
        gen = QuestionGenerator(
            model="gpt-4o-mini",
            questions_per_atom=1,
            prompt_template=custom_prompt,
        )
        atom = Atom(content="Test", chunk_id="c1")

        gen.generate(atom)

        call_args = mock_completion.call_args
        assert "Generate 1 questions for: Test" in call_args.kwargs["messages"][0]["content"]

    @patch("isotopedb.llm.generator.completion")
    def test_handles_empty_response(self, mock_completion):
        mock_completion.return_value = Mock(
            choices=[Mock(message=Mock(content=""))]
        )

        gen = QuestionGenerator(model="gpt-4o-mini", questions_per_atom=3)
        atom = Atom(content="Test", chunk_id="c1")

        questions = gen.generate(atom)
        assert questions == []
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/llm/test_generator.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Implement QuestionGenerator**

```python
# src/isotopedb/llm/generator.py
"""Question generation using LiteLLM."""

import re

from litellm import completion

from isotopedb.models import Atom, Question


DEFAULT_QUESTION_PROMPT = """Generate {n} diverse questions that this fact answers. The questions should:
- Be natural questions a user might ask
- Vary in phrasing and perspective
- Be answerable by the given fact

Fact: {atom}

Return the questions as a bulleted list, one per line, starting each line with "- "."""


class QuestionGenerator:
    """Generate questions for atoms using an LLM."""

    def __init__(
        self,
        model: str,
        questions_per_atom: int = 5,
        prompt_template: str | None = None,
    ) -> None:
        """Initialize the question generator."""
        self.model = model
        self.questions_per_atom = questions_per_atom
        self.prompt_template = prompt_template or DEFAULT_QUESTION_PROMPT

    def generate(self, atom: Atom) -> list[Question]:
        """Generate questions for a single atom."""
        prompt = self.prompt_template.format(n=self.questions_per_atom, atom=atom.content)

        response = completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )

        content = response.choices[0].message.content
        if not content:
            return []

        question_texts = self._parse_list(content)
        return [
            Question(text=text, chunk_id=atom.chunk_id, atom_id=atom.id)
            for text in question_texts
            if text
        ]

    def generate_batch(self, atoms: list[Atom]) -> list[Question]:
        """Generate questions for multiple atoms."""
        all_questions = []
        for atom in atoms:
            all_questions.extend(self.generate(atom))
        return all_questions

    def _parse_list(self, text: str) -> list[str]:
        """Parse a bulleted or numbered list from LLM response."""
        lines = text.strip().split("\n")
        questions = []

        for line in lines:
            line = line.strip()
            # Remove bullet points
            line = re.sub(r"^[-*•]\s*", "", line)
            # Remove numbered prefixes
            line = re.sub(r"^\d+\.\s*", "", line)
            if line:
                questions.append(line)

        return questions
```

**Step 4: Update llm/__init__.py**

```python
# src/isotopedb/llm/__init__.py
"""LLM integration for Isotope."""

from isotopedb.llm.embedder import Embedder
from isotopedb.llm.generator import QuestionGenerator

__all__ = ["Embedder", "QuestionGenerator"]
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/llm/test_generator.py -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add src/isotopedb/llm/
git commit -m "feat: add QuestionGenerator for LLM-based question generation"
```

---

### Task 4.3: Question Diversity Deduplication

**Files:**
- Modify: `src/isotopedb/llm/generator.py`
- Create: `tests/llm/test_diversity_dedup.py`

**Step 1: Write failing tests for diversity deduplication**

```python
# tests/llm/test_diversity_dedup.py
"""Tests for question diversity deduplication."""

import pytest
import math

from isotopedb.llm.generator import deduplicate_questions
from isotopedb.models import Question


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b)


class TestDeduplicateQuestions:
    def test_removes_similar_questions(self):
        # Create questions with embeddings
        # q1 and q2 are very similar (cosine > 0.85)
        # q3 is different
        questions = [
            Question(text="Q1", chunk_id="c1", embedding=[1.0, 0.0, 0.0]),
            Question(text="Q2", chunk_id="c1", embedding=[0.99, 0.1, 0.0]),  # Similar to Q1
            Question(text="Q3", chunk_id="c1", embedding=[0.0, 1.0, 0.0]),  # Different
        ]
        embeddings = [q.embedding for q in questions]

        # Verify Q1 and Q2 are similar
        sim = cosine_similarity(embeddings[0], embeddings[1])
        assert sim > 0.85

        filtered_q, filtered_e = deduplicate_questions(questions, embeddings, threshold=0.85)

        # Should keep Q1 and Q3, remove Q2
        assert len(filtered_q) == 2
        texts = {q.text for q in filtered_q}
        assert "Q1" in texts
        assert "Q3" in texts

    def test_keeps_all_when_different(self):
        questions = [
            Question(text="Q1", chunk_id="c1", embedding=[1.0, 0.0, 0.0]),
            Question(text="Q2", chunk_id="c1", embedding=[0.0, 1.0, 0.0]),
            Question(text="Q3", chunk_id="c1", embedding=[0.0, 0.0, 1.0]),
        ]
        embeddings = [q.embedding for q in questions]

        filtered_q, filtered_e = deduplicate_questions(questions, embeddings, threshold=0.85)

        assert len(filtered_q) == 3

    def test_empty_input(self):
        filtered_q, filtered_e = deduplicate_questions([], [], threshold=0.85)
        assert filtered_q == []
        assert filtered_e == []

    def test_single_question(self):
        questions = [Question(text="Q1", chunk_id="c1", embedding=[1.0, 0.0, 0.0])]
        embeddings = [[1.0, 0.0, 0.0]]

        filtered_q, filtered_e = deduplicate_questions(questions, embeddings, threshold=0.85)

        assert len(filtered_q) == 1

    def test_preserves_order(self):
        # First question of each similar pair should be kept
        questions = [
            Question(text="First", chunk_id="c1", embedding=[1.0, 0.0, 0.0]),
            Question(text="Second", chunk_id="c1", embedding=[0.0, 1.0, 0.0]),
            Question(text="FirstDupe", chunk_id="c1", embedding=[0.99, 0.1, 0.0]),
        ]
        embeddings = [q.embedding for q in questions]

        filtered_q, filtered_e = deduplicate_questions(questions, embeddings, threshold=0.85)

        assert len(filtered_q) == 2
        assert filtered_q[0].text == "First"
        assert filtered_q[1].text == "Second"

    def test_adjustable_threshold(self):
        questions = [
            Question(text="Q1", chunk_id="c1", embedding=[1.0, 0.0, 0.0]),
            Question(text="Q2", chunk_id="c1", embedding=[0.8, 0.6, 0.0]),  # Sim ~0.8 with Q1
        ]
        embeddings = [q.embedding for q in questions]

        # With low threshold, should keep both
        filtered_low, _ = deduplicate_questions(questions, embeddings, threshold=0.9)
        assert len(filtered_low) == 2

        # With high threshold, should remove duplicate
        filtered_high, _ = deduplicate_questions(questions, embeddings, threshold=0.7)
        assert len(filtered_high) == 1
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/llm/test_diversity_dedup.py -v`
Expected: FAIL with `ImportError` (function doesn't exist yet)

**Step 3: Add deduplicate_questions to generator.py**

Add at the end of `src/isotopedb/llm/generator.py`:

```python
import math


def deduplicate_questions(
    questions: list[Question],
    embeddings: list[list[float]],
    threshold: float = 0.85,
) -> tuple[list[Question], list[list[float]]]:
    """
    Remove questions with pairwise cosine similarity above threshold.

    Retains the first question of each similar pair.
    Returns filtered questions and their corresponding embeddings.

    Paper finding: Retaining 50% of questions maintains max performance.
    Even 20% retention shows minimal degradation.
    """
    if not questions:
        return [], []

    n = len(questions)
    keep = [True] * n

    for i in range(n):
        if not keep[i]:
            continue
        for j in range(i + 1, n):
            if not keep[j]:
                continue
            sim = _cosine_similarity(embeddings[i], embeddings[j])
            if sim > threshold:
                keep[j] = False

    filtered_questions = [q for q, k in zip(questions, keep) if k]
    filtered_embeddings = [e for e, k in zip(embeddings, keep) if k]

    return filtered_questions, filtered_embeddings


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
```

**Step 4: Update llm/__init__.py to export the function**

```python
# src/isotopedb/llm/__init__.py
"""LLM integration for Isotope."""

from isotopedb.llm.embedder import Embedder
from isotopedb.llm.generator import QuestionGenerator, deduplicate_questions

__all__ = ["Embedder", "QuestionGenerator", "deduplicate_questions"]
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/llm/test_diversity_dedup.py -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add src/isotopedb/llm/
git commit -m "feat: add question diversity deduplication"
```

---

### Task 4.4: Update Package Exports

**Files:**
- Modify: `src/isotopedb/__init__.py`

**Step 1: Write test for public API**

```python
# tests/test_public_api.py
"""Tests for public API exports."""

import pytest


class TestPublicAPI:
    def test_models_exported(self):
        from isotopedb import Atom, Chunk, Question, QueryResponse, SearchResult

        assert Chunk is not None
        assert Atom is not None
        assert Question is not None
        assert SearchResult is not None
        assert QueryResponse is not None

    def test_config_exported(self):
        from isotopedb import Settings

        assert Settings is not None

    def test_stores_exported(self):
        from isotopedb import ChromaVectorStore, DocStore, SQLiteDocStore, VectorStore

        assert VectorStore is not None
        assert DocStore is not None
        assert ChromaVectorStore is not None
        assert SQLiteDocStore is not None

    def test_atomizers_exported(self):
        from isotopedb import Atomizer, LLMAtomizer, SentenceAtomizer

        assert Atomizer is not None
        assert SentenceAtomizer is not None
        assert LLMAtomizer is not None

    def test_dedup_exported(self):
        from isotopedb import Deduplicator, NoDedup, SourceAwareDedup

        assert Deduplicator is not None
        assert NoDedup is not None
        assert SourceAwareDedup is not None

    def test_llm_exported(self):
        from isotopedb import Embedder, QuestionGenerator, deduplicate_questions

        assert Embedder is not None
        assert QuestionGenerator is not None
        assert deduplicate_questions is not None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_public_api.py -v`
Expected: FAIL (exports not defined)

**Step 3: Update __init__.py with all exports**

```python
# src/isotopedb/__init__.py
"""Isotope - Reverse RAG database."""

__version__ = "0.1.0"

# Models
from isotopedb.models import Atom, Chunk, Question, QueryResponse, SearchResult

# Config
from isotopedb.config import Settings

# Stores
from isotopedb.stores import ChromaVectorStore, DocStore, SQLiteDocStore, VectorStore

# Atomizers
from isotopedb.atomizer import Atomizer, LLMAtomizer, SentenceAtomizer

# Deduplication
from isotopedb.dedup import Deduplicator, NoDedup, SourceAwareDedup

# LLM
from isotopedb.llm import Embedder, QuestionGenerator, deduplicate_questions

__all__ = [
    # Version
    "__version__",
    # Models
    "Chunk",
    "Atom",
    "Question",
    "SearchResult",
    "QueryResponse",
    # Config
    "Settings",
    # Stores
    "VectorStore",
    "DocStore",
    "ChromaVectorStore",
    "SQLiteDocStore",
    # Atomizers
    "Atomizer",
    "SentenceAtomizer",
    "LLMAtomizer",
    # Deduplication
    "Deduplicator",
    "NoDedup",
    "SourceAwareDedup",
    # LLM
    "Embedder",
    "QuestionGenerator",
    "deduplicate_questions",
]
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_public_api.py -v`
Expected: All tests PASS

**Step 5: Run full test suite**

Run: `pytest -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add src/isotopedb/__init__.py tests/test_public_api.py
git commit -m "feat: export public API from package root"
```

---

## Summary

This plan covers **Phases 1-4** of the Isotope implementation:

| Phase | Tasks | Components |
|-------|-------|------------|
| Phase 1 | 1.1-1.7 | Project setup, data models, config, ABCs |
| Phase 2 | 2.1-2.3 | SQLite DocStore, ChromaDB VectorStore, dedup strategies |
| Phase 3 | 3.1-3.2 | Sentence atomizer, LLM atomizer |
| Phase 4 | 4.1-4.4 | Embedder, question generator, diversity dedup, exports |

**Total tasks:** 14

After completing these phases, you'll have a solid, tested foundation with all core components. The remaining phases (5-8) cover:
- Phase 5: Ingestor and Retriever pipelines
- Phase 6: File loaders
- Phase 7: CLI
- Phase 8: Polish (additional tests, README, error handling)

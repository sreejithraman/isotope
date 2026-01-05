"""Isotope - Reverse RAG.

A question-based retrieval system using atomic units for enterprise RAG.
Aligns with arXiv:2405.12363.

Quick Start (LiteLLM + Local Storage):
    from isotope import Isotope, LiteLLMProvider, LocalStorage, LoaderRegistry

    iso = Isotope(
        provider=LiteLLMProvider(llm="openai/gpt-4o", embedding="text-embedding-3-small"),
        storage=LocalStorage("./data"),
    )

    # Ingest documents
    ingestor = iso.ingestor()
    chunks = LoaderRegistry.default().load("document.pdf")
    ingestor.ingest_chunks(chunks)

    # Query
    retriever = iso.retriever()
    response = retriever.get_answer("What is...?")

Enterprise (Explicit Stores):
    from isotope import Isotope, LiteLLMProvider
    from isotope.stores import (
        ChromaEmbeddedQuestionStore, SQLiteChunkStore, SQLiteAtomStore, SQLiteSourceRegistry
    )

    iso = Isotope.from_stores(
        provider=LiteLLMProvider(llm="gpt-4o", embedding="text-embedding-3-small"),
        embedded_question_store=ChromaEmbeddedQuestionStore("./data/chroma"),
        chunk_store=SQLiteChunkStore("./data/chunks.db"),
        atom_store=SQLiteAtomStore("./data/atoms.db"),
        source_registry=SQLiteSourceRegistry("./data/sources.db"),
    )
"""

try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("isotope-rag")
except PackageNotFoundError:
    # Development / source-tree fallback (e.g. running tests without installing the wheel).
    try:
        import tomllib
        from pathlib import Path

        def _read_version_from_pyproject() -> str | None:
            for parent in Path(__file__).resolve().parents:
                pyproject = parent / "pyproject.toml"
                if pyproject.exists():
                    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
                    version = data.get("project", {}).get("version")
                    return str(version) if version is not None else None
            return None

        __version__ = _read_version_from_pyproject() or "unknown"
    except Exception:
        __version__ = "unknown"

# Core models
# Abstract base classes
from isotope.atomizer import Atomizer, SentenceAtomizer

# Configuration objects
from isotope.configuration import (
    LiteLLMProvider,
    LocalStorage,
    ProviderConfig,
    StorageConfig,
)
from isotope.embedder import Embedder

# Pipelines
from isotope.ingestor import Ingestor

# Central configuration
from isotope.isotope import Isotope

# File loading
from isotope.loaders import Loader, LoaderRegistry, TextLoader
from isotope.models import (
    Atom,
    Chunk,
    EmbeddedQuestion,
    QueryResponse,
    Question,
    SearchResult,
)

# Provider ABCs
from isotope.providers import EmbeddingClient, LLMClient
from isotope.question_generator import DiversityFilter, FilterScope, QuestionGenerator
from isotope.retriever import Retriever

# Configuration
from isotope.settings import Settings

# Storage ABCs
from isotope.stores import (
    AtomStore,
    ChromaEmbeddedQuestionStore,
    ChunkStore,
    EmbeddedQuestionStore,
    SourceRegistry,
    SQLiteAtomStore,
    SQLiteChunkStore,
    SQLiteSourceRegistry,
)

__all__ = [
    # Version
    "__version__",
    # Models
    "Atom",
    "Chunk",
    "EmbeddedQuestion",
    "QueryResponse",
    "Question",
    "SearchResult",
    # Config
    "Settings",
    # Configuration objects
    "ProviderConfig",
    "StorageConfig",
    "LiteLLMProvider",
    "LocalStorage",
    # Storage ABCs
    "AtomStore",
    "ChromaEmbeddedQuestionStore",
    "ChunkStore",
    "EmbeddedQuestionStore",
    "SourceRegistry",
    "SQLiteAtomStore",
    "SQLiteChunkStore",
    "SQLiteSourceRegistry",
    # Atomization ABCs
    "Atomizer",
    "SentenceAtomizer",
    # Embedding ABC
    "Embedder",
    # Question generation ABC
    "DiversityFilter",
    "FilterScope",
    "QuestionGenerator",
    # Provider ABCs
    "LLMClient",
    "EmbeddingClient",
    # Pipelines
    "Ingestor",
    "Retriever",
    # Central configuration
    "Isotope",
    # File loading
    "Loader",
    "LoaderRegistry",
    "TextLoader",
]

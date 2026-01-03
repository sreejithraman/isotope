"""IsotopeDB - Reverse RAG database.

A question-based retrieval system using atomic units for enterprise RAG.
Aligns with arXiv:2405.12363.

Quick Start (LiteLLM + Local Storage):
    from isotopedb import Isotope, LiteLLMProvider, LocalStorage, LoaderRegistry

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
    from isotopedb import Isotope, LiteLLMProvider
    from isotopedb.stores import (
        ChromaVectorStore, SQLiteChunkStore, SQLiteAtomStore, SQLiteSourceRegistry
    )

    iso = Isotope(
        provider=LiteLLMProvider(llm="gpt-4o", embedding="text-embedding-3-small"),
        vector_store=ChromaVectorStore("./data/chroma"),
        chunk_store=SQLiteChunkStore("./data/chunks.db"),
        atom_store=SQLiteAtomStore("./data/atoms.db"),
        source_registry=SQLiteSourceRegistry("./data/sources.db"),
    )
"""

__version__ = "0.1.0"

# Core models
# Abstract base classes
from isotopedb.atomizer import Atomizer, SentenceAtomizer

# Configuration
from isotopedb.config import Settings

# Configuration objects
from isotopedb.configuration import (
    LiteLLMProvider,
    LocalStorage,
    ProviderConfig,
    StorageConfig,
)
from isotopedb.embedder import Embedder

# Pipelines
from isotopedb.ingestor import Ingestor

# Central configuration
from isotopedb.isotope import Isotope

# File loading
from isotopedb.loaders import Loader, LoaderRegistry, TextLoader
from isotopedb.models import (
    Atom,
    Chunk,
    EmbeddedQuestion,
    QueryResponse,
    Question,
    SearchResult,
)

# Provider ABCs
from isotopedb.providers import EmbeddingClient, LLMClient
from isotopedb.question_generator import DiversityFilter, FilterScope, QuestionGenerator
from isotopedb.retriever import Retriever

# Storage ABCs
from isotopedb.stores import (
    AtomStore,
    ChromaVectorStore,
    ChunkStore,
    SourceRegistry,
    SQLiteAtomStore,
    SQLiteChunkStore,
    SQLiteSourceRegistry,
    VectorStore,
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
    "ChromaVectorStore",
    "ChunkStore",
    "SourceRegistry",
    "SQLiteAtomStore",
    "SQLiteChunkStore",
    "SQLiteSourceRegistry",
    "VectorStore",
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

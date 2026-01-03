"""IsotopeDB - Reverse RAG database.

A question-based retrieval system using atomic units for enterprise RAG.
Aligns with arXiv:2405.12363.

Quick Start (LiteLLM):
    from isotopedb import Isotope

    iso = Isotope.with_litellm(
        llm_model="openai/gpt-4o",
        embedding_model="openai/text-embedding-3-small",
    )

    # Ingest documents
    ingestor = iso.ingestor()
    ingestor.ingest_file("document.pdf")

    # Query
    retriever = iso.retriever()
    response = retriever.get_answer("What is...?")

Custom Implementations:
    from isotopedb import Isotope, Embedder, QuestionGenerator, Atomizer
    from isotopedb.litellm import LiteLLMEmbedder  # Or your own implementation

    iso = Isotope(
        vector_store=my_vector_store,
        doc_store=my_doc_store,
        atom_store=my_atom_store,
        embedder=LiteLLMEmbedder(model="openai/text-embedding-3-small"),
    )
"""

__version__ = "0.1.0"

# Core models
from isotopedb.models import (
    Atom,
    Chunk,
    EmbeddedQuestion,
    QueryResponse,
    Question,
    SearchResult,
)

# Abstract base classes
from isotopedb.atomizer import Atomizer, SentenceAtomizer
from isotopedb.dedup import Deduplicator, NoDedup, SourceAwareDedup
from isotopedb.embedder import Embedder
from isotopedb.generator import DiversityFilter, FilterScope, QuestionGenerator

# Configuration
from isotopedb.config import Settings

# Storage ABCs
from isotopedb.stores import (
    AtomStore,
    ChromaVectorStore,
    DocStore,
    SQLiteAtomStore,
    SQLiteDocStore,
    VectorStore,
)

# Pipelines
from isotopedb.ingestor import Ingestor
from isotopedb.retriever import Retriever

# Central configuration
from isotopedb.isotope import Isotope

# File loading
from isotopedb.loaders import Loader, LoaderRegistry, TextLoader

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
    # Storage ABCs
    "AtomStore",
    "ChromaVectorStore",
    "DocStore",
    "SQLiteAtomStore",
    "SQLiteDocStore",
    "VectorStore",
    # Atomization ABCs
    "Atomizer",
    "SentenceAtomizer",
    # Deduplication
    "Deduplicator",
    "NoDedup",
    "SourceAwareDedup",
    # Embedding ABC
    "Embedder",
    # Question generation ABC
    "DiversityFilter",
    "FilterScope",
    "QuestionGenerator",
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

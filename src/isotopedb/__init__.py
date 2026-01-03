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
    from isotopedb import Isotope
    from isotopedb.litellm import LiteLLMEmbedder, LiteLLMAtomizer, LiteLLMGenerator

    iso = Isotope(
        vector_store=my_vector_store,
        doc_store=my_doc_store,
        atom_store=my_atom_store,
        embedder=LiteLLMEmbedder(model="openai/text-embedding-3-small"),
        atomizer=LiteLLMAtomizer(model="openai/gpt-4o"),
        generator=LiteLLMGenerator(model="openai/gpt-4o"),
    )
    ingestor = iso.ingestor()  # All components configured at init
"""

__version__ = "0.1.0"

# Core models
# Abstract base classes
from isotopedb.atomizer import Atomizer, SentenceAtomizer

# Configuration
from isotopedb.config import Settings
from isotopedb.dedup import Deduplicator, NoDedup, SourceAwareDedup
from isotopedb.embedder import Embedder
from isotopedb.generator import DiversityFilter, FilterScope, QuestionGenerator

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
from isotopedb.retriever import Retriever

# Storage ABCs
from isotopedb.stores import (
    AtomStore,
    ChromaVectorStore,
    DocStore,
    SQLiteAtomStore,
    SQLiteDocStore,
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

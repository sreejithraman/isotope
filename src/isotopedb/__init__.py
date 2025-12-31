"""Isotope - Reverse RAG database.

A question-based retrieval system using atomic units for enterprise RAG.
Aligns with arXiv:2405.12363.
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

# Atomization
from isotopedb.atomizer import Atomizer, LLMAtomizer, SentenceAtomizer

# Deduplication
from isotopedb.dedup import Deduplicator, NoDedup, SourceAwareDedup

# Embedding
from isotopedb.embedder import Embedder

# Question generation
from isotopedb.generator import DiversityFilter, QuestionGenerator

# Pipelines
from isotopedb.ingestor import Ingestor
from isotopedb.retriever import Retriever

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
    # Storage
    "AtomStore",
    "ChromaVectorStore",
    "DocStore",
    "SQLiteAtomStore",
    "SQLiteDocStore",
    "VectorStore",
    # Atomization
    "Atomizer",
    "LLMAtomizer",
    "SentenceAtomizer",
    # Deduplication
    "Deduplicator",
    "NoDedup",
    "SourceAwareDedup",
    # Embedding
    "Embedder",
    # Question generation
    "DiversityFilter",
    "QuestionGenerator",
    # Pipelines
    "Ingestor",
    "Retriever",
]

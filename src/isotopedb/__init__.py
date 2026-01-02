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
from isotopedb.atomizer import Atomizer, LiteLLMAtomizer, LLMAtomizer, SentenceAtomizer

# Deduplication
from isotopedb.dedup import Deduplicator, NoDedup, SourceAwareDedup

# Embedding
from isotopedb.embedder import Embedder, LiteLLMEmbedder

# Question generation
from isotopedb.generator import (
    DiversityFilter,
    LiteLLMQuestionGenerator,
    QuestionGenerator,
)

# LLM model constants
from isotopedb.llm_models import ChatModels, EmbeddingModels

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
    # Storage
    "AtomStore",
    "ChromaVectorStore",
    "DocStore",
    "SQLiteAtomStore",
    "SQLiteDocStore",
    "VectorStore",
    # Atomization
    "Atomizer",
    "LiteLLMAtomizer",
    "LLMAtomizer",  # Backwards compatibility alias
    "SentenceAtomizer",
    # Deduplication
    "Deduplicator",
    "NoDedup",
    "SourceAwareDedup",
    # Embedding
    "Embedder",
    "LiteLLMEmbedder",
    # Question generation
    "DiversityFilter",
    "LiteLLMQuestionGenerator",
    "QuestionGenerator",
    # LLM model constants
    "ChatModels",
    "EmbeddingModels",
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

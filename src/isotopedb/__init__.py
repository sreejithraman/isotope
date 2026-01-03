"""Isotope - Reverse RAG database.

A question-based retrieval system using atomic units for enterprise RAG.
Aligns with arXiv:2405.12363.
"""

__version__ = "0.1.0"

# Core models
# Atomization
from isotopedb.atomizer import Atomizer, LiteLLMAtomizer, LLMAtomizer, SentenceAtomizer

# Configuration
from isotopedb.config import Settings

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

# Pipelines
from isotopedb.ingestor import Ingestor

# Central configuration
from isotopedb.isotope import Isotope

# LLM model constants
from isotopedb.llm_models import ChatModels, EmbeddingModels

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

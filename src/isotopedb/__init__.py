"""IsotopeDB - Reverse RAG database.

A question-based retrieval system using atomic units for enterprise RAG.
Aligns with arXiv:2405.12363.

Quick Start (LiteLLM):
    from isotopedb import Isotope, LoaderRegistry

    iso = Isotope.with_litellm(
        llm_model="openai/gpt-4o",
        embedding_model="openai/text-embedding-3-small",
    )

    # Ingest documents
    ingestor = iso.ingestor()
    chunks = LoaderRegistry.default().load("document.pdf")
    ingestor.ingest_chunks(chunks)

    # Query
    retriever = iso.retriever()
    response = retriever.get_answer("What is...?")

Custom Implementations:
    from isotopedb import Isotope
    from isotopedb.atomizer import LLMAtomizer
    from isotopedb.embedder import ClientEmbedder
    from isotopedb.providers.litellm import LiteLLMClient, LiteLLMEmbeddingClient
    from isotopedb.question_generator import ClientQuestionGenerator

    llm_client = LiteLLMClient(model="openai/gpt-4o")
    embedding_client = LiteLLMEmbeddingClient(model="openai/text-embedding-3-small")

    iso = Isotope(
        vector_store=my_vector_store,
        chunk_store=my_chunk_store,
        atom_store=my_atom_store,
        embedder=ClientEmbedder(embedding_client=embedding_client),
        atomizer=LLMAtomizer(llm_client=llm_client),
        question_generator=ClientQuestionGenerator(llm_client=llm_client),
    )
    ingestor = iso.ingestor()  # All components configured at init
"""

__version__ = "0.1.0"

# Core models
# Abstract base classes
from isotopedb.atomizer import Atomizer, SentenceAtomizer

# Configuration
from isotopedb.config import Settings
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

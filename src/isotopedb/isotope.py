# src/isotopedb/isotope.py
"""Central configuration class for IsotopeDB."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isotopedb.atomizer import Atomizer
    from isotopedb.embedder import Embedder
    from isotopedb.ingestor import Ingestor
    from isotopedb.loaders import LoaderRegistry
    from isotopedb.question_generator import DiversityFilter, FilterScope, QuestionGenerator
    from isotopedb.retriever import Retriever
    from isotopedb.stores import AtomStore, ChunkStore, SourceRegistry, VectorStore

from isotopedb.config import Settings


class Isotope:
    """Central configuration for IsotopeDB stores and components.

    Isotope bundles all the stores and embedder together so you can
    configure once and create Retrievers/Ingestors from it.

    There are two ways to use Isotope:

    1. Simple path (LiteLLM):
        Use the with_litellm() factory method for quick setup:

        iso = Isotope.with_litellm(
            data_dir="./my_data",
            llm_model="openai/gpt-4o",
            embedding_model="openai/text-embedding-3-small",
        )
        ingestor = iso.ingestor()

    2. Explicit path (custom/enterprise):
        Bring your own components:

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

    def __init__(
        self,
        *,
        vector_store: VectorStore,
        chunk_store: ChunkStore,
        atom_store: AtomStore,
        source_registry: SourceRegistry,
        embedder: Embedder,
        atomizer: Atomizer,
        question_generator: QuestionGenerator,
        loader_registry: LoaderRegistry | None = None,
    ) -> None:
        """Create an Isotope instance.

        Args:
            vector_store: Vector store for question embeddings (required)
            chunk_store: Chunk store for chunks (required)
            atom_store: Atom store for atomic statements (required)
            source_registry: Registry for tracking source content hashes (required)
            embedder: Embedder for creating embeddings (required)
            atomizer: Atomizer for breaking chunks into atoms (required)
            question_generator: Question generator for creating synthetic questions (required)
            loader_registry: Optional loader registry for file loading. If None, uses default.
        """
        # Load behavioral settings from env vars
        self._settings = Settings()

        # Store references
        self.vector_store = vector_store
        self.chunk_store = chunk_store
        self.atom_store = atom_store
        self._source_registry = source_registry
        self.embedder = embedder

        # Required components for ingestor
        self._atomizer = atomizer
        self._question_generator = question_generator

        # Loader registry (lazily created if not provided)
        self._loader_registry = loader_registry

    @classmethod
    def with_litellm(
        cls,
        *,
        llm_model: str,
        embedding_model: str,
        data_dir: str = "./isotope_data",
        use_sentence_atomizer: bool = False,
    ) -> Isotope:
        """Create Isotope with LiteLLM provider and local stores.

        Convenience factory for quick setup using LiteLLM.
        Requires the litellm package to be installed.

        Args:
            llm_model: LiteLLM model for question generation and atomization.
                       Examples: "openai/gpt-4o", "anthropic/claude-sonnet-4-5-20250929"
            embedding_model: LiteLLM embedding model.
                             Examples: "openai/text-embedding-3-small"
            data_dir: Base directory for local stores (Chroma + SQLite).
            use_sentence_atomizer: If True, use SentenceAtomizer instead of LLM atomizer.

        Returns:
            Configured Isotope instance with LiteLLM components.

        Example:
            iso = Isotope.with_litellm(
                llm_model="openai/gpt-4o",
                embedding_model="openai/text-embedding-3-small",
            )
            ingestor = iso.ingestor()
        """
        from isotopedb.atomizer import LLMAtomizer, SentenceAtomizer
        from isotopedb.embedder import ClientEmbedder
        from isotopedb.providers.litellm import LiteLLMClient, LiteLLMEmbeddingClient
        from isotopedb.question_generator import ClientQuestionGenerator
        from isotopedb.stores import (
            ChromaVectorStore,
            SQLiteAtomStore,
            SQLiteChunkStore,
            SQLiteSourceRegistry,
        )

        # Create local stores
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        vector_store = ChromaVectorStore(os.path.join(data_dir, "chroma"))
        chunk_store = SQLiteChunkStore(os.path.join(data_dir, "chunks.db"))
        atom_store = SQLiteAtomStore(os.path.join(data_dir, "atoms.db"))
        source_registry = SQLiteSourceRegistry(os.path.join(data_dir, "sources.db"))

        # Load behavioral settings for generator
        settings = Settings()

        # Create LiteLLM clients
        llm_client = LiteLLMClient(model=llm_model)
        embedding_client = LiteLLMEmbeddingClient(model=embedding_model)

        # Create components with injected clients
        embedder = ClientEmbedder(embedding_client=embedding_client)
        question_generator = ClientQuestionGenerator(
            llm_client=llm_client,
            num_questions=settings.questions_per_atom,
            prompt_template=settings.question_generator_prompt,
        )

        if use_sentence_atomizer:
            atomizer: Atomizer = SentenceAtomizer()
        else:
            atomizer = LLMAtomizer(llm_client=llm_client, prompt_template=settings.atomizer_prompt)

        return cls(
            vector_store=vector_store,
            chunk_store=chunk_store,
            atom_store=atom_store,
            source_registry=source_registry,
            embedder=embedder,
            atomizer=atomizer,
            question_generator=question_generator,
        )

    @classmethod
    def with_local_stores(
        cls,
        *,
        embedder: Embedder,
        atomizer: Atomizer,
        question_generator: QuestionGenerator,
        data_dir: str = "./isotope_data",
    ) -> Isotope:
        """Create Isotope with local stores (Chroma + SQLite).

        Use this when you want to bring your own embedder/atomizer/question_generator
        but use local stores for development.

        Args:
            embedder: Embedder implementation (required)
            atomizer: Atomizer implementation (required)
            question_generator: Question generator implementation (required)
            data_dir: Base directory for all stores.

        Returns:
            Configured Isotope instance with local stores.
        """
        from isotopedb.stores import (
            ChromaVectorStore,
            SQLiteAtomStore,
            SQLiteChunkStore,
            SQLiteSourceRegistry,
        )

        Path(data_dir).mkdir(parents=True, exist_ok=True)

        return cls(
            vector_store=ChromaVectorStore(os.path.join(data_dir, "chroma")),
            chunk_store=SQLiteChunkStore(os.path.join(data_dir, "chunks.db")),
            atom_store=SQLiteAtomStore(os.path.join(data_dir, "atoms.db")),
            source_registry=SQLiteSourceRegistry(os.path.join(data_dir, "sources.db")),
            embedder=embedder,
            atomizer=atomizer,
            question_generator=question_generator,
        )

    def _get_loader_registry(self) -> LoaderRegistry:
        """Get or create the loader registry."""
        if self._loader_registry is None:
            from isotopedb.loaders import LoaderRegistry

            self._loader_registry = LoaderRegistry.default()
        return self._loader_registry

    def _create_diversity_filter(self) -> DiversityFilter | None:
        """Create diversity filter if threshold is set."""
        if self._settings.question_diversity_threshold is not None:
            from isotopedb.question_generator import DiversityFilter

            return DiversityFilter(threshold=self._settings.question_diversity_threshold)
        return None

    def retriever(
        self,
        *,
        llm_model: str | None = None,
        synthesis_prompt: str | None = None,
        default_k: int | None = None,
    ) -> Retriever:
        """Create a Retriever using this instance's stores.

        Args:
            llm_model: LiteLLM model for answer synthesis. Pass None to disable synthesis.
            synthesis_prompt: Custom synthesis prompt template.
            default_k: Number of results to return. If None, uses settings default.

        Returns:
            Configured Retriever instance.
        """
        from isotopedb.retriever import Retriever

        return Retriever(
            vector_store=self.vector_store,
            chunk_store=self.chunk_store,
            atom_store=self.atom_store,
            embedder=self.embedder,
            default_k=default_k if default_k is not None else self._settings.default_k,
            llm_model=llm_model,
            synthesis_prompt=synthesis_prompt or self._settings.synthesis_prompt,
        )

    def ingestor(
        self,
        *,
        diversity_filter: DiversityFilter | None = None,
        use_diversity_filter: bool = True,
        diversity_scope: FilterScope | None = None,
    ) -> Ingestor:
        """Create an Ingestor using this instance's stores.

        Args:
            diversity_filter: Custom diversity filter. If None and
                              use_diversity_filter is True, created from settings.
            use_diversity_filter: Whether to use diversity filter. Set False to
                                  disable even if settings has a threshold.
            diversity_scope: Scope for diversity filtering. Options:
                - "global": Filter across all questions (default, paper-validated)
                - "per_chunk": Filter within each chunk (~100x faster)
                - "per_atom": Filter within each atom (~1000x faster)
                If None, uses settings default.

        Returns:
            Configured Ingestor instance using the atomizer and generator
            provided at Isotope initialization.
        """
        from isotopedb.ingestor import Ingestor

        # Handle diversity filter
        effective_diversity_filter: DiversityFilter | None
        if diversity_filter is not None:
            effective_diversity_filter = diversity_filter
        elif use_diversity_filter:
            effective_diversity_filter = self._create_diversity_filter()
        else:
            effective_diversity_filter = None

        # Use settings default for diversity_scope if not specified
        effective_diversity_scope: FilterScope = diversity_scope or self._settings.diversity_scope

        return Ingestor(
            vector_store=self.vector_store,
            chunk_store=self.chunk_store,
            atom_store=self.atom_store,
            atomizer=self._atomizer,
            embedder=self.embedder,
            question_generator=self._question_generator,
            diversity_filter=effective_diversity_filter,
            diversity_scope=effective_diversity_scope,
        )

    def ingest_file(
        self,
        filepath: str,
        source_id: str | None = None,
        *,
        diversity_filter: DiversityFilter | None = None,
        use_diversity_filter: bool = True,
        diversity_scope: FilterScope | None = None,
    ) -> dict:
        """Ingest a file, skipping if content unchanged.

        This is the recommended way to ingest files. It:
        1. Computes a content hash to detect changes
        2. Skips ingestion if the file hasn't changed
        3. Clears old data before re-ingesting changed files

        Args:
            filepath: Path to the file to ingest
            source_id: Optional custom source identifier. If not provided,
                      the absolute path will be used as the source.
            diversity_filter: Custom diversity filter for question deduplication
            use_diversity_filter: Whether to use diversity filter
            diversity_scope: Scope for diversity filtering

        Returns:
            Dict with ingestion statistics or skip info:
            - If skipped: {"skipped": True, "reason": "..."}
            - If ingested: {"chunks": N, "atoms": N, "questions": N, ...}
        """
        file_path = Path(filepath)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        # Compute content hash
        content = file_path.read_bytes()
        content_hash = hashlib.sha256(content).hexdigest()

        # Determine source identifier
        source = source_id or str(file_path.resolve())

        # Check if content unchanged
        existing_hash = self._source_registry.get_hash(source)
        if existing_hash == content_hash:
            return {"skipped": True, "reason": "content unchanged"}

        # Delete old data for this source (cascading via chunk_ids)
        chunk_ids = self.chunk_store.get_chunk_ids_by_source(source)
        if chunk_ids:
            self.vector_store.delete_by_chunk_ids(chunk_ids)
            self.atom_store.delete_by_chunk_ids(chunk_ids)
            self.chunk_store.delete_by_source(source)
            self._source_registry.delete(source)

        # Load file into chunks
        loader_registry = self._get_loader_registry()
        chunks = loader_registry.load(filepath, source_id)
        if not chunks:
            return {"skipped": True, "reason": "no content"}

        # Ingest chunks
        ingestor = self.ingestor(
            diversity_filter=diversity_filter,
            use_diversity_filter=use_diversity_filter,
            diversity_scope=diversity_scope,
        )
        result = ingestor.ingest_chunks(chunks)

        # Track the new hash after successful ingestion
        self._source_registry.set_hash(source, content_hash)

        return result

    def delete_source(self, source: str) -> dict:
        """Delete all data for a source.

        Use this to remove a file's data from the database.

        Args:
            source: The source identifier (typically the absolute file path)

        Returns:
            Dict with deletion statistics
        """
        chunk_ids = self.chunk_store.get_chunk_ids_by_source(source)
        if not chunk_ids:
            return {"deleted": False, "reason": "source not found"}

        self.vector_store.delete_by_chunk_ids(chunk_ids)
        self.atom_store.delete_by_chunk_ids(chunk_ids)
        self.chunk_store.delete_by_source(source)
        self._source_registry.delete(source)

        return {"deleted": True, "chunks_removed": len(chunk_ids)}

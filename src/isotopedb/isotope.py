# src/isotopedb/isotope.py
"""Central configuration class for IsotopeDB."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isotopedb.configuration import ProviderConfig, StorageConfig
    from isotopedb.ingestor import Ingestor
    from isotopedb.loaders import LoaderRegistry
    from isotopedb.question_generator import DiversityFilter, FilterScope
    from isotopedb.retriever import Retriever
    from isotopedb.stores import AtomStore, ChunkStore, SourceRegistry, VectorStore

from isotopedb.config import Settings


class Isotope:
    """Central configuration for IsotopeDB stores and components.

    Isotope bundles all the stores and AI components together so you can
    configure once and create Retrievers/Ingestors from it.

    There are two ways to create an Isotope instance:

    1. With a storage bundle (developer-friendly):

        from isotopedb import Isotope, LiteLLMProvider, LocalStorage

        iso = Isotope(
            provider=LiteLLMProvider(llm="openai/gpt-4o", embedding="text-embedding-3-small"),
            storage=LocalStorage("./data"),
        )
        ingestor = iso.ingestor()

    2. With explicit stores (enterprise):

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
        ingestor = iso.ingestor()
    """

    def __init__(
        self,
        *,
        provider: ProviderConfig,
        # EITHER storage bundle...
        storage: StorageConfig | None = None,
        # ...OR explicit stores
        vector_store: VectorStore | None = None,
        chunk_store: ChunkStore | None = None,
        atom_store: AtomStore | None = None,
        source_registry: SourceRegistry | None = None,
        # Common
        settings: Settings | None = None,
        loader_registry: LoaderRegistry | None = None,
    ) -> None:
        """Create an Isotope instance.

        Args:
            provider: Provider configuration (builds embedder, atomizer, question_generator).
                      Example: LiteLLMProvider(llm="gpt-4o", embedding="text-embedding-3-small")
            storage: Storage bundle (convenience). Mutually exclusive with explicit stores.
                     Example: LocalStorage("./data")
            vector_store: Explicit vector store. Use with other explicit stores.
            chunk_store: Explicit chunk store.
            atom_store: Explicit atom store.
            source_registry: Explicit source registry.
            settings: Behavioral settings (questions_per_atom, diversity_threshold, etc.)
            loader_registry: Optional loader registry for file loading. If None, uses default.

        Raises:
            ValueError: If neither storage bundle nor all explicit stores are provided,
                       or if both are provided.
        """
        # Use provided settings or defaults
        self._settings = settings if settings is not None else Settings()

        # Path 1: Storage bundle (convenience)
        if storage is not None:
            if any([vector_store, chunk_store, atom_store, source_registry]):
                raise ValueError("Cannot mix 'storage' bundle with explicit stores")
            (
                self.vector_store,
                self.chunk_store,
                self.atom_store,
                self._source_registry,
            ) = storage.build_stores()

        # Path 2: Explicit stores (enterprise)
        elif all([vector_store, chunk_store, atom_store, source_registry]):
            # Type narrowing - we checked all are not None above
            assert vector_store is not None
            assert chunk_store is not None
            assert atom_store is not None
            assert source_registry is not None
            self.vector_store = vector_store
            self.chunk_store = chunk_store
            self.atom_store = atom_store
            self._source_registry = source_registry

        else:
            raise ValueError(
                "Must provide either 'storage' bundle or all explicit stores "
                "(vector_store, chunk_store, atom_store, source_registry)"
            )

        # Build provider components
        self.embedder = provider.build_embedder()
        self._atomizer = provider.build_atomizer(self._settings)
        self._question_generator = provider.build_question_generator(self._settings)

        # Loader registry (lazily created if not provided)
        self._loader_registry = loader_registry

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

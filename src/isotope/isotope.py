# src/isotope/isotope.py
"""Central configuration class for Isotope."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isotope.configuration import ProviderConfig, StorageConfig
    from isotope.ingestor import Ingestor
    from isotope.loaders import LoaderRegistry
    from isotope.providers import LLMClient
    from isotope.question_generator import DiversityFilter, FilterScope
    from isotope.retriever import Retriever
    from isotope.stores import AtomStore, ChunkStore, EmbeddedQuestionStore, SourceRegistry

from isotope.settings import Settings


class Isotope:
    """Central configuration for Isotope stores and components.

    Isotope bundles all the stores and AI components together so you can
    configure once and create Retrievers/Ingestors from it.

    There are two ways to create an Isotope instance:

    1. With a storage bundle (developer-friendly):

        from isotope import Isotope, LiteLLMProvider, LocalStorage

        iso = Isotope(
            provider=LiteLLMProvider(llm="openai/gpt-4o", embedding="text-embedding-3-small"),
            storage=LocalStorage("./data"),
        )
        ingestor = iso.ingestor()

    2. With explicit stores (enterprise):

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
        ingestor = iso.ingestor()
    """

    def __init__(
        self,
        *,
        provider: ProviderConfig,
        # EITHER storage bundle...
        storage: StorageConfig | None = None,
        # ...OR explicit stores
        embedded_question_store: EmbeddedQuestionStore | None = None,
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
            embedded_question_store: Explicit embedded question store.
                Use with other explicit stores.
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
            if any([embedded_question_store, chunk_store, atom_store, source_registry]):
                raise ValueError("Cannot mix 'storage' bundle with explicit stores")
            (
                self.embedded_question_store,
                self.chunk_store,
                self.atom_store,
                self._source_registry,
            ) = storage.build_stores()

        # Path 2: Explicit stores (enterprise)
        elif all([embedded_question_store, chunk_store, atom_store, source_registry]):
            # Type narrowing - we checked all are not None above
            assert embedded_question_store is not None
            assert chunk_store is not None
            assert atom_store is not None
            assert source_registry is not None
            self.embedded_question_store = embedded_question_store
            self.chunk_store = chunk_store
            self.atom_store = atom_store
            self._source_registry = source_registry

        else:
            raise ValueError(
                "Must provide either 'storage' bundle or all explicit stores "
                "(embedded_question_store, chunk_store, atom_store, source_registry)"
            )

        # Build provider components
        self.embedder = provider.build_embedder()
        self._atomizer = provider.build_atomizer(self._settings)
        self._question_generator = provider.build_question_generator(self._settings)

        # Loader registry (lazily created if not provided)
        self._loader_registry = loader_registry

    @classmethod
    def from_stores(
        cls,
        *,
        provider: ProviderConfig,
        embedded_question_store: EmbeddedQuestionStore,
        chunk_store: ChunkStore,
        atom_store: AtomStore,
        source_registry: SourceRegistry,
        settings: Settings | None = None,
        loader_registry: LoaderRegistry | None = None,
    ) -> Isotope:
        """Create Isotope with explicit stores (enterprise pattern).

        This is the explicit alternative to using a StorageConfig bundle.
        Use this when you need fine-grained control over store configuration.

        Args:
            provider: Provider configuration for AI components.
            embedded_question_store: Store for embedded questions.
            chunk_store: Store for text chunks.
            atom_store: Store for atomic statements.
            source_registry: Registry for tracking source content hashes.
            settings: Optional behavioral settings.
            loader_registry: Optional custom loader registry.

        Returns:
            Configured Isotope instance.

        Example:
            iso = Isotope.from_stores(
                provider=LiteLLMProvider(...),
                embedded_question_store=ChromaEmbeddedQuestionStore("./data/questions"),
                chunk_store=SQLiteChunkStore("./data/chunks.db"),
                atom_store=SQLiteAtomStore("./data/atoms.db"),
                source_registry=SQLiteSourceRegistry("./data/sources.db"),
            )
        """
        return cls(
            provider=provider,
            embedded_question_store=embedded_question_store,
            chunk_store=chunk_store,
            atom_store=atom_store,
            source_registry=source_registry,
            settings=settings,
            loader_registry=loader_registry,
        )

    def _get_loader_registry(self) -> LoaderRegistry:
        """Get or create the loader registry."""
        if self._loader_registry is None:
            from isotope.loaders import LoaderRegistry

            self._loader_registry = LoaderRegistry.default()
        return self._loader_registry

    def _create_diversity_filter(self) -> DiversityFilter | None:
        """Create diversity filter if threshold is set."""
        if self._settings.question_diversity_threshold is not None:
            from isotope.question_generator import DiversityFilter

            return DiversityFilter(threshold=self._settings.question_diversity_threshold)
        return None

    def retriever(
        self,
        *,
        llm_client: LLMClient | None = None,
        llm_model: str | None = None,
        synthesis_prompt: str | None = None,
        synthesis_temperature: float | None = None,
        default_k: int | None = None,
    ) -> Retriever:
        """Create a Retriever using this instance's stores.

        Args:
            llm_client: LLM client for answer synthesis. Pass None to disable synthesis.
            llm_model: Convenience option for answer synthesis using LiteLLM.
                If set and llm_client is None, Isotope will create a LiteLLMClient(model=llm_model).
            synthesis_prompt: Custom synthesis prompt template.
            synthesis_temperature: Temperature for synthesis LLM calls.
            default_k: Number of results to return. If None, uses settings default.

        Returns:
            Configured Retriever instance.
        """
        from isotope.retriever import Retriever

        if llm_client is None and llm_model is not None:
            from isotope.providers import LiteLLMClient

            llm_client = LiteLLMClient(model=llm_model)

        return Retriever(
            embedded_question_store=self.embedded_question_store,
            chunk_store=self.chunk_store,
            atom_store=self.atom_store,
            embedder=self.embedder,
            default_k=default_k if default_k is not None else self._settings.default_k,
            llm_client=llm_client,
            synthesis_prompt=synthesis_prompt or self._settings.synthesis_prompt,
            synthesis_temperature=(
                synthesis_temperature
                if synthesis_temperature is not None
                else self._settings.synthesis_temperature
            ),
        )

    def ingestor(
        self,
        *,
        diversity_filter: DiversityFilter | None = None,
        use_diversity_filter: bool = True,
        diversity_scope: FilterScope | None = None,
        max_concurrent_questions: int | None = None,
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
            max_concurrent_questions: Maximum concurrent async question generation
                requests. Only used by aingest_chunks(). If None, uses settings.

        Returns:
            Configured Ingestor instance using the atomizer and generator
            provided at Isotope initialization.
        """
        from isotope.ingestor import Ingestor

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
            embedded_question_store=self.embedded_question_store,
            chunk_store=self.chunk_store,
            atom_store=self.atom_store,
            atomizer=self._atomizer,
            embedder=self.embedder,
            question_generator=self._question_generator,
            diversity_filter=effective_diversity_filter,
            diversity_scope=effective_diversity_scope,
            max_concurrent_questions=(
                max_concurrent_questions
                if max_concurrent_questions is not None
                else self._settings.max_concurrent_questions
            ),
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
            self.embedded_question_store.delete_by_chunk_ids(chunk_ids)
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

    async def aingest_file(
        self,
        filepath: str,
        source_id: str | None = None,
        *,
        diversity_filter: DiversityFilter | None = None,
        use_diversity_filter: bool = True,
        diversity_scope: FilterScope | None = None,
        max_concurrent_questions: int | None = None,
    ) -> dict:
        """Ingest a file with async question generation, skipping if content unchanged.

        Same as ingest_file() but uses concurrent async requests for question
        generation to reduce total wall-clock time.

        Args:
            filepath: Path to the file to ingest
            source_id: Optional custom source identifier. If not provided,
                      the absolute path will be used as the source.
            diversity_filter: Custom diversity filter for question deduplication
            use_diversity_filter: Whether to use diversity filter
            diversity_scope: Scope for diversity filtering
            max_concurrent_questions: Maximum concurrent async requests.
                                     If None, uses settings default.

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
            self.embedded_question_store.delete_by_chunk_ids(chunk_ids)
            self.atom_store.delete_by_chunk_ids(chunk_ids)
            self.chunk_store.delete_by_source(source)
            self._source_registry.delete(source)

        # Load file into chunks
        loader_registry = self._get_loader_registry()
        chunks = loader_registry.load(filepath, source_id)
        if not chunks:
            return {"skipped": True, "reason": "no content"}

        # Ingest chunks using async
        ingestor = self.ingestor(
            diversity_filter=diversity_filter,
            use_diversity_filter=use_diversity_filter,
            diversity_scope=diversity_scope,
            max_concurrent_questions=max_concurrent_questions,
        )
        result = await ingestor.aingest_chunks(chunks)

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

        self.embedded_question_store.delete_by_chunk_ids(chunk_ids)
        self.atom_store.delete_by_chunk_ids(chunk_ids)
        self.chunk_store.delete_by_source(source)
        self._source_registry.delete(source)

        return {"deleted": True, "chunks_removed": len(chunk_ids)}

    def close(self) -> None:
        """Close the embedded question store and release resources.

        Call this when you're done with the Isotope instance to release
        ChromaDB file handles. This is especially important in test suites
        to avoid 'too many open files' errors.

        Note: SQLite stores (chunk_store, atom_store) use per-operation
        connections and don't require explicit closing.

        After calling close(), the Isotope instance should not be used.
        """
        if hasattr(self.embedded_question_store, "close"):
            self.embedded_question_store.close()

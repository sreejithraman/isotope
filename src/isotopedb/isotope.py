# src/isotopedb/isotope.py
"""Central configuration class for IsotopeDB."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isotopedb.atomizer import Atomizer
    from isotopedb.dedup import Deduplicator
    from isotopedb.embedder import Embedder
    from isotopedb.generator import DiversityFilter, QuestionGenerator
    from isotopedb.ingestor import Ingestor
    from isotopedb.retriever import Retriever
    from isotopedb.stores import AtomStore, DocStore, VectorStore

from isotopedb.config import Settings


class Isotope:
    """Central configuration for IsotopeDB stores and embedder.

    Isotope bundles all the stores and embedder together so you can
    configure once and create Retrievers/Ingestors from it.

    Configuration is read from ISOTOPE_* environment variables by default,
    with explicit parameters taking precedence.

    Example:
        # Simple - uses defaults from ISOTOPE_* env vars
        iso = Isotope()

        # With custom data directory
        iso = Isotope(data_dir="./my_data")

        # With custom models
        iso = Isotope(
            data_dir="./my_data",
            embedding_model="openai/text-embedding-3-small",
            llm_model="openai/gpt-4",
        )

        # Override specific stores
        iso = Isotope(vector_store=MyPineconeStore())

        # Create retriever/ingestor
        retriever = iso.retriever()
        ingestor = iso.ingestor()
    """

    def __init__(
        self,
        *,
        data_dir: str | None = None,
        embedding_model: str | None = None,
        llm_model: str | None = None,
        vector_store: VectorStore | None = None,
        doc_store: DocStore | None = None,
        atom_store: AtomStore | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        """Create an Isotope instance.

        Args:
            data_dir: Data directory for stores. Defaults to ISOTOPE_DATA_DIR
                      env var or "./isotope_data".
            embedding_model: Embedding model name. Defaults to ISOTOPE_EMBEDDING_MODEL
                             env var or "gemini/text-embedding-004".
            llm_model: LLM model name for synthesis/generation. Defaults to
                       ISOTOPE_LLM_MODEL env var.
            vector_store: Custom vector store (default: ChromaVectorStore)
            doc_store: Custom doc store (default: SQLiteDocStore)
            atom_store: Custom atom store (default: SQLiteAtomStore)
            embedder: Custom embedder (default: LiteLLMEmbedder)
        """
        # Load defaults from env vars via Settings (internal)
        self._settings = Settings()

        # Override with explicit params
        self._data_dir = data_dir or self._settings.data_dir
        self._embedding_model = embedding_model or self._settings.embedding_model
        self._llm_model = llm_model or self._settings.llm_model

        # Create stores (use provided or create defaults)
        self.vector_store = vector_store or self._create_vector_store()
        self.doc_store = doc_store or self._create_doc_store()
        self.atom_store = atom_store or self._create_atom_store()
        self.embedder = embedder or self._create_embedder()

    def _ensure_data_dir(self) -> str:
        """Ensure data directory exists and return its path."""
        Path(self._data_dir).mkdir(parents=True, exist_ok=True)
        return self._data_dir

    def _create_vector_store(self) -> VectorStore:
        """Create default vector store from settings."""
        from isotopedb.stores import ChromaVectorStore

        data_dir = self._ensure_data_dir()
        return ChromaVectorStore(os.path.join(data_dir, "chroma"))

    def _create_doc_store(self) -> DocStore:
        """Create default doc store from settings."""
        from isotopedb.stores import SQLiteDocStore

        data_dir = self._ensure_data_dir()
        return SQLiteDocStore(os.path.join(data_dir, "docs.db"))

    def _create_atom_store(self) -> AtomStore:
        """Create default atom store from settings."""
        from isotopedb.stores import SQLiteAtomStore

        data_dir = self._ensure_data_dir()
        return SQLiteAtomStore(os.path.join(data_dir, "atoms.db"))

    def _create_embedder(self) -> Embedder:
        """Create default embedder."""
        from isotopedb.embedder import LiteLLMEmbedder

        return LiteLLMEmbedder(model=self._embedding_model)

    def _create_atomizer(self) -> Atomizer:
        """Create atomizer based on settings."""
        if self._settings.atomizer == "llm":
            from isotopedb.atomizer import LLMAtomizer

            return LLMAtomizer(model=self._llm_model)
        else:
            from isotopedb.atomizer import SentenceAtomizer

            return SentenceAtomizer()

    def _create_generator(self) -> QuestionGenerator:
        """Create question generator."""
        from isotopedb.generator import LiteLLMQuestionGenerator

        return LiteLLMQuestionGenerator(
            model=self._llm_model,
            num_questions=self._settings.questions_per_atom,
        )

    def _create_deduplicator(self) -> Deduplicator:
        """Create deduplicator based on settings."""
        if self._settings.dedup_strategy == "source_aware":
            from isotopedb.dedup import SourceAwareDedup

            return SourceAwareDedup()
        else:
            from isotopedb.dedup import NoDedup

            return NoDedup()

    def _create_diversity_filter(self) -> DiversityFilter | None:
        """Create diversity filter if threshold is set."""
        if self._settings.question_diversity_threshold is not None:
            from isotopedb.generator import DiversityFilter

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
            llm_model: LLM model for answer synthesis. If None, uses the
                       instance's llm_model. Pass empty string "" to disable.
            synthesis_prompt: Custom synthesis prompt template.
            default_k: Number of results to return. If None, uses default.

        Returns:
            Configured Retriever instance.
        """
        from isotopedb.retriever import Retriever

        # Handle llm_model: None = use instance default, "" = disable
        effective_llm_model: str | None
        if llm_model is None:
            effective_llm_model = self._llm_model
        elif llm_model == "":
            effective_llm_model = None
        else:
            effective_llm_model = llm_model

        return Retriever(
            vector_store=self.vector_store,
            doc_store=self.doc_store,
            atom_store=self.atom_store,
            embedder=self.embedder,
            default_k=default_k if default_k is not None else self._settings.default_k,
            llm_model=effective_llm_model,
            synthesis_prompt=synthesis_prompt,
        )

    def ingestor(
        self,
        *,
        atomizer: Atomizer | None = None,
        generator: QuestionGenerator | None = None,
        deduplicator: Deduplicator | None = None,
        diversity_filter: DiversityFilter | None = None,
        use_diversity_filter: bool = True,
    ) -> Ingestor:
        """Create an Ingestor using this instance's stores.

        Args:
            atomizer: Custom atomizer. If None, created from settings.
            generator: Custom question generator. If None, created from settings.
            deduplicator: Custom deduplicator. If None, created from settings.
            diversity_filter: Custom diversity filter. If None and
                              use_diversity_filter is True, created from settings.
            use_diversity_filter: Whether to use diversity filter. Set False to
                                  disable even if settings has a threshold.

        Returns:
            Configured Ingestor instance.
        """
        from isotopedb.ingestor import Ingestor

        # Create components from settings if not provided
        effective_atomizer = atomizer or self._create_atomizer()
        effective_generator = generator or self._create_generator()
        effective_deduplicator = deduplicator or self._create_deduplicator()

        # Handle diversity filter
        effective_diversity_filter: DiversityFilter | None
        if diversity_filter is not None:
            effective_diversity_filter = diversity_filter
        elif use_diversity_filter:
            effective_diversity_filter = self._create_diversity_filter()
        else:
            effective_diversity_filter = None

        return Ingestor(
            vector_store=self.vector_store,
            doc_store=self.doc_store,
            atom_store=self.atom_store,
            atomizer=effective_atomizer,
            embedder=self.embedder,
            generator=effective_generator,
            deduplicator=effective_deduplicator,
            diversity_filter=effective_diversity_filter,
        )

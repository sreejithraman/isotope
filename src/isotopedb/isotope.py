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
    from isotopedb.generator import DiversityFilter, FilterScope, QuestionGenerator
    from isotopedb.ingestor import Ingestor
    from isotopedb.retriever import Retriever
    from isotopedb.stores import AtomStore, DocStore, VectorStore

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

        from isotopedb.providers import LiteClientEmbedder, LiteLLMGenerator, LiteLLMAtomizer

        iso = Isotope(
            vector_store=my_vector_store,
            doc_store=my_doc_store,
            atom_store=my_atom_store,
            embedder=LiteClientEmbedder(model="openai/text-embedding-3-small"),
            atomizer=LiteLLMAtomizer(model="openai/gpt-4o"),
            generator=LiteLLMGenerator(model="openai/gpt-4o"),
        )
        ingestor = iso.ingestor()  # All components configured at init
    """

    def __init__(
        self,
        *,
        vector_store: VectorStore,
        doc_store: DocStore,
        atom_store: AtomStore,
        embedder: Embedder,
        atomizer: Atomizer,
        generator: QuestionGenerator,
    ) -> None:
        """Create an Isotope instance.

        Args:
            vector_store: Vector store for question embeddings (required)
            doc_store: Document store for chunks (required)
            atom_store: Atom store for atomic statements (required)
            embedder: Embedder for creating embeddings (required)
            atomizer: Atomizer for breaking chunks into atoms (required)
            generator: Question generator for creating synthetic questions (required)
        """
        # Load behavioral settings from env vars
        self._settings = Settings()

        # Store references
        self.vector_store = vector_store
        self.doc_store = doc_store
        self.atom_store = atom_store
        self.embedder = embedder

        # Required components for ingestor
        self._atomizer = atomizer
        self._generator = generator

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
        from isotopedb.generator import ClientQuestionGenerator
        from isotopedb.providers.litellm import LiteLLMClient, LiteLLMEmbeddingClient
        from isotopedb.stores import ChromaVectorStore, SQLiteAtomStore, SQLiteDocStore

        # Create local stores
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        vector_store = ChromaVectorStore(os.path.join(data_dir, "chroma"))
        doc_store = SQLiteDocStore(os.path.join(data_dir, "docs.db"))
        atom_store = SQLiteAtomStore(os.path.join(data_dir, "atoms.db"))

        # Load behavioral settings for generator
        settings = Settings()

        # Create LiteLLM clients
        llm_client = LiteLLMClient(model=llm_model)
        embedding_client = LiteLLMEmbeddingClient(model=embedding_model)

        # Create components with injected clients
        embedder = ClientEmbedder(embedding_client=embedding_client)
        generator = ClientQuestionGenerator(
            llm_client=llm_client,
            num_questions=settings.questions_per_atom,
            prompt_template=settings.question_prompt,
        )

        if use_sentence_atomizer:
            atomizer: Atomizer = SentenceAtomizer()
        else:
            atomizer = LLMAtomizer(llm_client=llm_client, prompt_template=settings.atomizer_prompt)

        return cls(
            vector_store=vector_store,
            doc_store=doc_store,
            atom_store=atom_store,
            embedder=embedder,
            atomizer=atomizer,
            generator=generator,
        )

    @classmethod
    def with_local_stores(
        cls,
        *,
        embedder: Embedder,
        atomizer: Atomizer,
        generator: QuestionGenerator,
        data_dir: str = "./isotope_data",
    ) -> Isotope:
        """Create Isotope with local stores (Chroma + SQLite).

        Use this when you want to bring your own embedder/atomizer/generator
        but use local stores for development.

        Args:
            embedder: Embedder implementation (required)
            atomizer: Atomizer implementation (required)
            generator: Question generator implementation (required)
            data_dir: Base directory for all stores.

        Returns:
            Configured Isotope instance with local stores.
        """
        from isotopedb.stores import ChromaVectorStore, SQLiteAtomStore, SQLiteDocStore

        Path(data_dir).mkdir(parents=True, exist_ok=True)

        return cls(
            vector_store=ChromaVectorStore(os.path.join(data_dir, "chroma")),
            doc_store=SQLiteDocStore(os.path.join(data_dir, "docs.db")),
            atom_store=SQLiteAtomStore(os.path.join(data_dir, "atoms.db")),
            embedder=embedder,
            atomizer=atomizer,
            generator=generator,
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
            llm_model: LiteLLM model for answer synthesis. Pass None to disable synthesis.
            synthesis_prompt: Custom synthesis prompt template.
            default_k: Number of results to return. If None, uses settings default.

        Returns:
            Configured Retriever instance.
        """
        from isotopedb.retriever import Retriever

        return Retriever(
            vector_store=self.vector_store,
            doc_store=self.doc_store,
            atom_store=self.atom_store,
            embedder=self.embedder,
            default_k=default_k if default_k is not None else self._settings.default_k,
            llm_model=llm_model,
            synthesis_prompt=synthesis_prompt or self._settings.synthesis_prompt,
        )

    def ingestor(
        self,
        *,
        deduplicator: Deduplicator | None = None,
        diversity_filter: DiversityFilter | None = None,
        use_diversity_filter: bool = True,
        diversity_scope: FilterScope | None = None,
    ) -> Ingestor:
        """Create an Ingestor using this instance's stores.

        Args:
            deduplicator: Custom deduplicator. If None, created from settings.
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

        # Create deduplicator from settings if not provided
        effective_deduplicator = deduplicator or self._create_deduplicator()

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
            doc_store=self.doc_store,
            atom_store=self.atom_store,
            atomizer=self._atomizer,
            embedder=self.embedder,
            generator=self._generator,
            deduplicator=effective_deduplicator,
            diversity_filter=effective_diversity_filter,
            diversity_scope=effective_diversity_scope,
        )

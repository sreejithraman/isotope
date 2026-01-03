# tests/test_isotope.py
"""Tests for the Isotope class."""

import os

import pytest

from isotopedb.isotope import Isotope
from isotopedb.stores import ChromaVectorStore, SQLiteAtomStore, SQLiteDocStore


class TestIsotopeInit:
    def test_init_with_stores(self, temp_dir, mock_embedder):
        """Test initialization with explicit stores."""
        vector_store = ChromaVectorStore(os.path.join(temp_dir, "chroma"))
        doc_store = SQLiteDocStore(os.path.join(temp_dir, "docs.db"))
        atom_store = SQLiteAtomStore(os.path.join(temp_dir, "atoms.db"))

        iso = Isotope(
            vector_store=vector_store,
            doc_store=doc_store,
            atom_store=atom_store,
            embedder=mock_embedder,
        )

        assert iso.vector_store is vector_store
        assert iso.doc_store is doc_store
        assert iso.atom_store is atom_store
        assert iso.embedder is mock_embedder

    def test_init_with_atomizer_and_generator(self, temp_dir, mock_embedder, mock_atomizer, mock_generator):
        """Test initialization with optional atomizer and generator."""
        vector_store = ChromaVectorStore(os.path.join(temp_dir, "chroma"))
        doc_store = SQLiteDocStore(os.path.join(temp_dir, "docs.db"))
        atom_store = SQLiteAtomStore(os.path.join(temp_dir, "atoms.db"))

        iso = Isotope(
            vector_store=vector_store,
            doc_store=doc_store,
            atom_store=atom_store,
            embedder=mock_embedder,
            atomizer=mock_atomizer,
            generator=mock_generator,
        )

        assert iso._atomizer is mock_atomizer
        assert iso._generator is mock_generator

    def test_with_local_stores_creates_all(self, temp_dir, mock_embedder):
        """Test with_local_stores factory creates all stores."""
        iso = Isotope.with_local_stores(data_dir=temp_dir, embedder=mock_embedder)

        assert isinstance(iso.vector_store, ChromaVectorStore)
        assert isinstance(iso.doc_store, SQLiteDocStore)
        assert isinstance(iso.atom_store, SQLiteAtomStore)
        assert iso.embedder is mock_embedder

    def test_with_local_stores_creates_directory(self, temp_dir, mock_embedder):
        """Test that with_local_stores creates the data directory if it doesn't exist."""
        new_dir = os.path.join(temp_dir, "new_data")

        Isotope.with_local_stores(data_dir=new_dir, embedder=mock_embedder)

        assert os.path.isdir(new_dir)


class TestIsotopeWithLiteLLM:
    def test_with_litellm_requires_litellm(self, temp_dir):
        """Test that with_litellm fails gracefully if litellm not installed."""
        pytest.importorskip("litellm", reason="This test requires litellm package")
        # If litellm is installed, this should work
        iso = Isotope.with_litellm(
            data_dir=temp_dir,
            llm_model="openai/gpt-4o",
            embedding_model="openai/text-embedding-3-small",
        )
        assert iso is not None

    def test_with_litellm_creates_local_stores(self, temp_dir):
        """Test that with_litellm creates local stores."""
        pytest.importorskip("litellm", reason="This test requires litellm package")

        iso = Isotope.with_litellm(
            data_dir=temp_dir,
            llm_model="openai/gpt-4o",
            embedding_model="openai/text-embedding-3-small",
        )

        assert isinstance(iso.vector_store, ChromaVectorStore)
        assert isinstance(iso.doc_store, SQLiteDocStore)
        assert isinstance(iso.atom_store, SQLiteAtomStore)

    def test_with_litellm_creates_embedder(self, temp_dir):
        """Test that with_litellm creates LiteLLM embedder."""
        pytest.importorskip("litellm", reason="This test requires litellm package")
        from isotopedb.litellm import LiteLLMEmbedder

        iso = Isotope.with_litellm(
            data_dir=temp_dir,
            llm_model="openai/gpt-4o",
            embedding_model="openai/text-embedding-3-small",
        )

        assert isinstance(iso.embedder, LiteLLMEmbedder)
        assert iso.embedder.model == "openai/text-embedding-3-small"

    def test_with_litellm_creates_llm_atomizer_by_default(self, temp_dir):
        """Test that with_litellm creates LLM atomizer by default."""
        pytest.importorskip("litellm", reason="This test requires litellm package")
        from isotopedb.litellm import LiteLLMAtomizer

        iso = Isotope.with_litellm(
            data_dir=temp_dir,
            llm_model="openai/gpt-4o",
            embedding_model="openai/text-embedding-3-small",
        )

        assert isinstance(iso._atomizer, LiteLLMAtomizer)

    def test_with_litellm_can_use_sentence_atomizer(self, temp_dir):
        """Test that with_litellm can use sentence atomizer."""
        pytest.importorskip("litellm", reason="This test requires litellm package")
        from isotopedb.atomizer import SentenceAtomizer

        iso = Isotope.with_litellm(
            data_dir=temp_dir,
            llm_model="openai/gpt-4o",
            embedding_model="openai/text-embedding-3-small",
            use_sentence_atomizer=True,
        )

        assert isinstance(iso._atomizer, SentenceAtomizer)


class TestIsotopeRetriever:
    def test_retriever_creates_retriever(self, temp_dir, mock_embedder):
        """Test that retriever() returns a Retriever instance."""
        from isotopedb.retriever import Retriever

        iso = Isotope.with_local_stores(data_dir=temp_dir, embedder=mock_embedder)
        retriever = iso.retriever()

        assert isinstance(retriever, Retriever)
        assert retriever.vector_store is iso.vector_store
        assert retriever.doc_store is iso.doc_store
        assert retriever.atom_store is iso.atom_store
        assert retriever.embedder is iso.embedder

    def test_retriever_uses_default_k(self, temp_dir, mock_embedder, monkeypatch):
        """Test that retriever uses default_k from env var."""
        monkeypatch.setenv("ISOTOPE_DEFAULT_K", "15")

        iso = Isotope.with_local_stores(data_dir=temp_dir, embedder=mock_embedder)
        retriever = iso.retriever()

        assert retriever.default_k == 15

    def test_retriever_with_custom_k(self, temp_dir, mock_embedder):
        """Test that retriever respects custom default_k override."""
        iso = Isotope.with_local_stores(data_dir=temp_dir, embedder=mock_embedder)
        retriever = iso.retriever(default_k=20)

        assert retriever.default_k == 20

    def test_retriever_with_custom_llm_model(self, temp_dir, mock_embedder):
        """Test that retriever respects llm_model override."""
        iso = Isotope.with_local_stores(data_dir=temp_dir, embedder=mock_embedder)
        retriever = iso.retriever(llm_model="custom/model")

        assert retriever.llm_model == "custom/model"

    def test_retriever_no_llm_by_default(self, temp_dir, mock_embedder):
        """Test that retriever has no LLM model by default."""
        iso = Isotope.with_local_stores(data_dir=temp_dir, embedder=mock_embedder)
        retriever = iso.retriever()

        assert retriever.llm_model is None

    def test_retriever_with_synthesis_prompt(self, temp_dir, mock_embedder):
        """Test that retriever respects synthesis_prompt override."""
        iso = Isotope.with_local_stores(data_dir=temp_dir, embedder=mock_embedder)
        custom_prompt = "Custom prompt: {context}\n{query}"
        retriever = iso.retriever(synthesis_prompt=custom_prompt)

        assert retriever.synthesis_prompt == custom_prompt


class TestIsotopeIngestor:
    def test_ingestor_creates_ingestor(self, temp_dir, mock_embedder, mock_atomizer, mock_generator):
        """Test that ingestor() returns an Ingestor instance."""
        from isotopedb.ingestor import Ingestor

        iso = Isotope.with_local_stores(
            data_dir=temp_dir,
            embedder=mock_embedder,
            atomizer=mock_atomizer,
            generator=mock_generator,
        )
        ingestor = iso.ingestor()

        assert isinstance(ingestor, Ingestor)
        assert ingestor.vector_store is iso.vector_store
        assert ingestor.doc_store is iso.doc_store
        assert ingestor.atom_store is iso.atom_store
        assert ingestor.embedder is iso.embedder

    def test_ingestor_requires_atomizer(self, temp_dir, mock_embedder, mock_generator):
        """Test that ingestor raises error if no atomizer provided."""
        iso = Isotope.with_local_stores(
            data_dir=temp_dir,
            embedder=mock_embedder,
            generator=mock_generator,
        )

        with pytest.raises(ValueError, match="atomizer is required"):
            iso.ingestor()

    def test_ingestor_requires_generator(self, temp_dir, mock_embedder, mock_atomizer):
        """Test that ingestor raises error if no generator provided."""
        iso = Isotope.with_local_stores(
            data_dir=temp_dir,
            embedder=mock_embedder,
            atomizer=mock_atomizer,
        )

        with pytest.raises(ValueError, match="generator is required"):
            iso.ingestor()

    def test_ingestor_with_override_atomizer(self, temp_dir, mock_embedder, mock_atomizer, mock_generator):
        """Test that ingestor respects atomizer override."""
        from isotopedb.atomizer import SentenceAtomizer

        iso = Isotope.with_local_stores(
            data_dir=temp_dir,
            embedder=mock_embedder,
            atomizer=mock_atomizer,
            generator=mock_generator,
        )

        override_atomizer = SentenceAtomizer()
        ingestor = iso.ingestor(atomizer=override_atomizer)

        assert ingestor.atomizer is override_atomizer

    def test_ingestor_uses_env_dedup(self, temp_dir, mock_embedder, mock_atomizer, mock_generator, monkeypatch):
        """Test that ingestor creates deduplicator from env var."""
        from isotopedb.dedup import SourceAwareDedup

        monkeypatch.setenv("ISOTOPE_DEDUP_STRATEGY", "source_aware")

        iso = Isotope.with_local_stores(
            data_dir=temp_dir,
            embedder=mock_embedder,
            atomizer=mock_atomizer,
            generator=mock_generator,
        )
        ingestor = iso.ingestor()

        assert isinstance(ingestor.deduplicator, SourceAwareDedup)

    def test_ingestor_with_no_dedup(self, temp_dir, mock_embedder, mock_atomizer, mock_generator, monkeypatch):
        """Test that ingestor creates NoDedup when strategy is 'none'."""
        from isotopedb.dedup import NoDedup

        monkeypatch.setenv("ISOTOPE_DEDUP_STRATEGY", "none")

        iso = Isotope.with_local_stores(
            data_dir=temp_dir,
            embedder=mock_embedder,
            atomizer=mock_atomizer,
            generator=mock_generator,
        )
        ingestor = iso.ingestor()

        assert isinstance(ingestor.deduplicator, NoDedup)

    def test_ingestor_creates_diversity_filter(self, temp_dir, mock_embedder, mock_atomizer, mock_generator, monkeypatch):
        """Test that ingestor creates diversity filter from env var."""
        from isotopedb.generator import DiversityFilter

        monkeypatch.setenv("ISOTOPE_QUESTION_DIVERSITY_THRESHOLD", "0.9")

        iso = Isotope.with_local_stores(
            data_dir=temp_dir,
            embedder=mock_embedder,
            atomizer=mock_atomizer,
            generator=mock_generator,
        )
        ingestor = iso.ingestor()

        assert ingestor.diversity_filter is not None
        assert isinstance(ingestor.diversity_filter, DiversityFilter)

    def test_ingestor_has_diversity_filter_by_default(self, temp_dir, mock_embedder, mock_atomizer, mock_generator):
        """Test that ingestor has diversity filter by default (threshold=0.85)."""
        from isotopedb.generator import DiversityFilter

        iso = Isotope.with_local_stores(
            data_dir=temp_dir,
            embedder=mock_embedder,
            atomizer=mock_atomizer,
            generator=mock_generator,
        )
        ingestor = iso.ingestor()

        # Default threshold is 0.85
        assert ingestor.diversity_filter is not None
        assert isinstance(ingestor.diversity_filter, DiversityFilter)

    def test_ingestor_no_diversity_filter_when_empty_threshold(
        self, temp_dir, mock_embedder, mock_atomizer, mock_generator, monkeypatch
    ):
        """Test that ingestor has no diversity filter when threshold is empty string."""
        monkeypatch.setenv("ISOTOPE_QUESTION_DIVERSITY_THRESHOLD", "")

        iso = Isotope.with_local_stores(
            data_dir=temp_dir,
            embedder=mock_embedder,
            atomizer=mock_atomizer,
            generator=mock_generator,
        )
        ingestor = iso.ingestor()

        assert ingestor.diversity_filter is None

    def test_ingestor_disable_diversity_filter(
        self, temp_dir, mock_embedder, mock_atomizer, mock_generator, monkeypatch
    ):
        """Test that use_diversity_filter=False disables filter."""
        monkeypatch.setenv("ISOTOPE_QUESTION_DIVERSITY_THRESHOLD", "0.9")

        iso = Isotope.with_local_stores(
            data_dir=temp_dir,
            embedder=mock_embedder,
            atomizer=mock_atomizer,
            generator=mock_generator,
        )
        ingestor = iso.ingestor(use_diversity_filter=False)

        assert ingestor.diversity_filter is None

    def test_ingestor_uses_diversity_scope_from_settings(
        self, temp_dir, mock_embedder, mock_atomizer, mock_generator, monkeypatch
    ):
        """Test that ingestor uses diversity_scope from env var."""
        monkeypatch.setenv("ISOTOPE_DIVERSITY_SCOPE", "per_chunk")

        iso = Isotope.with_local_stores(
            data_dir=temp_dir,
            embedder=mock_embedder,
            atomizer=mock_atomizer,
            generator=mock_generator,
        )
        ingestor = iso.ingestor()

        assert ingestor.diversity_scope == "per_chunk"

    def test_ingestor_diversity_scope_override(
        self, temp_dir, mock_embedder, mock_atomizer, mock_generator, monkeypatch
    ):
        """Test that diversity_scope parameter overrides settings."""
        monkeypatch.setenv("ISOTOPE_DIVERSITY_SCOPE", "per_chunk")

        iso = Isotope.with_local_stores(
            data_dir=temp_dir,
            embedder=mock_embedder,
            atomizer=mock_atomizer,
            generator=mock_generator,
        )
        ingestor = iso.ingestor(diversity_scope="per_atom")

        assert ingestor.diversity_scope == "per_atom"


class TestIsotopeSharedStores:
    def test_retriever_and_ingestor_share_stores(self, temp_dir, mock_embedder, mock_atomizer, mock_generator):
        """Test that retriever and ingestor share the same store instances."""
        iso = Isotope.with_local_stores(
            data_dir=temp_dir,
            embedder=mock_embedder,
            atomizer=mock_atomizer,
            generator=mock_generator,
        )

        retriever = iso.retriever()
        ingestor = iso.ingestor()

        # Should be the exact same instances
        assert retriever.vector_store is ingestor.vector_store
        assert retriever.doc_store is ingestor.doc_store
        assert retriever.atom_store is ingestor.atom_store
        assert retriever.embedder is ingestor.embedder

    def test_multiple_retrievers_share_stores(self, temp_dir, mock_embedder):
        """Test that multiple retrievers share stores."""
        iso = Isotope.with_local_stores(data_dir=temp_dir, embedder=mock_embedder)

        r1 = iso.retriever()
        r2 = iso.retriever(default_k=10)

        assert r1.vector_store is r2.vector_store
        assert r1.doc_store is r2.doc_store
        assert r1.atom_store is r2.atom_store
        assert r1.embedder is r2.embedder

"""Tests for the Isotope class."""

import os

import pytest

from isotopedb.isotope import Isotope
from isotopedb.stores import ChromaVectorStore, SQLiteAtomStore, SQLiteDocStore


class TestIsotopeInit:
    def test_init_with_defaults(self, temp_dir, monkeypatch):
        """Test initialization with default settings from env vars."""
        monkeypatch.setenv("ISOTOPE_DATA_DIR", temp_dir)

        iso = Isotope()

        assert iso.vector_store is not None
        assert iso.doc_store is not None
        assert iso.atom_store is not None
        assert iso.embedder is not None

    def test_init_with_data_dir(self, temp_dir):
        """Test initialization with custom data directory."""
        iso = Isotope(data_dir=temp_dir)

        assert isinstance(iso.vector_store, ChromaVectorStore)
        assert isinstance(iso.doc_store, SQLiteDocStore)
        assert isinstance(iso.atom_store, SQLiteAtomStore)

    def test_init_creates_data_directory(self, temp_dir):
        """Test that Isotope creates the data directory if it doesn't exist."""
        new_dir = os.path.join(temp_dir, "new_data")

        iso = Isotope(data_dir=new_dir)

        assert os.path.isdir(new_dir)

    def test_init_with_embedding_model(self, temp_dir):
        """Test initialization with custom embedding model."""
        iso = Isotope(data_dir=temp_dir, embedding_model="openai/text-embedding-3-small")

        # Embedder should be created with the custom model
        assert iso.embedder is not None

    def test_init_with_llm_model(self, temp_dir):
        """Test initialization with custom LLM model."""
        iso = Isotope(data_dir=temp_dir, llm_model="openai/gpt-4")

        # LLM model is used when creating retriever/ingestor
        retriever = iso.retriever()
        assert retriever.llm_model == "openai/gpt-4"

    def test_init_with_custom_vector_store(self, temp_dir):
        """Test initialization with custom vector store."""
        custom_store = ChromaVectorStore(os.path.join(temp_dir, "custom_chroma"))

        iso = Isotope(data_dir=temp_dir, vector_store=custom_store)

        assert iso.vector_store is custom_store

    def test_init_with_custom_doc_store(self, temp_dir):
        """Test initialization with custom doc store."""
        custom_store = SQLiteDocStore(os.path.join(temp_dir, "custom_docs.db"))

        iso = Isotope(data_dir=temp_dir, doc_store=custom_store)

        assert iso.doc_store is custom_store

    def test_init_with_custom_atom_store(self, temp_dir):
        """Test initialization with custom atom store."""
        custom_store = SQLiteAtomStore(os.path.join(temp_dir, "custom_atoms.db"))

        iso = Isotope(data_dir=temp_dir, atom_store=custom_store)

        assert iso.atom_store is custom_store


class TestIsotopeRetriever:
    def test_retriever_creates_retriever(self, temp_dir):
        """Test that retriever() returns a Retriever instance."""
        from isotopedb.retriever import Retriever

        iso = Isotope(data_dir=temp_dir)
        retriever = iso.retriever()

        assert isinstance(retriever, Retriever)
        assert retriever.vector_store is iso.vector_store
        assert retriever.doc_store is iso.doc_store
        assert retriever.atom_store is iso.atom_store
        assert retriever.embedder is iso.embedder

    def test_retriever_uses_default_k(self, temp_dir, monkeypatch):
        """Test that retriever uses default_k from env var."""
        monkeypatch.setenv("ISOTOPE_DEFAULT_K", "15")

        iso = Isotope(data_dir=temp_dir)
        retriever = iso.retriever()

        assert retriever.default_k == 15

    def test_retriever_with_custom_k(self, temp_dir):
        """Test that retriever respects custom default_k override."""
        iso = Isotope(data_dir=temp_dir)
        retriever = iso.retriever(default_k=20)

        assert retriever.default_k == 20

    def test_retriever_uses_instance_llm_model(self, temp_dir):
        """Test that retriever uses Isotope's llm_model by default."""
        iso = Isotope(data_dir=temp_dir, llm_model="gemini/gemini-2.0-flash")
        retriever = iso.retriever()

        assert retriever.llm_model == "gemini/gemini-2.0-flash"

    def test_retriever_with_custom_llm_model(self, temp_dir):
        """Test that retriever respects llm_model override."""
        iso = Isotope(data_dir=temp_dir)
        retriever = iso.retriever(llm_model="custom/model")

        assert retriever.llm_model == "custom/model"

    def test_retriever_disable_llm(self, temp_dir):
        """Test that passing empty string disables LLM synthesis."""
        iso = Isotope(data_dir=temp_dir, llm_model="gemini/gemini-2.0-flash")
        retriever = iso.retriever(llm_model="")

        assert retriever.llm_model is None

    def test_retriever_with_synthesis_prompt(self, temp_dir):
        """Test that retriever respects synthesis_prompt override."""
        iso = Isotope(data_dir=temp_dir)
        custom_prompt = "Custom prompt: {context}\n{query}"
        retriever = iso.retriever(synthesis_prompt=custom_prompt)

        assert retriever.synthesis_prompt == custom_prompt


class TestIsotopeIngestor:
    def test_ingestor_creates_ingestor(self, temp_dir):
        """Test that ingestor() returns an Ingestor instance."""
        from isotopedb.ingestor import Ingestor

        iso = Isotope(data_dir=temp_dir)
        ingestor = iso.ingestor()

        assert isinstance(ingestor, Ingestor)
        assert ingestor.vector_store is iso.vector_store
        assert ingestor.doc_store is iso.doc_store
        assert ingestor.atom_store is iso.atom_store
        assert ingestor.embedder is iso.embedder

    def test_ingestor_uses_env_atomizer(self, temp_dir, monkeypatch):
        """Test that ingestor creates atomizer from env var."""
        from isotopedb.atomizer import SentenceAtomizer

        monkeypatch.setenv("ISOTOPE_ATOMIZER", "sentence")

        iso = Isotope(data_dir=temp_dir)
        ingestor = iso.ingestor()

        assert isinstance(ingestor.atomizer, SentenceAtomizer)

    def test_ingestor_with_custom_atomizer(self, temp_dir):
        """Test that ingestor respects custom atomizer."""
        from isotopedb.atomizer import SentenceAtomizer

        custom_atomizer = SentenceAtomizer()

        iso = Isotope(data_dir=temp_dir)
        ingestor = iso.ingestor(atomizer=custom_atomizer)

        assert ingestor.atomizer is custom_atomizer

    def test_ingestor_uses_env_dedup(self, temp_dir, monkeypatch):
        """Test that ingestor creates deduplicator from env var."""
        from isotopedb.dedup import SourceAwareDedup

        monkeypatch.setenv("ISOTOPE_DEDUP_STRATEGY", "source_aware")

        iso = Isotope(data_dir=temp_dir)
        ingestor = iso.ingestor()

        assert isinstance(ingestor.deduplicator, SourceAwareDedup)

    def test_ingestor_with_no_dedup(self, temp_dir, monkeypatch):
        """Test that ingestor creates NoDedup when strategy is 'none'."""
        from isotopedb.dedup import NoDedup

        monkeypatch.setenv("ISOTOPE_DEDUP_STRATEGY", "none")

        iso = Isotope(data_dir=temp_dir)
        ingestor = iso.ingestor()

        assert isinstance(ingestor.deduplicator, NoDedup)

    def test_ingestor_creates_diversity_filter(self, temp_dir, monkeypatch):
        """Test that ingestor creates diversity filter from env var."""
        from isotopedb.generator import DiversityFilter

        monkeypatch.setenv("ISOTOPE_QUESTION_DIVERSITY_THRESHOLD", "0.9")

        iso = Isotope(data_dir=temp_dir)
        ingestor = iso.ingestor()

        assert ingestor.diversity_filter is not None
        assert isinstance(ingestor.diversity_filter, DiversityFilter)

    def test_ingestor_has_diversity_filter_by_default(self, temp_dir):
        """Test that ingestor has diversity filter by default (threshold=0.85)."""
        from isotopedb.generator import DiversityFilter

        iso = Isotope(data_dir=temp_dir)
        ingestor = iso.ingestor()

        # Default threshold is 0.85
        assert ingestor.diversity_filter is not None
        assert isinstance(ingestor.diversity_filter, DiversityFilter)

    def test_ingestor_no_diversity_filter_when_empty_threshold(self, temp_dir, monkeypatch):
        """Test that ingestor has no diversity filter when threshold is empty string."""
        monkeypatch.setenv("ISOTOPE_QUESTION_DIVERSITY_THRESHOLD", "")

        iso = Isotope(data_dir=temp_dir)
        ingestor = iso.ingestor()

        assert ingestor.diversity_filter is None

    def test_ingestor_disable_diversity_filter(self, temp_dir, monkeypatch):
        """Test that use_diversity_filter=False disables filter."""
        monkeypatch.setenv("ISOTOPE_QUESTION_DIVERSITY_THRESHOLD", "0.9")

        iso = Isotope(data_dir=temp_dir)
        ingestor = iso.ingestor(use_diversity_filter=False)

        assert ingestor.diversity_filter is None


class TestIsotopeSharedStores:
    def test_retriever_and_ingestor_share_stores(self, temp_dir):
        """Test that retriever and ingestor share the same store instances."""
        iso = Isotope(data_dir=temp_dir)

        retriever = iso.retriever()
        ingestor = iso.ingestor()

        # Should be the exact same instances
        assert retriever.vector_store is ingestor.vector_store
        assert retriever.doc_store is ingestor.doc_store
        assert retriever.atom_store is ingestor.atom_store
        assert retriever.embedder is ingestor.embedder

    def test_multiple_retrievers_share_stores(self, temp_dir):
        """Test that multiple retrievers share stores."""
        iso = Isotope(data_dir=temp_dir)

        r1 = iso.retriever()
        r2 = iso.retriever(default_k=10)

        assert r1.vector_store is r2.vector_store
        assert r1.doc_store is r2.doc_store
        assert r1.atom_store is r2.atom_store
        assert r1.embedder is r2.embedder

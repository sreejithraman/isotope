# tests/test_isotope.py
"""Tests for the Isotope class."""

import os

import pytest

from isotope.configuration import LocalStorage
from isotope.isotope import Isotope
from isotope.stores import (
    ChromaEmbeddedQuestionStore,
    SQLiteAtomStore,
    SQLiteChunkStore,
    SQLiteSourceRegistry,
)


class TestIsotopeInit:
    def test_init_with_storage_bundle(self, temp_dir, mock_provider):
        """Test initialization with storage bundle."""
        iso = Isotope(
            provider=mock_provider,
            storage=LocalStorage(temp_dir),
        )

        assert isinstance(iso.embedded_question_store, ChromaEmbeddedQuestionStore)
        assert isinstance(iso.chunk_store, SQLiteChunkStore)
        assert isinstance(iso.atom_store, SQLiteAtomStore)

    def test_init_with_explicit_stores(self, temp_dir, mock_provider):
        """Test initialization with explicit stores."""
        embedded_question_store = ChromaEmbeddedQuestionStore(os.path.join(temp_dir, "chroma"))
        chunk_store = SQLiteChunkStore(os.path.join(temp_dir, "chunks.db"))
        atom_store = SQLiteAtomStore(os.path.join(temp_dir, "atoms.db"))
        source_registry = SQLiteSourceRegistry(os.path.join(temp_dir, "sources.db"))

        iso = Isotope(
            provider=mock_provider,
            embedded_question_store=embedded_question_store,
            chunk_store=chunk_store,
            atom_store=atom_store,
            source_registry=source_registry,
        )

        assert iso.embedded_question_store is embedded_question_store
        assert iso.chunk_store is chunk_store
        assert iso.atom_store is atom_store

    def test_init_rejects_mixed_storage(self, temp_dir, mock_provider):
        """Test that mixing storage bundle with explicit stores raises error."""
        embedded_question_store = ChromaEmbeddedQuestionStore(os.path.join(temp_dir, "chroma"))

        with pytest.raises(ValueError, match="Cannot mix"):
            Isotope(
                provider=mock_provider,
                storage=LocalStorage(temp_dir),
                embedded_question_store=embedded_question_store,
            )

    def test_init_requires_storage_or_explicit_stores(self, mock_provider):
        """Test that at least storage or all explicit stores are required."""
        with pytest.raises(ValueError, match="Must provide either"):
            Isotope(provider=mock_provider)

    def test_init_requires_all_explicit_stores(self, temp_dir, mock_provider):
        """Test that partial explicit stores raise error."""
        embedded_question_store = ChromaEmbeddedQuestionStore(os.path.join(temp_dir, "chroma"))

        with pytest.raises(ValueError, match="Must provide either"):
            Isotope(
                provider=mock_provider,
                embedded_question_store=embedded_question_store,
                # Missing chunk_store, atom_store, source_registry
            )

    def test_storage_bundle_creates_directory(self, temp_dir, mock_provider):
        """Test that storage bundle creates the data directory if it doesn't exist."""
        new_dir = os.path.join(temp_dir, "new_data")

        Isotope(
            provider=mock_provider,
            storage=LocalStorage(new_dir),
        )

        assert os.path.isdir(new_dir)


class TestIsotopeWithLiteLLM:
    def test_litellm_provider_creates_stores(self, temp_dir):
        """Test that LiteLLMProvider with LocalStorage creates stores."""
        pytest.importorskip("litellm", reason="This test requires litellm package")
        from isotope.configuration import LiteLLMProvider

        iso = Isotope(
            provider=LiteLLMProvider(
                llm="openai/gpt-4o",
                embedding="openai/text-embedding-3-small",
            ),
            storage=LocalStorage(temp_dir),
        )

        assert isinstance(iso.embedded_question_store, ChromaEmbeddedQuestionStore)
        assert isinstance(iso.chunk_store, SQLiteChunkStore)
        assert isinstance(iso.atom_store, SQLiteAtomStore)

    def test_litellm_provider_creates_embedder(self, temp_dir):
        """Test that LiteLLMProvider creates ClientEmbedder."""
        pytest.importorskip("litellm", reason="This test requires litellm package")
        from isotope.configuration import LiteLLMProvider
        from isotope.embedder import ClientEmbedder

        iso = Isotope(
            provider=LiteLLMProvider(
                llm="openai/gpt-4o",
                embedding="openai/text-embedding-3-small",
            ),
            storage=LocalStorage(temp_dir),
        )

        assert isinstance(iso.embedder, ClientEmbedder)

    def test_litellm_provider_creates_llm_atomizer_by_default(self, temp_dir):
        """Test that LiteLLMProvider creates LLM atomizer by default."""
        pytest.importorskip("litellm", reason="This test requires litellm package")
        from isotope.atomizer import LLMAtomizer
        from isotope.configuration import LiteLLMProvider

        iso = Isotope(
            provider=LiteLLMProvider(
                llm="openai/gpt-4o",
                embedding="openai/text-embedding-3-small",
            ),
            storage=LocalStorage(temp_dir),
        )

        assert isinstance(iso._atomizer, LLMAtomizer)

    def test_litellm_provider_can_use_sentence_atomizer(self, temp_dir):
        """Test that LiteLLMProvider can use sentence atomizer."""
        pytest.importorskip("litellm", reason="This test requires litellm package")
        from isotope.atomizer import SentenceAtomizer
        from isotope.configuration import LiteLLMProvider

        iso = Isotope(
            provider=LiteLLMProvider(
                llm="openai/gpt-4o",
                embedding="openai/text-embedding-3-small",
                atomizer_type="sentence",
            ),
            storage=LocalStorage(temp_dir),
        )

        assert isinstance(iso._atomizer, SentenceAtomizer)


class TestIsotopeRetriever:
    def test_retriever_creates_retriever(self, temp_dir, mock_provider):
        """Test that retriever() returns a Retriever instance."""
        from isotope.retriever import Retriever

        iso = Isotope(
            provider=mock_provider,
            storage=LocalStorage(temp_dir),
        )
        retriever = iso.retriever()

        assert isinstance(retriever, Retriever)
        assert retriever.embedded_question_store is iso.embedded_question_store
        assert retriever.chunk_store is iso.chunk_store
        assert retriever.atom_store is iso.atom_store
        assert retriever.embedder is iso.embedder

    def test_retriever_uses_default_k(self, temp_dir, mock_provider):
        """Test that retriever uses default_k from settings."""
        from isotope.settings import Settings

        iso = Isotope(
            provider=mock_provider,
            storage=LocalStorage(temp_dir),
            settings=Settings(default_k=15),
        )
        retriever = iso.retriever()

        assert retriever.default_k == 15

    def test_retriever_with_custom_k(self, temp_dir, mock_provider):
        """Test that retriever respects custom default_k override."""
        iso = Isotope(
            provider=mock_provider,
            storage=LocalStorage(temp_dir),
        )
        retriever = iso.retriever(default_k=20)

        assert retriever.default_k == 20

    def test_retriever_with_custom_llm_client(self, temp_dir, mock_provider):
        """Test that retriever respects llm_client override."""
        from unittest.mock import MagicMock

        from isotope.providers import LLMClient

        mock_llm_client = MagicMock(spec=LLMClient)

        iso = Isotope(
            provider=mock_provider,
            storage=LocalStorage(temp_dir),
        )
        retriever = iso.retriever(llm_client=mock_llm_client)

        assert retriever._llm_client is mock_llm_client

    def test_retriever_no_llm_client_by_default(self, temp_dir, mock_provider):
        """Test that retriever has no LLM client by default."""
        iso = Isotope(
            provider=mock_provider,
            storage=LocalStorage(temp_dir),
        )
        retriever = iso.retriever()

        assert retriever._llm_client is None

    def test_retriever_with_synthesis_prompt(self, temp_dir, mock_provider):
        """Test that retriever respects synthesis_prompt override."""
        iso = Isotope(
            provider=mock_provider,
            storage=LocalStorage(temp_dir),
        )
        custom_prompt = "Custom prompt: {context}\n{query}"
        retriever = iso.retriever(synthesis_prompt=custom_prompt)

        assert retriever.synthesis_prompt == custom_prompt


class TestIsotopeIngestor:
    def test_ingestor_creates_ingestor(self, temp_dir, mock_provider):
        """Test that ingestor() returns an Ingestor instance."""
        from isotope.ingestor import Ingestor

        iso = Isotope(
            provider=mock_provider,
            storage=LocalStorage(temp_dir),
        )
        ingestor = iso.ingestor()

        assert isinstance(ingestor, Ingestor)
        assert ingestor.embedded_question_store is iso.embedded_question_store
        assert ingestor.chunk_store is iso.chunk_store
        assert ingestor.atom_store is iso.atom_store
        assert ingestor.embedder is iso.embedder

    def test_ingestor_uses_provider_atomizer(self, temp_dir, mock_provider, mock_atomizer):
        """Test that ingestor uses atomizer from provider."""
        iso = Isotope(
            provider=mock_provider,
            storage=LocalStorage(temp_dir),
        )
        ingestor = iso.ingestor()

        assert ingestor.atomizer is mock_atomizer

    def test_ingestor_uses_provider_generator(self, temp_dir, mock_provider, mock_generator):
        """Test that ingestor uses question_generator from provider."""
        iso = Isotope(
            provider=mock_provider,
            storage=LocalStorage(temp_dir),
        )
        ingestor = iso.ingestor()

        assert ingestor.question_generator is mock_generator

    def test_ingestor_creates_diversity_filter(self, temp_dir, mock_provider):
        """Test that ingestor creates diversity filter from settings."""
        from isotope.question_generator import DiversityFilter
        from isotope.settings import Settings

        iso = Isotope(
            provider=mock_provider,
            storage=LocalStorage(temp_dir),
            settings=Settings(question_diversity_threshold=0.9),
        )
        ingestor = iso.ingestor()

        assert ingestor.diversity_filter is not None
        assert isinstance(ingestor.diversity_filter, DiversityFilter)

    def test_ingestor_has_diversity_filter_by_default(self, temp_dir, mock_provider):
        """Test that ingestor has diversity filter by default (threshold=0.85)."""
        from isotope.question_generator import DiversityFilter

        iso = Isotope(
            provider=mock_provider,
            storage=LocalStorage(temp_dir),
        )
        ingestor = iso.ingestor()

        # Default threshold is 0.85
        assert ingestor.diversity_filter is not None
        assert isinstance(ingestor.diversity_filter, DiversityFilter)

    def test_ingestor_no_diversity_filter_when_none_threshold(self, temp_dir, mock_provider):
        """Test that ingestor has no diversity filter when threshold is None."""
        from isotope.settings import Settings

        iso = Isotope(
            provider=mock_provider,
            storage=LocalStorage(temp_dir),
            settings=Settings(question_diversity_threshold=None),
        )
        ingestor = iso.ingestor()

        assert ingestor.diversity_filter is None

    def test_ingestor_disable_diversity_filter(self, temp_dir, mock_provider):
        """Test that use_diversity_filter=False disables filter."""
        from isotope.settings import Settings

        iso = Isotope(
            provider=mock_provider,
            storage=LocalStorage(temp_dir),
            settings=Settings(question_diversity_threshold=0.9),
        )
        ingestor = iso.ingestor(use_diversity_filter=False)

        assert ingestor.diversity_filter is None

    def test_ingestor_uses_diversity_scope_from_settings(self, temp_dir, mock_provider):
        """Test that ingestor uses diversity_scope from settings."""
        from isotope.settings import Settings

        iso = Isotope(
            provider=mock_provider,
            storage=LocalStorage(temp_dir),
            settings=Settings(diversity_scope="per_chunk"),
        )
        ingestor = iso.ingestor()

        assert ingestor.diversity_scope == "per_chunk"

    def test_ingestor_diversity_scope_override(self, temp_dir, mock_provider):
        """Test that diversity_scope parameter overrides settings."""
        from isotope.settings import Settings

        iso = Isotope(
            provider=mock_provider,
            storage=LocalStorage(temp_dir),
            settings=Settings(diversity_scope="per_chunk"),
        )
        ingestor = iso.ingestor(diversity_scope="per_atom")

        assert ingestor.diversity_scope == "per_atom"


class TestIsotopeSharedStores:
    def test_retriever_and_ingestor_share_stores(self, temp_dir, mock_provider):
        """Test that retriever and ingestor share the same store instances."""
        iso = Isotope(
            provider=mock_provider,
            storage=LocalStorage(temp_dir),
        )

        retriever = iso.retriever()
        ingestor = iso.ingestor()

        # Should be the exact same instances
        assert retriever.embedded_question_store is ingestor.embedded_question_store
        assert retriever.chunk_store is ingestor.chunk_store
        assert retriever.atom_store is ingestor.atom_store
        assert retriever.embedder is ingestor.embedder

    def test_multiple_retrievers_share_stores(self, temp_dir, mock_provider):
        """Test that multiple retrievers share stores."""
        iso = Isotope(
            provider=mock_provider,
            storage=LocalStorage(temp_dir),
        )

        r1 = iso.retriever()
        r2 = iso.retriever(default_k=10)

        assert r1.embedded_question_store is r2.embedded_question_store
        assert r1.chunk_store is r2.chunk_store
        assert r1.atom_store is r2.atom_store
        assert r1.embedder is r2.embedder

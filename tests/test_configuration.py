# tests/test_configuration.py
"""Tests for the configuration module."""

import os

import pytest


class TestLocalStorage:
    def test_build_stores_creates_all_four(self, temp_dir):
        """Test that LocalStorage.build_stores() creates all four stores."""
        pytest.importorskip("chromadb", reason="This test requires chromadb")
        from isotope.configuration import LocalStorage
        from isotope.stores import (
            ChromaEmbeddedQuestionStore,
            SQLiteAtomStore,
            SQLiteChunkStore,
            SQLiteSourceRegistry,
        )

        storage = LocalStorage(temp_dir)
        embedded_question_store, chunk_store, atom_store, source_registry = storage.build_stores()

        assert isinstance(embedded_question_store, ChromaEmbeddedQuestionStore)
        assert isinstance(chunk_store, SQLiteChunkStore)
        assert isinstance(atom_store, SQLiteAtomStore)
        assert isinstance(source_registry, SQLiteSourceRegistry)

    def test_build_stores_creates_directory(self, temp_dir):
        """Test that LocalStorage creates directory if it doesn't exist."""
        pytest.importorskip("chromadb", reason="This test requires chromadb")
        from isotope.configuration import LocalStorage

        new_dir = os.path.join(temp_dir, "new_storage")
        storage = LocalStorage(new_dir)
        storage.build_stores()

        assert os.path.isdir(new_dir)

    def test_is_frozen_dataclass(self, temp_dir):
        """Test that LocalStorage is immutable."""
        from dataclasses import FrozenInstanceError

        from isotope.configuration import LocalStorage

        storage = LocalStorage(temp_dir)

        with pytest.raises(FrozenInstanceError):
            storage.data_dir = "/other/path"  # type: ignore[misc]


class TestLiteLLMProvider:
    def test_build_embedder_returns_client_embedder(self, temp_dir):
        """Test that LiteLLMProvider.build_embedder() returns ClientEmbedder."""
        pytest.importorskip("litellm", reason="This test requires litellm")
        from isotope.configuration import LiteLLMProvider
        from isotope.embedder import ClientEmbedder

        provider = LiteLLMProvider(
            llm="openai/gpt-4o",
            embedding="openai/text-embedding-3-small",
        )
        embedder = provider.build_embedder()

        assert isinstance(embedder, ClientEmbedder)

    def test_build_atomizer_returns_llm_atomizer_by_default(self, temp_dir):
        """Test that LiteLLMProvider.build_atomizer() returns LLMAtomizer by default."""
        pytest.importorskip("litellm", reason="This test requires litellm")
        from isotope.atomizer import LLMAtomizer
        from isotope.configuration import LiteLLMProvider
        from isotope.settings import Settings

        provider = LiteLLMProvider(
            llm="openai/gpt-4o",
            embedding="openai/text-embedding-3-small",
        )
        atomizer = provider.build_atomizer(Settings())

        assert isinstance(atomizer, LLMAtomizer)

    def test_build_atomizer_respects_atomizer_type_sentence(self, temp_dir):
        """Test that LiteLLMProvider.build_atomizer() respects atomizer_type='sentence'."""
        pytest.importorskip("litellm", reason="This test requires litellm")
        from isotope.atomizer import SentenceAtomizer
        from isotope.configuration import LiteLLMProvider
        from isotope.settings import Settings

        provider = LiteLLMProvider(
            llm="openai/gpt-4o",
            embedding="openai/text-embedding-3-small",
            atomizer_type="sentence",
        )
        atomizer = provider.build_atomizer(Settings())

        assert isinstance(atomizer, SentenceAtomizer)

    def test_build_question_generator_returns_client_generator(self, temp_dir):
        """Test that LiteLLMProvider.build_question_generator() returns ClientQuestionGenerator."""
        pytest.importorskip("litellm", reason="This test requires litellm")
        from isotope.configuration import LiteLLMProvider
        from isotope.question_generator import ClientQuestionGenerator
        from isotope.settings import Settings

        provider = LiteLLMProvider(
            llm="openai/gpt-4o",
            embedding="openai/text-embedding-3-small",
        )
        generator = provider.build_question_generator(Settings())

        assert isinstance(generator, ClientQuestionGenerator)

    def test_build_question_generator_uses_settings(self, temp_dir):
        """Test that LiteLLMProvider.build_question_generator() uses settings."""
        pytest.importorskip("litellm", reason="This test requires litellm")
        from isotope.configuration import LiteLLMProvider
        from isotope.settings import Settings

        provider = LiteLLMProvider(
            llm="openai/gpt-4o",
            embedding="openai/text-embedding-3-small",
        )
        settings = Settings(questions_per_atom=20)
        generator = provider.build_question_generator(settings)

        assert generator.num_questions == 20

    def test_is_frozen_dataclass(self, temp_dir):
        """Test that LiteLLMProvider is immutable."""
        from dataclasses import FrozenInstanceError

        pytest.importorskip("litellm", reason="This test requires litellm")
        from isotope.configuration import LiteLLMProvider

        provider = LiteLLMProvider(
            llm="openai/gpt-4o",
            embedding="openai/text-embedding-3-small",
        )

        with pytest.raises(FrozenInstanceError):
            provider.llm = "other/model"  # type: ignore[misc]


class TestProtocols:
    def test_litellm_provider_satisfies_protocol(self, temp_dir):
        """Test that LiteLLMProvider satisfies ProviderConfig protocol."""
        pytest.importorskip("litellm", reason="This test requires litellm")
        from isotope.configuration import LiteLLMProvider, ProviderConfig

        provider = LiteLLMProvider(
            llm="openai/gpt-4o",
            embedding="openai/text-embedding-3-small",
        )

        assert isinstance(provider, ProviderConfig)

    def test_local_storage_satisfies_protocol(self, temp_dir):
        """Test that LocalStorage satisfies StorageConfig protocol."""
        pytest.importorskip("chromadb", reason="This test requires chromadb")
        from isotope.configuration import LocalStorage, StorageConfig

        storage = LocalStorage(temp_dir)

        assert isinstance(storage, StorageConfig)

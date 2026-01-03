# src/isotopedb/configuration/__init__.py
"""Configuration objects for IsotopeDB.

This module provides the Configuration Objects pattern for setting up IsotopeDB.
Instead of factory methods, you pass configuration objects that know how to
build their components.

Provider configurations (build AI/ML components):
- LiteLLMProvider: Uses LiteLLM for LLM and embedding calls

Storage configurations (build data stores):
- LocalStorage: Chroma + SQLite for local development

Example:
    from isotopedb import Isotope, LiteLLMProvider, LocalStorage

    iso = Isotope(
        provider=LiteLLMProvider(llm="openai/gpt-4o", embedding="text-embedding-3-small"),
        storage=LocalStorage("./data"),
    )
"""

from isotopedb.configuration.base import ProviderConfig, StorageConfig
from isotopedb.configuration.providers import LiteLLMProvider
from isotopedb.configuration.storage import LocalStorage

__all__ = [
    "ProviderConfig",
    "StorageConfig",
    "LiteLLMProvider",
    "LocalStorage",
]

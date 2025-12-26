# src/isotopedb/dedup/base.py
"""Deduplicator abstract base class."""

from abc import ABC, abstractmethod

from isotopedb.models import Chunk
from isotopedb.stores.base import DocStore


class Deduplicator(ABC):
    """Abstract base class for re-ingestion deduplication."""

    @abstractmethod
    def get_chunks_to_remove(
        self,
        new_chunks: list[Chunk],
        doc_store: DocStore,
    ) -> list[str]:
        """
        Determine which existing chunks should be removed before ingesting new chunks.

        Returns chunk IDs to remove.
        """
        ...

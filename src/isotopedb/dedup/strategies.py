# src/isotopedb/dedup/strategies.py
"""Deduplication strategy implementations."""

from isotopedb.dedup.base import Deduplicator
from isotopedb.models import Chunk
from isotopedb.stores.base import DocStore


class NoDedup(Deduplicator):
    """No deduplication - never removes existing chunks."""

    def get_chunks_to_remove(
        self,
        new_chunks: list[Chunk],
        doc_store: DocStore,
    ) -> list[str]:
        """Return empty list - never remove anything."""
        return []


class SourceAwareDedup(Deduplicator):
    """Remove all existing chunks from the same source(s) as new chunks.

    This strategy ensures that re-ingesting a document replaces all its
    previous chunks, preventing duplicate content from the same source.
    """

    def get_chunks_to_remove(
        self,
        new_chunks: list[Chunk],
        doc_store: DocStore,
    ) -> list[str]:
        """Find all existing chunks from the same sources as new chunks."""
        if not new_chunks:
            return []

        # Get unique sources from new chunks
        sources = {chunk.source for chunk in new_chunks}

        # Find all existing chunk IDs from those sources
        chunk_ids_to_remove = []
        for source in sources:
            existing_chunks = doc_store.get_by_source(source)
            chunk_ids_to_remove.extend(chunk.id for chunk in existing_chunks)

        return chunk_ids_to_remove

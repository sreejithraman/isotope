"""Tests for ChunkEmbeddingStore ABC."""

from isotope.stores.base import ChunkEmbeddingStore


class TestChunkEmbeddingStoreABC:
    def test_cannot_instantiate(self):
        """ChunkEmbeddingStore is abstract and cannot be instantiated."""
        import pytest

        with pytest.raises(TypeError):
            ChunkEmbeddingStore()  # type: ignore[abstract]

    def test_has_required_methods(self):
        """ChunkEmbeddingStore defines the expected abstract methods."""
        assert hasattr(ChunkEmbeddingStore, "add")
        assert hasattr(ChunkEmbeddingStore, "search")
        assert hasattr(ChunkEmbeddingStore, "delete_by_chunk_ids")
        assert hasattr(ChunkEmbeddingStore, "count")

"""Tests for ChunkEmbeddingStore ABC."""

from abc import ABC

from isotope.stores.base import ChunkEmbeddingStore


class _ChunkEmbeddingStoreCanary(ChunkEmbeddingStore, ABC):
    """Minimal concrete subclass to verify the ABC contract is implementable."""

    def add(self, chunk_ids: list[str], embeddings: list[list[float]]) -> None:
        self._last_added = (chunk_ids, embeddings)

    def search(self, embedding: list[float], k: int = 5) -> list[tuple[str, float]]:
        return [("chunk-1", 0.99)][:k]

    def delete_by_chunk_ids(self, chunk_ids: list[str]) -> None:
        self._last_deleted = chunk_ids

    def count(self) -> int:
        return 0


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

    def test_concrete_subclass_can_implement_contract(self):
        """A concrete subclass implementing all abstract methods is instantiable."""
        store = _ChunkEmbeddingStoreCanary()
        store.add(["c1"], [[1.0, 0.0]])
        assert store.search([1.0, 0.0], k=1) == [("chunk-1", 0.99)]
        store.delete_by_chunk_ids(["c1"])
        assert store.count() == 0

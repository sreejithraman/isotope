# src/isotope/stores/chroma.py
"""ChromaDB embedded question store implementation."""

from pathlib import Path

import chromadb

from isotope.models import EmbeddedQuestion, Question
from isotope.stores.base import EmbeddedQuestionStore


class ChromaEmbeddedQuestionStore(EmbeddedQuestionStore):
    """ChromaDB-based embedded question store."""

    def __init__(self, persist_dir: str, collection_name: str = "isotope") -> None:
        """Initialize the ChromaDB store."""
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def close(self) -> None:
        """Close the store and release resources.

        ChromaDB doesn't have an official close method, so we call the internal
        _system.stop() to release file handles. This is necessary to avoid
        'too many open files' errors in test suites.

        See: https://github.com/chroma-core/chroma/issues/5868
        """
        self._collection = None  # type: ignore[assignment]

        # ChromaDB lacks official close() - use internal _system.stop() workaround
        try:
            if self._client is not None and hasattr(self._client, "_system"):
                self._client._system.stop()
        except Exception:
            pass  # Best effort cleanup

        self._client = None  # type: ignore[assignment]

    def add(self, questions: list[EmbeddedQuestion]) -> None:
        """Add questions with embeddings to the store."""
        if not questions:
            return

        self._collection.add(
            ids=[eq.question.id for eq in questions],
            embeddings=[eq.embedding for eq in questions],  # type: ignore[arg-type]
            metadatas=[
                {
                    "text": eq.question.text,
                    "chunk_id": eq.question.chunk_id,
                    "atom_id": eq.question.atom_id,
                }
                for eq in questions
            ],
        )

    def search(self, embedding: list[float], k: int = 5) -> list[tuple[Question, float]]:
        """Search for similar questions."""
        if self._collection.count() == 0:
            return []

        results = self._collection.query(
            query_embeddings=[embedding],  # type: ignore[arg-type]
            n_results=min(k, self._collection.count()),
            include=["metadatas", "distances"],
        )

        questions_with_scores = []
        ids = results["ids"][0]
        metadatas = results["metadatas"][0]  # type: ignore[index]
        distances = results["distances"][0]  # type: ignore[index]

        for qid, meta, dist in zip(ids, metadatas, distances, strict=True):
            question = Question(
                id=qid,
                text=meta["text"],
                chunk_id=meta["chunk_id"],
                atom_id=meta["atom_id"],
            )
            # ChromaDB returns distance; convert to similarity score
            # For cosine distance: similarity = 1 - distance
            score = 1.0 - dist
            questions_with_scores.append((question, score))

        return questions_with_scores

    def delete_by_chunk_ids(self, chunk_ids: list[str]) -> None:
        """Delete all questions associated with the given chunk IDs."""
        if not chunk_ids:
            return

        self._collection.delete(where={"chunk_id": {"$in": chunk_ids}})  # type: ignore[dict-item]

    def list_chunk_ids(self) -> set[str]:
        """List all unique chunk IDs in the store."""
        if self._collection.count() == 0:
            return set()

        results = self._collection.get(include=["metadatas"])
        metadatas = results["metadatas"] or []
        return {str(meta["chunk_id"]) for meta in metadatas}

    def count_questions(self) -> int:
        """Count the total number of questions in the store."""
        return self._collection.count()

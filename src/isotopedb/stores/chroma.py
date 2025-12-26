# src/isotopedb/stores/chroma.py
"""ChromaDB vector store implementation."""

from pathlib import Path

import chromadb

from isotopedb.models import EmbeddedQuestion, Question
from isotopedb.stores.base import VectorStore


class ChromaVectorStore(VectorStore):
    """ChromaDB-based vector store."""

    def __init__(self, persist_dir: str, collection_name: str = "isotope") -> None:
        """Initialize the ChromaDB store."""
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, questions: list[EmbeddedQuestion]) -> None:
        """Add questions with embeddings to the store."""
        if not questions:
            return

        self._collection.add(
            ids=[eq.question.id for eq in questions],
            embeddings=[eq.embedding for eq in questions],
            metadatas=[
                {
                    "text": eq.question.text,
                    "chunk_id": eq.question.chunk_id,
                    "atom_id": eq.question.atom_id or "",
                }
                for eq in questions
            ],
        )

    def search(self, embedding: list[float], k: int = 5) -> list[tuple[Question, float]]:
        """Search for similar questions."""
        if self._collection.count() == 0:
            return []

        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(k, self._collection.count()),
            include=["metadatas", "distances"],
        )

        questions_with_scores = []
        ids = results["ids"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for qid, meta, dist in zip(ids, metadatas, distances):
            question = Question(
                id=qid,
                text=meta["text"],
                chunk_id=meta["chunk_id"],
                atom_id=meta["atom_id"] if meta["atom_id"] else None,
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

        self._collection.delete(where={"chunk_id": {"$in": chunk_ids}})

    def list_chunk_ids(self) -> set[str]:
        """List all unique chunk IDs in the store."""
        if self._collection.count() == 0:
            return set()

        results = self._collection.get(include=["metadatas"])
        return {meta["chunk_id"] for meta in results["metadatas"]}

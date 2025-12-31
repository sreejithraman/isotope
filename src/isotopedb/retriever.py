"""Retrieval pipeline for Isotope."""

from isotopedb.embedder import Embedder
from isotopedb.models import SearchResult
from isotopedb.stores import DocStore, VectorStore


class Retriever:
    """Orchestrates the retrieval pipeline.

    Pipeline:
    1. Embed query
    2. Search vector store for similar questions
    3. Fetch corresponding chunks from doc store
    4. Return SearchResults with question, chunk, and score
    """

    def __init__(
        self,
        vector_store: VectorStore,
        doc_store: DocStore,
        embedder: Embedder,
        default_k: int = 5,
    ) -> None:
        """Initialize the retriever.

        Args:
            vector_store: Vector store for question search
            doc_store: Document store for chunk retrieval
            embedder: Embedder for query embedding
            default_k: Default number of results to return
        """
        self.vector_store = vector_store
        self.doc_store = doc_store
        self.embedder = embedder
        self.default_k = default_k

    def search(self, query: str, k: int | None = None) -> list[SearchResult]:
        """Search for relevant chunks matching the query.

        Args:
            query: User's search query
            k: Number of results to return (default: self.default_k)

        Returns:
            List of SearchResult objects ordered by relevance
        """
        k = k or self.default_k

        # Step 1: Embed query
        query_embedding = self.embedder.embed_text(query)

        # Step 2: Search vector store
        question_scores = self.vector_store.search(query_embedding, k=k)

        if not question_scores:
            return []

        # Step 3: Fetch chunks
        results = []
        for question, score in question_scores:
            chunk = self.doc_store.get(question.chunk_id)
            if chunk:
                results.append(SearchResult(
                    question=question,
                    chunk=chunk,
                    score=score,
                ))

        return results

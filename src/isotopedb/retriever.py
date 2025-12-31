"""Retrieval pipeline for Isotope."""

import litellm

from isotopedb.embedder import Embedder
from isotopedb.models import QueryResponse, SearchResult
from isotopedb.stores import DocStore, VectorStore


SYNTHESIS_PROMPT = """Based on the following context, answer the user's question.
If the context doesn't contain enough information to answer, say so.

Context:
{context}

Question: {query}

Answer:"""


class Retriever:
    """Orchestrates the retrieval pipeline."""

    def __init__(
        self,
        vector_store: VectorStore,
        doc_store: DocStore,
        embedder: Embedder,
        default_k: int = 5,
        llm_model: str | None = None,
        synthesis_prompt: str | None = None,
    ) -> None:
        """Initialize the retriever.

        Args:
            vector_store: Vector store for question search
            doc_store: Document store for chunk retrieval
            embedder: Embedder for query embedding
            default_k: Default number of results to return
            llm_model: LLM model for answer synthesis (optional)
            synthesis_prompt: Custom synthesis prompt template
        """
        self.vector_store = vector_store
        self.doc_store = doc_store
        self.embedder = embedder
        self.default_k = default_k
        self.llm_model = llm_model
        self.synthesis_prompt = synthesis_prompt or SYNTHESIS_PROMPT

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

    def query(
        self,
        query: str,
        k: int | None = None,
        synthesize: bool = True,
    ) -> QueryResponse:
        """Query the database and optionally synthesize an answer.

        Args:
            query: User's question
            k: Number of results to retrieve
            synthesize: Whether to generate an LLM answer (default: True)

        Returns:
            QueryResponse with results and optional synthesized answer
        """
        results = self.search(query, k=k)

        answer = None
        if synthesize and results and self.llm_model:
            answer = self._synthesize_answer(query, results)

        return QueryResponse(
            query=query,
            answer=answer,
            results=results,
        )

    def _synthesize_answer(self, query: str, results: list[SearchResult]) -> str:
        """Synthesize an answer from the search results using an LLM."""
        # Build context from results
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(f"[{i}] {result.chunk.content}")

        context = "\n\n".join(context_parts)

        prompt = self.synthesis_prompt.format(
            context=context,
            query=query,
        )

        response = litellm.completion(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
        )

        return response.choices[0].message.content.strip()

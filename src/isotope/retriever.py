"""Retrieval pipeline for Isotope."""

from isotope.embedder import Embedder
from isotope.models import QueryResponse, SearchResult
from isotope.providers import LLMClient
from isotope.stores import AtomStore, ChunkStore, EmbeddedQuestionStore

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
        embedded_question_store: EmbeddedQuestionStore,
        chunk_store: ChunkStore,
        atom_store: AtomStore,
        embedder: Embedder,
        default_k: int = 5,
        llm_client: LLMClient | None = None,
        synthesis_prompt: str | None = None,
        synthesis_temperature: float | None = 0.3,
    ) -> None:
        """Initialize the retriever.

        Args:
            embedded_question_store: Store for embedded question search
            chunk_store: Chunk store for chunk retrieval
            atom_store: Atom store for atom retrieval
            embedder: Embedder for query embedding
            default_k: Default number of results to return
            llm_client: LLM client for answer synthesis (optional)
            synthesis_prompt: Custom synthesis prompt template
            synthesis_temperature: Temperature for synthesis LLM calls
        """
        self.embedded_question_store = embedded_question_store
        self.chunk_store = chunk_store
        self.atom_store = atom_store
        self.embedder = embedder
        self.default_k = default_k
        self._llm_client = llm_client
        self.synthesis_prompt = synthesis_prompt or SYNTHESIS_PROMPT
        self.synthesis_temperature = synthesis_temperature

    def get_context(self, query: str, k: int | None = None) -> list[SearchResult]:
        """Get relevant context (chunks/atoms) for a query.

        Args:
            query: User's search query
            k: Number of results to return (default: self.default_k)

        Returns:
            List of SearchResult objects ordered by relevance
        """
        k = self.default_k if k is None else k

        # Step 1: Embed query
        query_embedding = self.embedder.embed_text(query)

        # Step 2: Search embedded question store
        question_scores = self.embedded_question_store.search(query_embedding, k=k)

        if not question_scores:
            return []

        # Step 3: Fetch chunks and atoms
        results = []
        for question, score in question_scores:
            chunk = self.chunk_store.get(question.chunk_id)
            atom = self.atom_store.get(question.atom_id)

            if chunk and atom:
                results.append(
                    SearchResult(
                        question=question,
                        chunk=chunk,
                        score=score,
                        atom=atom,
                    )
                )

        return results

    def get_answer(self, query: str, k: int | None = None) -> QueryResponse:
        """Get an answer to a query using retrieved context.

        If llm_client is configured, synthesizes an answer from the context.
        Otherwise, returns QueryResponse with answer=None.

        Args:
            query: User's question
            k: Number of results to retrieve

        Returns:
            QueryResponse with results and synthesized answer (if llm_client set)
        """
        results = self.get_context(query, k=k)

        answer = None
        if results and self._llm_client:
            answer = self._synthesize_answer(query, results)

        return QueryResponse(
            query=query,
            answer=answer,
            results=results,
        )

    def _synthesize_answer(self, query: str, results: list[SearchResult]) -> str:
        """Synthesize an answer from the search results using an LLM."""
        if self._llm_client is None:
            return ""

        # Build context from results
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(f"[{i}] {result.chunk.content}")

        context = "\n\n".join(context_parts)

        prompt = self.synthesis_prompt.format(
            context=context,
            query=query,
        )

        return self._llm_client.complete(
            messages=[{"role": "user", "content": prompt}],
            temperature=self.synthesis_temperature,
        ).strip()

# src/isotopedb/embedder/litellm_embedder.py
"""LiteLLM-based embedder implementation."""

import litellm

from isotopedb.embedder.base import Embedder
from isotopedb.models import EmbeddedQuestion, Question


class LiteLLMEmbedder(Embedder):
    """LiteLLM-based embedder for generating embeddings.

    Converts Questions into EmbeddedQuestions by adding embedding vectors.
    Uses LiteLLM to support multiple embedding providers.
    """

    def __init__(self, model: str = "gemini/text-embedding-004") -> None:
        """Initialize the embedder.

        Args:
            model: LiteLLM embedding model identifier.
                   Examples: "gemini/text-embedding-004", "openai/text-embedding-3-small"
        """
        self.model = model

    def embed_text(self, text: str) -> list[float]:
        """Generate an embedding vector for a single text."""
        response = litellm.embedding(
            model=self.model,
            input=[text],
        )
        return response.data[0]["embedding"]  # type: ignore[no-any-return]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embedding vectors for multiple texts (batched)."""
        if not texts:
            return []

        response = litellm.embedding(
            model=self.model,
            input=texts,
        )
        # Sort by index to maintain order
        sorted_data = sorted(response.data, key=lambda x: x["index"])
        return [item["embedding"] for item in sorted_data]

    def embed_question(self, question: Question) -> EmbeddedQuestion:
        """Embed a single question."""
        embedding = self.embed_text(question.text)
        return EmbeddedQuestion(question=question, embedding=embedding)

    def embed_questions(self, questions: list[Question]) -> list[EmbeddedQuestion]:
        """Embed multiple questions (batched for efficiency)."""
        if not questions:
            return []

        texts = [q.text for q in questions]
        embeddings = self.embed_texts(texts)

        return [
            EmbeddedQuestion(question=q, embedding=emb)
            for q, emb in zip(questions, embeddings, strict=True)
        ]

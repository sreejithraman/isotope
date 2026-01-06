# src/isotope/configuration/providers/litellm.py
"""LiteLLM provider configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from isotope.atomizer import Atomizer
    from isotope.embedder import Embedder
    from isotope.providers import LLMClient
    from isotope.question_generator import QuestionGenerator
    from isotope.settings import Settings


@dataclass(frozen=True)
class LiteLLMProvider:
    """Provider configuration using LiteLLM for LLM and embedding calls.

    LiteLLM provides a unified interface to 100+ LLM providers including
    OpenAI, Anthropic, Azure, Bedrock, and more.

    Args:
        llm: LiteLLM model identifier for LLM calls (atomization, question generation).
             Examples: "openai/gpt-5-mini-2025-08-07", "anthropic/claude-sonnet-4-5-20250929"
        embedding: LiteLLM model identifier for embeddings.
                   Examples: "openai/text-embedding-3-small", "openai/text-embedding-3-large"
        atomizer_type: Type of atomizer to use. "llm" uses the LLM for intelligent
                       atomization, "sentence" uses simple sentence splitting.
                       Default: "llm"

    Example:
        provider = LiteLLMProvider(
            llm="openai/gpt-5-mini-2025-08-07",
            embedding="openai/text-embedding-3-small",
        )

        # With sentence atomizer (faster, no LLM calls for atomization)
        provider = LiteLLMProvider(
            llm="openai/gpt-5-mini-2025-08-07",
            embedding="openai/text-embedding-3-small",
            atomizer_type="sentence",
        )
    """

    llm: str
    embedding: str
    atomizer_type: Literal["llm", "sentence"] = "llm"

    def build_embedder(self, settings: Settings) -> Embedder:
        """Build a ClientEmbedder using LiteLLM embedding client.

        Args:
            settings: Settings containing num_retries for rate limit handling.
        """
        from isotope.embedder import ClientEmbedder
        from isotope.providers.litellm import LiteLLMEmbeddingClient

        embedding_client = LiteLLMEmbeddingClient(
            model=self.embedding,
            num_retries=settings.num_retries,
        )
        return ClientEmbedder(embedding_client=embedding_client)

    def build_atomizer(self, settings: Settings) -> Atomizer:
        """Build an atomizer based on atomizer_type.

        Args:
            settings: Settings containing atomizer_prompt and num_retries.

        Returns:
            LLMAtomizer if atomizer_type="llm", SentenceAtomizer otherwise.
        """
        from isotope.atomizer import LLMAtomizer, SentenceAtomizer
        from isotope.providers.litellm import LiteLLMClient

        if self.atomizer_type == "sentence":
            return SentenceAtomizer()

        llm_client = LiteLLMClient(model=self.llm, num_retries=settings.num_retries)
        return LLMAtomizer(
            llm_client=llm_client,
            prompt_template=settings.atomizer_prompt,
        )

    def build_question_generator(self, settings: Settings) -> QuestionGenerator:
        """Build a ClientQuestionGenerator using LiteLLM client.

        Args:
            settings: Settings containing questions_per_atom,
                      question_generator_prompt, and num_retries.
        """
        from isotope.providers.litellm import LiteLLMClient
        from isotope.question_generator import ClientQuestionGenerator

        llm_client = LiteLLMClient(model=self.llm, num_retries=settings.num_retries)
        return ClientQuestionGenerator(
            llm_client=llm_client,
            num_questions=settings.questions_per_atom,
            prompt_template=settings.question_generator_prompt,
        )

    def build_llm_client(self, settings: Settings | None = None) -> LLMClient:
        """Build a LiteLLMClient for general-purpose LLM calls.

        Args:
            settings: Optional settings containing num_retries. If None,
                     uses default retry value.
        """
        from isotope.providers.litellm import LiteLLMClient

        num_retries = settings.num_retries if settings else 3
        return LiteLLMClient(model=self.llm, num_retries=num_retries)

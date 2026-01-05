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
             Examples: "openai/gpt-4o", "anthropic/claude-sonnet-4-5-20250929"
        embedding: LiteLLM model identifier for embeddings.
                   Examples: "openai/text-embedding-3-small", "openai/text-embedding-3-large"
        atomizer_type: Type of atomizer to use. "llm" uses the LLM for intelligent
                       atomization, "sentence" uses simple sentence splitting.
                       Default: "llm"

    Example:
        provider = LiteLLMProvider(
            llm="openai/gpt-4o",
            embedding="openai/text-embedding-3-small",
        )

        # With sentence atomizer (faster, no LLM calls for atomization)
        provider = LiteLLMProvider(
            llm="openai/gpt-4o",
            embedding="openai/text-embedding-3-small",
            atomizer_type="sentence",
        )
    """

    llm: str
    embedding: str
    atomizer_type: Literal["llm", "sentence"] = "llm"

    def build_embedder(self) -> Embedder:
        """Build a ClientEmbedder using LiteLLM embedding client."""
        from isotope.embedder import ClientEmbedder
        from isotope.providers.litellm import LiteLLMEmbeddingClient

        embedding_client = LiteLLMEmbeddingClient(model=self.embedding)
        return ClientEmbedder(embedding_client=embedding_client)

    def build_atomizer(self, settings: Settings) -> Atomizer:
        """Build an atomizer based on atomizer_type.

        Args:
            settings: Settings containing atomizer_prompt if customized.

        Returns:
            LLMAtomizer if atomizer_type="llm", SentenceAtomizer otherwise.
        """
        from isotope.atomizer import LLMAtomizer, SentenceAtomizer
        from isotope.providers.litellm import LiteLLMClient

        if self.atomizer_type == "sentence":
            return SentenceAtomizer()

        llm_client = LiteLLMClient(model=self.llm)
        return LLMAtomizer(
            llm_client=llm_client,
            prompt_template=settings.atomizer_prompt,
        )

    def build_question_generator(self, settings: Settings) -> QuestionGenerator:
        """Build a ClientQuestionGenerator using LiteLLM client.

        Args:
            settings: Settings containing questions_per_atom and
                      question_generator_prompt if customized.
        """
        from isotope.providers.litellm import LiteLLMClient
        from isotope.question_generator import ClientQuestionGenerator

        llm_client = LiteLLMClient(model=self.llm)
        return ClientQuestionGenerator(
            llm_client=llm_client,
            num_questions=settings.questions_per_atom,
            prompt_template=settings.question_generator_prompt,
        )

    def build_llm_client(self) -> LLMClient:
        """Build a LiteLLMClient for general-purpose LLM calls."""
        from isotope.providers.litellm import LiteLLMClient

        return LiteLLMClient(model=self.llm)

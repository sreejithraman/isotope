# src/isotope/providers/litellm/models.py
"""Curated LLM model constants for LiteLLM provider.

These are convenience constants for IDE autocomplete when using LiteLLM.
You can always pass any valid LiteLLM model string directly.

Example:
    from isotope.atomizer import LLMAtomizer
    from isotope.providers.litellm import ChatModels, LiteLLMClient

    # Using constants (IDE autocomplete works)
    llm_client = LiteLLMClient(model=ChatModels.CLAUDE_SONNET_45)
    atomizer = LLMAtomizer(llm_client=llm_client)

    # Custom models still work
    llm_client = LiteLLMClient(model="my-custom/model")
    atomizer = LLMAtomizer(llm_client=llm_client)
"""


class ChatModels:
    """Chat/completion models for LLMAtomizer and ClientQuestionGenerator (via LiteLLMClient)."""

    # OpenAI - GPT 5 Series
    GPT_52 = "openai/gpt-5.2"
    GPT_52_PRO = "openai/gpt-5.2-pro"
    GPT_5_MINI = "openai/gpt-5-mini"
    GPT_5_NANO = "openai/gpt-5-nano"

    # Anthropic - Claude 4.5 Series
    CLAUDE_SONNET_45 = "anthropic/claude-sonnet-4-5-20250929"
    CLAUDE_HAIKU_45 = "anthropic/claude-haiku-4-5-20251001"
    CLAUDE_OPUS_45 = "anthropic/claude-opus-4-5-20251101"

    # Google Gemini 3
    GEMINI_3_PRO = "gemini/gemini-3-pro-preview"
    GEMINI_3_FLASH = "gemini/gemini-3-flash-preview"

    # AWS Bedrock - Anthropic Claude 4.5
    BEDROCK_CLAUDE_SONNET_45 = "bedrock/anthropic.claude-sonnet-4-5-20250929-v1:0"
    BEDROCK_CLAUDE_HAIKU_45 = "bedrock/anthropic.claude-haiku-4-5-20251001-v1:0"
    BEDROCK_CLAUDE_OPUS_45 = "bedrock/anthropic.claude-opus-4-5-20251101-v1:0"

    # AWS Bedrock - DeepSeek
    BEDROCK_DEEPSEEK_R1 = "bedrock/deepseek.r1-v1:0"
    BEDROCK_DEEPSEEK_V31 = "bedrock/deepseek.v3-v1:0"

    # AWS Bedrock - Kimi (Moonshot AI)
    BEDROCK_KIMI_K2 = "bedrock/moonshot.kimi-k2-thinking"


class EmbeddingModels:
    """Embedding models for LiteLLMEmbeddingClient / ClientEmbedder."""

    # OpenAI
    TEXT_3_SMALL = "openai/text-embedding-3-small"
    TEXT_3_LARGE = "openai/text-embedding-3-large"

    # Google Gemini
    GEMINI_004 = "gemini/text-embedding-004"

    # AWS Bedrock
    BEDROCK_TITAN_V2 = "bedrock/amazon.titan-embed-text-v2:0"
    BEDROCK_COHERE_V3 = "bedrock/cohere.embed-english-v3"

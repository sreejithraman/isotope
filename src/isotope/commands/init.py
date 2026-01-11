# src/isotope/commands/init.py
"""Init command - initialize isotope configuration.

This module provides the init logic that both CLI and TUI use.
It uses callbacks for interactive prompts, allowing each UI to
handle user input in its own way.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from isotope.commands.base import (
    InitPrompt,
    InitResult,
    PromptCallback,
    PromptRequest,
)
from isotope.config import is_local_model
from isotope.providers.litellm.models import ChatModels, EmbeddingModels

# Default model suggestions (from models.py - single source of truth)
DEFAULT_LLM_MODEL = ChatModels.GPT_5_MINI
DEFAULT_EMBEDDING_MODEL = EmbeddingModels.TEXT_3_SMALL

# Model choices for selection (cloud models from models.py, Ollama as string)
LLM_MODEL_CHOICES = [
    ChatModels.GPT_5_MINI,
    ChatModels.CLAUDE_SONNET_45,
    ChatModels.GEMINI_3_FLASH,
    "ollama/llama3.2",  # Local model - not in models.py
]

EMBEDDING_MODEL_CHOICES = [
    EmbeddingModels.TEXT_3_SMALL,
    EmbeddingModels.TEXT_3_LARGE,
    EmbeddingModels.GEMINI_EMBEDDING_001,
    "ollama/nomic-embed-text",  # Local model - not in models.py
]


class InitCancelled(Exception):
    """Raised when init is cancelled by user."""


def get_settings_for_init(
    is_local: bool,
    rate_limited: bool | None,
    priority: str,
) -> dict[str, Any]:
    """Get settings based on user's environment and priority.

    Args:
        is_local: Whether the model is local (Ollama, etc.)
        rate_limited: Whether API is rate-limited (None if not applicable)
        priority: User's priority ("speed", "quality", or "balanced")

    Returns:
        Dictionary of settings that differ from defaults
    """
    if is_local:
        # Local models: low concurrency, sentence atomizer for speed
        return {
            "use_sentence_atomizer": True,
            "questions_per_atom": 5,
            "max_concurrent_llm_calls": 1,
        }

    if rate_limited:
        if priority == "speed":
            return {
                "use_sentence_atomizer": True,
                "questions_per_atom": 3,
                "max_concurrent_llm_calls": 2,
            }
        elif priority == "quality":
            return {
                "use_sentence_atomizer": False,
                "questions_per_atom": 5,
                "max_concurrent_llm_calls": 2,
            }
        else:  # balanced
            return {
                "use_sentence_atomizer": False,
                "questions_per_atom": 5,
                "max_concurrent_llm_calls": 2,
            }
    else:
        # High-limit APIs
        if priority == "speed":
            return {
                "use_sentence_atomizer": True,
                "questions_per_atom": 5,
                "max_concurrent_llm_calls": 10,
            }
        elif priority == "quality":
            return {
                "use_sentence_atomizer": False,
                "questions_per_atom": 10,
                "max_concurrent_llm_calls": 10,
            }
        else:  # balanced - these are the defaults
            return {}


def generate_config_content(
    llm_model: str,
    embedding_model: str,
    settings: dict[str, Any],
) -> str:
    """Generate isotope.yaml content with settings.

    Args:
        llm_model: LLM model name
        embedding_model: Embedding model name
        settings: Settings to include in config

    Returns:
        YAML content as string
    """
    content = f"""# Isotope Configuration
provider: litellm

# LiteLLM model identifiers
llm_model: {llm_model}
embedding_model: {embedding_model}
"""

    if settings:
        content += "\n# Settings (configured based on your environment)\nsettings:\n"
        for key, value in settings.items():
            if isinstance(value, bool):
                content += f"  {key}: {str(value).lower()}\n"
            else:
                content += f"  {key}: {value}\n"

    content += """
# Uncomment to customize (showing defaults):
#   use_sentence_atomizer: false  # true = fast, false = LLM quality
#   questions_per_atom: 5         # more = better recall, higher cost
#   diversity_scope: global       # global | per_chunk | per_atom
#   max_concurrent_llm_calls: 10  # parallel LLM requests
#
# Advanced settings:
#   num_retries: 5
#   question_diversity_threshold: 0.85
#   default_k: 5
"""

    return content


def generate_custom_config_content() -> str:
    """Generate isotope.yaml content for custom provider."""
    return """# Isotope Configuration
provider: custom

# Custom implementation classes (dotted import paths)
embedder: my_package.MyEmbedder
question_generator: my_package.MyGenerator
atomizer: my_package.MyAtomizer

# Optional: kwargs passed to each class
# embedder_kwargs:
#   region: us-east-1
# question_generator_kwargs: {}
# atomizer_kwargs: {}

# Optional: Data directory
# data_dir: ./isotope_data
"""


def save_api_key_to_env(env_var: str, api_key: str, env_path: Path) -> None:
    """Save API key to .env file.

    Args:
        env_var: Environment variable name
        api_key: API key value
        env_path: Path to .env file
    """
    existing_lines: list[str] = []
    if env_path.exists():
        with open(env_path, encoding="utf-8") as f:
            existing_lines = f.readlines()

    key_exists = False
    new_lines = []
    for line in existing_lines:
        if line.strip().startswith(f"{env_var}="):
            new_lines.append(f"{env_var}={api_key}\n")
            key_exists = True
        else:
            new_lines.append(line)

    if not key_exists:
        if new_lines and not new_lines[-1].endswith("\n"):
            new_lines.append("\n")
        new_lines.append(f"{env_var}={api_key}\n")

    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)


def update_gitignore_for_env(gitignore_path: Path) -> bool:
    """Ensure .env is in .gitignore.

    Args:
        gitignore_path: Path to .gitignore

    Returns:
        True if .gitignore was updated
    """
    if gitignore_path.exists():
        content = gitignore_path.read_text(encoding="utf-8")
        # Check line-by-line for exact .env entry (not substring like .env.example)
        lines = content.splitlines()
        if not any(line.strip() == ".env" for line in lines):
            with open(gitignore_path, "a", encoding="utf-8") as f:
                f.write("\n# Environment variables\n.env\n")
            return True
    return False


def init(
    on_prompt: PromptCallback,
    provider: str = "litellm",
    llm_model: str | None = None,
    embedding_model: str | None = None,
    config_dir: str | Path = ".",
) -> InitResult:
    """Initialize isotope configuration.

    This function uses callbacks for interactive prompts, allowing
    both CLI and TUI to provide their own input handling.

    Args:
        on_prompt: Callback to get user input
        provider: Provider type ("litellm" or "custom")
        llm_model: Pre-set LLM model (skips prompt if provided)
        embedding_model: Pre-set embedding model (skips prompt if provided)
        config_dir: Directory to create config in

    Returns:
        InitResult with paths to created files

    Raises:
        InitCancelled: If user cancels the initialization
    """
    config_dir = Path(config_dir)
    config_path = config_dir / "isotope.yaml"
    env_path = config_dir / ".env"
    gitignore_path = config_dir / ".gitignore"

    # Check if config exists
    if config_path.exists():
        response = on_prompt(
            PromptRequest(
                prompt_type=InitPrompt.OVERWRITE_CONFIG,
                message="Config file already exists. Overwrite?",
                choices=["yes", "no"],
                default="no",
            )
        )
        if response.lower() != "yes":
            raise InitCancelled()

    if provider == "litellm":
        # Get LLM model
        effective_llm = llm_model
        if not effective_llm:
            effective_llm = on_prompt(
                PromptRequest(
                    prompt_type=InitPrompt.LLM_MODEL,
                    message="Select your LLM model",
                    choices=LLM_MODEL_CHOICES,
                    default=DEFAULT_LLM_MODEL,
                )
            )

        # Get embedding model
        effective_embedding = embedding_model
        if not effective_embedding:
            effective_embedding = on_prompt(
                PromptRequest(
                    prompt_type=InitPrompt.EMBEDDING_MODEL,
                    message="Select your embedding model",
                    choices=EMBEDDING_MODEL_CHOICES,
                    default=DEFAULT_EMBEDDING_MODEL,
                )
            )

        # Detect if local model
        is_local = is_local_model(effective_llm)

        rate_limited: bool | None = None
        priority = "balanced"

        if not is_local:
            # Ask about rate limits
            rate_choice = on_prompt(
                PromptRequest(
                    prompt_type=InitPrompt.RATE_LIMIT,
                    message="Are you on a rate-limited or free tier API?",
                    choices=[
                        "Yes - configure for rate limits",
                        "No - I have high rate limits",
                        "Not sure - use safe defaults",
                    ],
                    default="Not sure - use safe defaults",
                )
            )
            rate_limited = "No -" not in rate_choice

            # Ask about priority
            priority_choice = on_prompt(
                PromptRequest(
                    prompt_type=InitPrompt.PRIORITY,
                    message="What's your priority?",
                    choices=[
                        "Retrieval quality (slower, more API calls)",
                        "Speed & cost savings (faster, fewer calls)",
                        "Balanced",
                    ],
                    default="Balanced",
                )
            )
            if "quality" in priority_choice.lower():
                priority = "quality"
            elif "speed" in priority_choice.lower():
                priority = "speed"
            else:
                priority = "balanced"

        # Get settings based on choices
        settings = get_settings_for_init(is_local, rate_limited, priority)

        # Generate and write config
        content = generate_config_content(effective_llm, effective_embedding, settings)
        config_path.write_text(content, encoding="utf-8")

        result = InitResult(
            success=True,
            config_path=str(config_path),
            provider=provider,
            llm_model=effective_llm,
            embedding_model=effective_embedding,
        )

        # Collect API keys (if not local)
        if not is_local:
            llm_key = on_prompt(
                PromptRequest(
                    prompt_type=InitPrompt.API_KEY_LLM,
                    message="Enter your LLM API key (leave empty if not needed)",
                    is_secret=True,
                    default="",
                )
            )
            llm_key_stripped: str | None = llm_key.strip() if llm_key else None

            embed_key = None
            if llm_key:
                embed_choice = on_prompt(
                    PromptRequest(
                        prompt_type=InitPrompt.API_KEY_SAME,
                        message="Embedding API key",
                        choices=[
                            "Same as LLM",
                            "None (not needed)",
                            "Different key",
                        ],
                        default="Same as LLM",
                    )
                )

                if embed_choice == "Same as LLM":
                    embed_key = llm_key_stripped
                elif embed_choice == "Different key":
                    embed_key = on_prompt(
                        PromptRequest(
                            prompt_type=InitPrompt.API_KEY_EMBEDDING,
                            message="Enter embedding API key",
                            is_secret=True,
                            default="",
                        )
                    )
                    embed_key = embed_key.strip() if embed_key else None

            # Save API keys
            if llm_key_stripped:
                save_api_key_to_env("ISOTOPE_LLM_API_KEY", llm_key_stripped, env_path)
                os.environ["ISOTOPE_LLM_API_KEY"] = llm_key_stripped
                result.env_path = str(env_path)

            if embed_key and embed_key != llm_key_stripped:
                save_api_key_to_env("ISOTOPE_EMBEDDING_API_KEY", embed_key, env_path)
                os.environ["ISOTOPE_EMBEDDING_API_KEY"] = embed_key
                result.env_path = str(env_path)

            if result.env_path:
                update_gitignore_for_env(gitignore_path)

        return result

    else:
        # Custom provider
        content = generate_custom_config_content()
        config_path.write_text(content, encoding="utf-8")

        return InitResult(
            success=True,
            config_path=str(config_path),
            provider=provider,
            llm_model="",
            embedding_model="",
        )


def init_non_interactive(
    provider: str = "litellm",
    llm_model: str = DEFAULT_LLM_MODEL,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    config_dir: str | Path = ".",
    overwrite: bool = False,
) -> InitResult:
    """Initialize isotope configuration without interactive prompts.

    This is useful for scripts and automated setups.

    Args:
        provider: Provider type
        llm_model: LLM model name
        embedding_model: Embedding model name
        config_dir: Directory to create config in
        overwrite: If True, overwrite existing config

    Returns:
        InitResult with paths to created files
    """
    config_dir = Path(config_dir)
    config_path = config_dir / "isotope.yaml"

    if config_path.exists() and not overwrite:
        return InitResult(
            success=False,
            error="Config file already exists. Set overwrite=True to replace.",
        )

    if provider == "litellm":
        is_local = is_local_model(llm_model)
        settings = get_settings_for_init(is_local, rate_limited=None, priority="balanced")
        content = generate_config_content(llm_model, embedding_model, settings)
    else:
        content = generate_custom_config_content()

    config_path.write_text(content, encoding="utf-8")

    return InitResult(
        success=True,
        config_path=str(config_path),
        provider=provider,
        llm_model=llm_model if provider == "litellm" else "",
        embedding_model=embedding_model if provider == "litellm" else "",
    )

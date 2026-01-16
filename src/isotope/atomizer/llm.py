# src/isotope/atomizer/llm.py
"""LLM-based atomizer implementation."""

from __future__ import annotations

import json
import re
from typing import Literal

from isotope.atomizer.base import Atomizer
from isotope.models import Atom, Chunk
from isotope.providers.base import LLMClient

# Granularity preset prompts
ATOMIZER_PROMPTS: dict[str, str] = {
    "fine": """Please breakdown the following paragraph into stand-alone atomic facts.
Each fact should be a single, self-contained statement that can be understood without context.

Return your response as a JSON array of strings, where each string is one atomic fact.

Example input:
"Python was created by Guido van Rossum in 1991."

Example output:
["Guido van Rossum created Python.", "Python was created in 1991."]

Paragraph to atomize:
{content}

Return ONLY the JSON array, no other text.""",
    "medium": """Extract the KEY facts from the following paragraph.
Combine closely related details into single statements rather than splitting them.
Each fact should be self-contained and understandable without context.

Return your response as a JSON array of strings, where each string is one atomic fact.

Paragraph to atomize:
{content}

Return ONLY the JSON array, no other text.""",
    "coarse": """Extract only the 3-5 MOST ESSENTIAL facts from the following paragraph.
Focus on the core meaning and main points. Prioritize breadth over granularity.
Each fact should be self-contained and understandable without context.

Return your response as a JSON array of strings, where each string is one atomic fact.

Paragraph to atomize:
{content}

Return ONLY the JSON array, no other text.""",
}

# Default prompt (fine granularity) for backward compatibility
DEFAULT_PROMPT = ATOMIZER_PROMPTS["fine"]


class LLMAtomizer(Atomizer):
    """LLM-based atomizer for extracting atomic facts.

    This is the "unstructured" atomization approach from the paper.
    Uses any LLMClient to extract semantic atomic statements from the chunk.

    Example:
        from isotope.providers.litellm import LiteLLMClient
        from isotope.atomizer import LLMAtomizer

        client = LiteLLMClient(model="openai/gpt-5-mini-2025-08-07")
        atomizer = LLMAtomizer(llm_client=client, granularity="coarse")
    """

    def __init__(
        self,
        llm_client: LLMClient,
        prompt_template: str | None = None,
        granularity: Literal["coarse", "medium", "fine"] = "fine",
        temperature: float | None = 0.0,
    ) -> None:
        """Initialize the LLM atomizer.

        Args:
            llm_client: Any LLMClient implementation
            prompt_template: Custom prompt template with {content} placeholder.
                If provided, overrides granularity setting.
            granularity: Extraction granularity preset ("coarse", "medium", "fine").
                Ignored if prompt_template is provided.
            temperature: LLM temperature (0.0-1.0). None to use model default.
        """
        self._client = llm_client
        # Priority: explicit prompt > granularity preset
        if prompt_template:
            self.prompt_template = prompt_template
        else:
            self.prompt_template = ATOMIZER_PROMPTS[granularity]
        self.temperature = temperature

    def atomize(self, chunk: Chunk) -> list[Atom]:
        """Extract atomic facts from a chunk using an LLM."""
        content = chunk.content.strip()
        if not content:
            return []

        prompt = self.prompt_template.format(content=content)
        response_text = self._client.complete(
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        response_text = response_text.strip()

        # Parse JSON response
        try:
            # Handle potential markdown code blocks (even with introductory text)
            json_match = re.search(r"```(?:json)?\n(.*?)\n```", response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)

            facts = json.loads(response_text)
        except json.JSONDecodeError:
            # Fallback: treat each line as a fact, stripping list markers
            lines = [line.strip() for line in response_text.split("\n") if line.strip()]
            facts = [re.sub(r"^\s*(?:[-*]|\d+[.)])\s+", "", line) for line in lines]

        atoms = []
        for index, fact in enumerate(facts):
            if isinstance(fact, str) and fact.strip():
                atoms.append(
                    Atom(
                        content=fact.strip(),
                        chunk_id=chunk.id,
                        index=index,
                    )
                )

        return atoms

# src/isotopedb/question_generator/client.py
"""Client-based question generator implementation."""

import json
import re

from isotopedb.models import Atom, Question
from isotopedb.providers.base import LLMClient
from isotopedb.question_generator.base import QuestionGenerator

DEFAULT_PROMPT = """Generate {num_questions} diverse questions that this atomic fact answers.
The questions should be natural queries a user might ask when searching for this information.

Atomic fact: {atom_content}

Context (from the source document): {chunk_content}

Requirements:
1. Questions should be diverse in phrasing and perspective
2. Each question should be answerable by the atomic fact
3. Include both direct questions and indirect queries
4. Vary question types (what, how, why, when, who, etc.)

Return your response as a JSON array of question strings.

Example output:
["What is X?", "How does X work?", "Why is X important?"]

Return ONLY the JSON array, no other text."""


class ClientQuestionGenerator(QuestionGenerator):
    """Question generator that uses an LLMClient.

    Example:
        from isotopedb.providers.litellm import LiteLLMClient
        from isotopedb.question_generator import ClientQuestionGenerator

        client = LiteLLMClient(model="openai/gpt-4o")
        generator = ClientQuestionGenerator(llm_client=client)
    """

    def __init__(
        self,
        llm_client: LLMClient,
        num_questions: int = 15,
        prompt_template: str | None = None,
        temperature: float | None = 0.7,
    ) -> None:
        """Initialize the question generator.

        Args:
            llm_client: Any LLMClient implementation
            num_questions: Number of questions to generate per atom
            prompt_template: Custom prompt with {num_questions}, {atom_content}, {chunk_content}
            temperature: LLM temperature (0.0-1.0). None to use model default.
        """
        self._client = llm_client
        self.num_questions = num_questions
        self.prompt_template = prompt_template or DEFAULT_PROMPT
        self.temperature = temperature

    def _parse_response(self, response_text: str, atom: Atom) -> list[Question]:
        """Parse LLM response text into Question objects."""
        response_text = response_text.strip()

        try:
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                json_lines = []
                in_block = False
                for line in lines:
                    if line.startswith("```"):
                        in_block = not in_block
                        continue
                    if in_block:
                        json_lines.append(line)
                response_text = "\n".join(json_lines)

            question_texts = json.loads(response_text)
        except json.JSONDecodeError:
            question_texts = [
                re.sub(r"^\s*(?:[-*]|\d+[.)])\s+", "", line.strip())
                for line in response_text.split("\n")
                if line.strip()
            ]

        questions = []
        for text in question_texts:
            if isinstance(text, str) and text.strip():
                text = text.strip()
                if not text.endswith("?"):
                    text += "?"
                questions.append(
                    Question(
                        text=text,
                        chunk_id=atom.chunk_id,
                        atom_id=atom.id,
                    )
                )

        return questions

    def _build_prompt(self, atom: Atom, chunk_content: str) -> str:
        """Build the prompt for question generation."""
        return self.prompt_template.format(
            num_questions=self.num_questions,
            atom_content=atom.content,
            chunk_content=chunk_content or "(no additional context)",
        )

    def generate(self, atom: Atom, chunk_content: str = "") -> list[Question]:
        """Generate questions for a single atom."""
        prompt = self._build_prompt(atom, chunk_content)
        response_text = self._client.complete(
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        return self._parse_response(response_text, atom)

    def generate_batch(self, atoms: list[Atom], chunk_content: str = "") -> list[Question]:
        """Generate questions for multiple atoms."""
        all_questions = []
        for atom in atoms:
            questions = self.generate(atom, chunk_content)
            all_questions.extend(questions)
        return all_questions

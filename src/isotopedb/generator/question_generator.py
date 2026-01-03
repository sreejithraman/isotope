# src/isotopedb/generator/question_generator.py
"""LiteLLM-based question generator implementation."""

import json
import re

import litellm

from isotopedb.generator.base import QuestionGenerator
from isotopedb.models import Atom, Question

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


class LiteLLMQuestionGenerator(QuestionGenerator):
    """LiteLLM-based question generator.

    Generates synthetic questions from atoms using an LLM via LiteLLM.
    This is the core of the reverse-RAG approach: instead of embedding
    the content, we generate questions that the content answers, then
    embed those questions.
    """

    def __init__(
        self,
        model: str = "gemini/gemini-2.0-flash-exp",
        num_questions: int = 15,
        prompt_template: str | None = None,
        temperature: float | None = 0.7,
    ) -> None:
        """Initialize the question generator.

        Args:
            model: LiteLLM model identifier
            num_questions: Number of questions to generate per atom
            prompt_template: Custom prompt with {num_questions}, {atom_content}, {chunk_content}
            temperature: LLM temperature (0.0-1.0). None to use model default.
        """
        self.model = model
        self.num_questions = num_questions
        self.prompt_template = prompt_template or DEFAULT_PROMPT
        self.temperature = temperature

    def generate(
        self,
        atom: Atom,
        chunk_content: str = "",
    ) -> list[Question]:
        """Generate questions for a single atom.

        Args:
            atom: The atom to generate questions for
            chunk_content: Optional context from the parent chunk

        Returns:
            List of Question objects
        """
        prompt = self.prompt_template.format(
            num_questions=self.num_questions,
            atom_content=atom.content,
            chunk_content=chunk_content or "(no additional context)",
        )

        completion_kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "drop_params": True,
        }
        if self.temperature is not None:
            completion_kwargs["temperature"] = self.temperature

        response = litellm.completion(**completion_kwargs)

        response_text = response.choices[0].message.content.strip()

        # Parse JSON response
        try:
            # Handle potential markdown code blocks
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
            # Fallback: treat each line as a question, stripping list markers
            question_texts = [
                re.sub(r"^\s*[\d.\-*]+\s*", "", line.strip())
                for line in response_text.split("\n")
                if line.strip() and "?" in line
            ]

        questions = []
        for text in question_texts:
            if isinstance(text, str) and text.strip():
                # Clean up the question text
                text = text.strip()
                # Ensure it ends with a question mark
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

    def generate_batch(
        self,
        atoms: list[Atom],
        chunk_content: str = "",
    ) -> list[Question]:
        """Generate questions for multiple atoms.

        Args:
            atoms: List of atoms to generate questions for
            chunk_content: Optional context from the parent chunk

        Returns:
            Flat list of all generated Question objects
        """
        all_questions = []
        for atom in atoms:
            questions = self.generate(atom, chunk_content)
            all_questions.extend(questions)
        return all_questions

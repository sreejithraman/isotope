# src/isotope/question_generator/client.py
"""Client-based question generator implementation with multi-atom batching."""

import asyncio
import json
import re

from isotope.models import Atom, Question
from isotope.providers.base import LLMClient
from isotope.question_generator.base import (
    AsyncOnlyGeneratorMixin,
    BatchConfig,
    QuestionGenerator,
)
from isotope.question_generator.exceptions import BatchGenerationError

# Single-atom prompt template (used when batch_size=1)
SINGLE_ATOM_PROMPT = """Generate {num_questions} diverse questions that this atomic fact answers.
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

# Multi-atom prompt template (used when batch_size>1)
MULTI_ATOM_PROMPT = """Generate {num_questions} diverse questions for EACH atomic fact below.
The questions should be natural queries a user might ask when searching for this information.

{atoms_section}

Requirements for EACH atom:
1. Questions should be diverse in phrasing and perspective
2. Each question should be answerable by that specific atomic fact
3. Include both direct questions and indirect queries
4. Vary question types (what, how, why, when, who, etc.)

Return your response as a JSON object where keys are atom numbers (as strings)
and values are arrays of question strings.

Example format:
{{"0": ["What is X?", "How does X work?"], "1": ["Who invented Y?", "When was Y created?"]}}

Return ONLY the JSON object, no other text."""


class ClientQuestionGenerator(AsyncOnlyGeneratorMixin, QuestionGenerator):
    """Question generator that uses an LLMClient with multi-atom batching.

    This implementation batches multiple atoms into a single LLM prompt to reduce
    API calls. The batch_size is configured via BatchConfig.

    Example:
        from isotope.providers.litellm import LiteLLMClient
        from isotope.question_generator import ClientQuestionGenerator
        from isotope.question_generator.base import BatchConfig

        client = LiteLLMClient(model="openai/gpt-4o-mini")
        generator = ClientQuestionGenerator(llm_client=client)

        # For local models, use larger batch_size
        config = BatchConfig(batch_size=5, max_concurrent=2)
        questions = await generator.agenerate_batch(atoms, chunk_contents, config)
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
            prompt_template: Custom single-atom prompt with {num_questions},
                {atom_content}, {chunk_content}
            temperature: LLM temperature (0.0-1.0). None to use model default.
        """
        self._client = llm_client
        self.num_questions = num_questions
        self.single_atom_prompt = prompt_template or SINGLE_ATOM_PROMPT
        self.temperature = temperature

    def _build_single_atom_prompt(self, atom: Atom, chunk_content: str) -> str:
        """Build prompt for a single atom."""
        return self.single_atom_prompt.format(
            num_questions=self.num_questions,
            atom_content=atom.content,
            chunk_content=chunk_content or "(no additional context)",
        )

    def _build_multi_atom_prompt(self, atoms: list[Atom], chunk_contents: list[str]) -> str:
        """Build prompt for multiple atoms."""
        atoms_section = []
        for i, (atom, chunk_content) in enumerate(zip(atoms, chunk_contents, strict=True)):
            atoms_section.append(
                f"---\nATOM [{i}]:\n"
                f"Fact: {atom.content}\n"
                f"Context: {chunk_content or '(no additional context)'}"
            )
        return MULTI_ATOM_PROMPT.format(
            num_questions=self.num_questions,
            atoms_section="\n\n".join(atoms_section),
        )

    def _parse_single_atom_response(self, response_text: str, atom: Atom) -> list[Question]:
        """Parse LLM response for a single atom into Question objects."""
        response_text = response_text.strip()

        try:
            # Handle potential markdown code blocks
            json_match = re.search(r"```(?:json)?\n(.*?)\n```", response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)

            question_texts = json.loads(response_text)
        except json.JSONDecodeError:
            # Fallback: line-by-line parsing
            question_texts = [
                re.sub(r"^\s*(?:[-*]|\d+[.)])\s+", "", line.strip())
                for line in response_text.split("\n")
                if line.strip()
            ]

        return self._create_questions(question_texts, atom)

    def _parse_multi_atom_response(self, response_text: str, atoms: list[Atom]) -> list[Question]:
        """Parse LLM response for multiple atoms into Question objects.

        Tries multiple parsing strategies:
        1. JSON object {"0": [...], "1": [...]}
        2. JSON array [[...], [...]]
        3. Line-by-line with ATOM delimiters
        """
        response_text = response_text.strip()

        # Extract from markdown code block if present
        json_match = re.search(r"```(?:json)?\n(.*?)\n```", response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(1).strip()

        all_questions: list[Question] = []

        # Strategy 1: JSON object with string keys {"0": [...], "1": [...]}
        try:
            parsed = json.loads(response_text)
            if isinstance(parsed, dict):
                for i, atom in enumerate(atoms):
                    key = str(i)
                    if key in parsed and isinstance(parsed[key], list):
                        all_questions.extend(self._create_questions(parsed[key], atom))
                return all_questions
        except json.JSONDecodeError:
            pass

        # Strategy 2: JSON array [[...], [...]]
        try:
            parsed = json.loads(response_text)
            if isinstance(parsed, list) and len(parsed) == len(atoms):
                for atom, question_texts in zip(atoms, parsed, strict=True):
                    if isinstance(question_texts, list):
                        all_questions.extend(self._create_questions(question_texts, atom))
                return all_questions
        except json.JSONDecodeError:
            pass

        # Strategy 3: Line-by-line with ATOM delimiters
        current_atom_idx = 0
        current_questions: list[str] = []

        for line in response_text.split("\n"):
            line = line.strip()
            # Check for atom delimiter
            atom_match = re.match(r"ATOM\s*\[?(\d+)\]?:?", line, re.IGNORECASE)
            if atom_match:
                # Save questions for previous atom
                if current_questions and current_atom_idx < len(atoms):
                    all_questions.extend(
                        self._create_questions(current_questions, atoms[current_atom_idx])
                    )
                current_atom_idx = int(atom_match.group(1))
                current_questions = []
            elif line and not line.startswith("---"):
                # Remove list markers
                cleaned = re.sub(r"^\s*(?:[-*]|\d+[.)])\s+", "", line)
                if cleaned:
                    current_questions.append(cleaned)

        # Don't forget the last atom
        if current_questions and current_atom_idx < len(atoms):
            all_questions.extend(self._create_questions(current_questions, atoms[current_atom_idx]))

        return all_questions

    def _create_questions(self, question_texts: list, atom: Atom) -> list[Question]:
        """Create Question objects from text list."""
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

    async def _generate_single_batch(
        self, atoms: list[Atom], chunk_contents: list[str]
    ) -> list[Question]:
        """Generate questions for a batch of atoms in a single LLM call."""
        if len(atoms) == 1:
            # Single atom - use single-atom prompt
            prompt = self._build_single_atom_prompt(atoms[0], chunk_contents[0])
            response_text = await self._client.acomplete(
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
            )
            return self._parse_single_atom_response(response_text, atoms[0])
        else:
            # Multiple atoms - use multi-atom prompt
            prompt = self._build_multi_atom_prompt(atoms, chunk_contents)
            response_text = await self._client.acomplete(
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
            )
            return self._parse_multi_atom_response(response_text, atoms)

    async def agenerate_batch(
        self,
        atoms: list[Atom],
        chunk_contents: list[str] | None = None,
        config: BatchConfig | None = None,
    ) -> list[Question]:
        """Generate questions for multiple atoms with configurable batching.

        Atoms are grouped into batches of config.batch_size, and batches are
        processed concurrently up to config.max_concurrent.

        Args:
            atoms: List of atoms to generate questions for.
            chunk_contents: Optional chunk context for each atom.
            config: Batch configuration. Default: BatchConfig(batch_size=1, max_concurrent=10)

        Returns:
            List of all generated questions.

        Raises:
            BatchGenerationError: If more than 50% of batches fail.
        """
        if not atoms:
            return []

        config = config or BatchConfig()
        chunk_contents = chunk_contents or [""] * len(atoms)

        if len(chunk_contents) != len(atoms):
            raise ValueError(
                f"chunk_contents length ({len(chunk_contents)}) must match "
                f"atoms length ({len(atoms)})"
            )

        # Split atoms into batches
        batches: list[tuple[list[Atom], list[str]]] = []
        for i in range(0, len(atoms), config.batch_size):
            batch_atoms = atoms[i : i + config.batch_size]
            batch_contents = chunk_contents[i : i + config.batch_size]
            batches.append((batch_atoms, batch_contents))

        # Process batches concurrently with semaphore
        semaphore = asyncio.Semaphore(config.max_concurrent)

        async def process_batch(
            batch_atoms: list[Atom], batch_contents: list[str]
        ) -> list[Question]:
            async with semaphore:
                return await self._generate_single_batch(batch_atoms, batch_contents)

        results = await asyncio.gather(
            *[
                process_batch(batch_atoms, batch_contents)
                for batch_atoms, batch_contents in batches
            ],
            return_exceptions=True,
        )

        # Collect results and errors
        all_questions: list[Question] = []
        errors: list[tuple[int, BaseException]] = []

        for i, result in enumerate(results):
            if isinstance(result, BaseException):
                errors.append((i, result))
            else:
                all_questions.extend(result)

        # Raise if more than 50% of batches failed
        if errors and len(errors) / len(batches) > 0.5:
            raise BatchGenerationError(
                f"Too many failures: {len(errors)}/{len(batches)} batches failed",
                partial_results=all_questions,
                errors=errors,
            )

        return all_questions

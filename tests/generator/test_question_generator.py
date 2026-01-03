# tests/generator/test_question_generator.py
"""Tests for the LiteLLMGenerator (formerly LiteLLMQuestionGenerator)."""

import json
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("litellm", reason="Tests require litellm package")

from isotopedb.generator import QuestionGenerator
from isotopedb.litellm import ChatModels, LiteLLMGenerator
from isotopedb.models import Atom, Question


def mock_completion_response(questions: list[str]):
    """Create a mock LiteLLM completion response."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps(questions)
    return mock_response


@pytest.fixture
def generator():
    """Create a LiteLLMGenerator instance."""
    return LiteLLMGenerator()


@pytest.fixture
def sample_atom():
    """Create a sample atom."""
    return Atom(
        content="Python was created by Guido van Rossum.",
        chunk_id="chunk-123",
    )


class TestLiteLLMGenerator:
    def test_is_question_generator(self, generator):
        assert isinstance(generator, QuestionGenerator)

    def test_default_settings(self, generator):
        assert generator.model == ChatModels.GEMINI_3_FLASH
        assert generator.num_questions == 15
        assert generator.temperature == 0.7

    def test_custom_settings(self):
        gen = LiteLLMGenerator(
            model="openai/gpt-4",
            num_questions=5,
            temperature=0.5,
        )
        assert gen.model == "openai/gpt-4"
        assert gen.num_questions == 5
        assert gen.temperature == 0.5

    def test_temperature_none(self):
        gen = LiteLLMGenerator(temperature=None)
        assert gen.temperature is None

    @patch("isotopedb.litellm.generator.litellm.completion")
    def test_generate_questions(self, mock_completion, generator, sample_atom):
        mock_completion.return_value = mock_completion_response(
            [
                "Who created Python?",
                "What programming language did Guido create?",
            ]
        )

        questions = generator.generate(sample_atom)

        assert len(questions) == 2
        assert all(isinstance(q, Question) for q in questions)
        assert questions[0].text == "Who created Python?"
        assert questions[1].text == "What programming language did Guido create?"

    @patch("isotopedb.litellm.generator.litellm.completion")
    def test_questions_have_correct_chunk_id(self, mock_completion, generator, sample_atom):
        mock_completion.return_value = mock_completion_response(["Q1?", "Q2?"])

        questions = generator.generate(sample_atom)

        for q in questions:
            assert q.chunk_id == sample_atom.chunk_id

    @patch("isotopedb.litellm.generator.litellm.completion")
    def test_questions_have_correct_atom_id(self, mock_completion, generator, sample_atom):
        mock_completion.return_value = mock_completion_response(["Q1?", "Q2?"])

        questions = generator.generate(sample_atom)

        for q in questions:
            assert q.atom_id == sample_atom.id

    @patch("isotopedb.litellm.generator.litellm.completion")
    def test_questions_have_unique_ids(self, mock_completion, generator, sample_atom):
        mock_completion.return_value = mock_completion_response(["Q1?", "Q2?"])

        questions = generator.generate(sample_atom)

        assert questions[0].id != questions[1].id

    @patch("isotopedb.litellm.generator.litellm.completion")
    def test_handles_code_block_response(self, mock_completion, generator, sample_atom):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '```json\n["Q1?", "Q2?"]\n```'
        mock_completion.return_value = mock_response

        questions = generator.generate(sample_atom)

        assert len(questions) == 2

    @patch("isotopedb.litellm.generator.litellm.completion")
    def test_fallback_on_invalid_json(self, mock_completion, generator, sample_atom):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "1. Who created Python?\n2. What is Python?"
        mock_completion.return_value = mock_response

        questions = generator.generate(sample_atom)

        assert len(questions) == 2
        assert "Who created Python?" in questions[0].text

    @patch("isotopedb.litellm.generator.litellm.completion")
    def test_adds_question_mark_if_missing(self, mock_completion, generator, sample_atom):
        mock_completion.return_value = mock_completion_response(
            [
                "Who created Python",  # Missing ?
            ]
        )

        questions = generator.generate(sample_atom)

        assert questions[0].text.endswith("?")

    @patch("isotopedb.litellm.generator.litellm.completion")
    def test_filters_empty_questions(self, mock_completion, generator, sample_atom):
        mock_completion.return_value = mock_completion_response(
            [
                "Q1?",
                "",
                "  ",
                "Q2?",
            ]
        )

        questions = generator.generate(sample_atom)

        assert len(questions) == 2

    @patch("isotopedb.litellm.generator.litellm.completion")
    def test_generate_batch(self, mock_completion, generator):
        mock_completion.return_value = mock_completion_response(["Q1?", "Q2?"])

        atoms = [
            Atom(content="Fact 1", chunk_id="c1"),
            Atom(content="Fact 2", chunk_id="c1"),
        ]

        questions = generator.generate_batch(atoms)

        # Should call LLM once per atom
        assert mock_completion.call_count == 2
        # Should return all questions
        assert len(questions) == 4

    @patch("isotopedb.litellm.generator.litellm.completion")
    def test_uses_chunk_content_context(self, mock_completion, generator, sample_atom):
        mock_completion.return_value = mock_completion_response(["Q1?"])

        generator.generate(sample_atom, chunk_content="Full document context here")

        # Verify the context was passed in the prompt
        call_args = mock_completion.call_args
        prompt = call_args.kwargs["messages"][0]["content"]
        assert "Full document context here" in prompt

    def test_custom_prompt_template(self, sample_atom):
        custom_prompt = "Generate questions for: {atom_content}"
        gen = LiteLLMGenerator(prompt_template=custom_prompt)
        assert gen.prompt_template == custom_prompt

    @patch("isotopedb.litellm.generator.litellm.completion")
    def test_temperature_passed_to_llm(self, mock_completion, sample_atom):
        mock_completion.return_value = mock_completion_response(["Q1?"])
        gen = LiteLLMGenerator(temperature=0.9)

        gen.generate(sample_atom)

        call_kwargs = mock_completion.call_args.kwargs
        assert call_kwargs["temperature"] == 0.9
        assert call_kwargs["drop_params"] is True

    @patch("isotopedb.litellm.generator.litellm.completion")
    def test_temperature_none_not_passed(self, mock_completion, sample_atom):
        mock_completion.return_value = mock_completion_response(["Q1?"])
        gen = LiteLLMGenerator(temperature=None)

        gen.generate(sample_atom)

        call_kwargs = mock_completion.call_args.kwargs
        assert "temperature" not in call_kwargs
        assert call_kwargs["drop_params"] is True

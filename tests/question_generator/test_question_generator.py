# tests/question_generator/test_question_generator.py
"""Tests for the ClientQuestionGenerator."""

import json
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("litellm", reason="Tests require litellm package")

from isotope.models import Atom, Question
from isotope.providers.litellm import LiteLLMClient
from isotope.question_generator import ClientQuestionGenerator, QuestionGenerator


def mock_completion_response(questions: list[str]):
    """Create a mock LiteLLM completion response."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps(questions)
    return mock_response


@pytest.fixture
def llm_client():
    """Create a LiteLLMClient instance."""
    return LiteLLMClient()


@pytest.fixture
def generator(llm_client):
    """Create an ClientQuestionGenerator instance."""
    return ClientQuestionGenerator(llm_client=llm_client)


@pytest.fixture
def sample_atom():
    """Create a sample atom."""
    return Atom(
        content="Python was created by Guido van Rossum.",
        chunk_id="chunk-123",
    )


class TestClientQuestionGenerator:
    def test_is_question_generator(self, generator):
        assert isinstance(generator, QuestionGenerator)

    def test_default_settings(self, generator):
        assert generator.num_questions == 15
        assert generator.temperature == 0.7

    def test_custom_settings(self, llm_client):
        gen = ClientQuestionGenerator(
            llm_client=llm_client,
            num_questions=5,
            temperature=0.5,
        )
        assert gen.num_questions == 5
        assert gen.temperature == 0.5

    def test_temperature_none(self, llm_client):
        gen = ClientQuestionGenerator(llm_client=llm_client, temperature=None)
        assert gen.temperature is None

    @patch("isotope.providers.litellm.client.litellm.completion")
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

    @patch("isotope.providers.litellm.client.litellm.completion")
    def test_questions_have_correct_chunk_id(self, mock_completion, generator, sample_atom):
        mock_completion.return_value = mock_completion_response(["Q1?", "Q2?"])

        questions = generator.generate(sample_atom)

        for q in questions:
            assert q.chunk_id == sample_atom.chunk_id

    @patch("isotope.providers.litellm.client.litellm.completion")
    def test_questions_have_correct_atom_id(self, mock_completion, generator, sample_atom):
        mock_completion.return_value = mock_completion_response(["Q1?", "Q2?"])

        questions = generator.generate(sample_atom)

        for q in questions:
            assert q.atom_id == sample_atom.id

    @patch("isotope.providers.litellm.client.litellm.completion")
    def test_questions_have_unique_ids(self, mock_completion, generator, sample_atom):
        mock_completion.return_value = mock_completion_response(["Q1?", "Q2?"])

        questions = generator.generate(sample_atom)

        assert questions[0].id != questions[1].id

    @patch("isotope.providers.litellm.client.litellm.completion")
    def test_handles_code_block_response(self, mock_completion, generator, sample_atom):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '```json\n["Q1?", "Q2?"]\n```'
        mock_completion.return_value = mock_response

        questions = generator.generate(sample_atom)

        assert len(questions) == 2

    @patch("isotope.providers.litellm.client.litellm.completion")
    def test_fallback_on_invalid_json(self, mock_completion, generator, sample_atom):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "1. Who created Python?\n2. What is Python?"
        mock_completion.return_value = mock_response

        questions = generator.generate(sample_atom)

        assert len(questions) == 2
        assert "Who created Python?" in questions[0].text

    @patch("isotope.providers.litellm.client.litellm.completion")
    def test_adds_question_mark_if_missing(self, mock_completion, generator, sample_atom):
        mock_completion.return_value = mock_completion_response(
            [
                "Who created Python",  # Missing ?
            ]
        )

        questions = generator.generate(sample_atom)

        assert questions[0].text.endswith("?")

    @patch("isotope.providers.litellm.client.litellm.completion")
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

    @patch("isotope.providers.litellm.client.litellm.completion")
    def test_uses_chunk_content_context(self, mock_completion, generator, sample_atom):
        mock_completion.return_value = mock_completion_response(["Q1?"])

        generator.generate(sample_atom, chunk_content="Full document context here")

        # Verify the context was passed in the prompt
        call_args = mock_completion.call_args
        prompt = call_args.kwargs["messages"][0]["content"]
        assert "Full document context here" in prompt

    def test_custom_prompt_template(self, llm_client, sample_atom):
        custom_prompt = "Generate questions for: {atom_content}"
        gen = ClientQuestionGenerator(llm_client=llm_client, prompt_template=custom_prompt)
        assert gen.prompt_template == custom_prompt

    @patch("isotope.providers.litellm.client.litellm.completion")
    def test_temperature_passed_to_llm(self, mock_completion, llm_client, sample_atom):
        mock_completion.return_value = mock_completion_response(["Q1?"])
        gen = ClientQuestionGenerator(llm_client=llm_client, temperature=0.9)

        gen.generate(sample_atom)

        call_kwargs = mock_completion.call_args.kwargs
        assert call_kwargs["temperature"] == 0.9
        assert call_kwargs["drop_params"] is True

    @patch("isotope.providers.litellm.client.litellm.completion")
    def test_temperature_none_not_passed(self, mock_completion, llm_client, sample_atom):
        mock_completion.return_value = mock_completion_response(["Q1?"])
        gen = ClientQuestionGenerator(llm_client=llm_client, temperature=None)

        gen.generate(sample_atom)

        call_kwargs = mock_completion.call_args.kwargs
        assert "temperature" not in call_kwargs
        assert call_kwargs["drop_params"] is True

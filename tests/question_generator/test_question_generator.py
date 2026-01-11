# tests/question_generator/test_question_generator.py
"""Tests for the ClientQuestionGenerator."""

import json
from unittest.mock import MagicMock, patch

import pytest

from isotope.models import Atom, Question
from isotope.providers.litellm import LiteLLMClient
from isotope.question_generator import (
    BatchConfig,
    ClientQuestionGenerator,
    QuestionGenerator,
)


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

    @patch("isotope.providers.litellm.client.litellm.acompletion")
    def test_generate_questions(self, mock_acompletion, generator, sample_atom):
        mock_acompletion.return_value = mock_completion_response(
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

    @patch("isotope.providers.litellm.client.litellm.acompletion")
    def test_questions_have_correct_chunk_id(self, mock_acompletion, generator, sample_atom):
        mock_acompletion.return_value = mock_completion_response(["Q1?", "Q2?"])

        questions = generator.generate(sample_atom)

        for q in questions:
            assert q.chunk_id == sample_atom.chunk_id

    @patch("isotope.providers.litellm.client.litellm.acompletion")
    def test_questions_have_correct_atom_id(self, mock_acompletion, generator, sample_atom):
        mock_acompletion.return_value = mock_completion_response(["Q1?", "Q2?"])

        questions = generator.generate(sample_atom)

        for q in questions:
            assert q.atom_id == sample_atom.id

    @patch("isotope.providers.litellm.client.litellm.acompletion")
    def test_questions_have_unique_ids(self, mock_acompletion, generator, sample_atom):
        mock_acompletion.return_value = mock_completion_response(["Q1?", "Q2?"])

        questions = generator.generate(sample_atom)

        assert questions[0].id != questions[1].id

    @patch("isotope.providers.litellm.client.litellm.acompletion")
    def test_handles_code_block_response(self, mock_acompletion, generator, sample_atom):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '```json\n["Q1?", "Q2?"]\n```'
        mock_acompletion.return_value = mock_response

        questions = generator.generate(sample_atom)

        assert len(questions) == 2

    @patch("isotope.providers.litellm.client.litellm.acompletion")
    def test_fallback_on_invalid_json(self, mock_acompletion, generator, sample_atom):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "1. Who created Python?\n2. What is Python?"
        mock_acompletion.return_value = mock_response

        questions = generator.generate(sample_atom)

        assert len(questions) == 2
        assert "Who created Python?" in questions[0].text

    @patch("isotope.providers.litellm.client.litellm.acompletion")
    def test_adds_question_mark_if_missing(self, mock_acompletion, generator, sample_atom):
        mock_acompletion.return_value = mock_completion_response(
            [
                "Who created Python",  # Missing ?
            ]
        )

        questions = generator.generate(sample_atom)

        assert questions[0].text.endswith("?")

    @patch("isotope.providers.litellm.client.litellm.acompletion")
    def test_filters_empty_questions(self, mock_acompletion, generator, sample_atom):
        mock_acompletion.return_value = mock_completion_response(
            [
                "Q1?",
                "",
                "  ",
                "Q2?",
            ]
        )

        questions = generator.generate(sample_atom)

        assert len(questions) == 2

    @patch("isotope.providers.litellm.client.litellm.acompletion")
    def test_uses_chunk_content_context(self, mock_acompletion, generator, sample_atom):
        mock_acompletion.return_value = mock_completion_response(["Q1?"])

        generator.generate(sample_atom, chunk_content="Full document context here")

        # Verify the context was passed in the prompt
        call_args = mock_acompletion.call_args
        prompt = call_args.kwargs["messages"][0]["content"]
        assert "Full document context here" in prompt

    def test_custom_prompt_template(self, llm_client, sample_atom):
        custom_prompt = "Generate questions for: {atom_content}"
        gen = ClientQuestionGenerator(llm_client=llm_client, prompt_template=custom_prompt)
        assert gen.single_atom_prompt == custom_prompt

    @patch("isotope.providers.litellm.client.litellm.acompletion")
    def test_temperature_passed_to_llm(self, mock_acompletion, llm_client, sample_atom):
        mock_acompletion.return_value = mock_completion_response(["Q1?"])
        gen = ClientQuestionGenerator(llm_client=llm_client, temperature=0.9)

        gen.generate(sample_atom)

        call_kwargs = mock_acompletion.call_args.kwargs
        assert call_kwargs["temperature"] == 0.9
        assert call_kwargs["drop_params"] is True

    @patch("isotope.providers.litellm.client.litellm.acompletion")
    def test_temperature_none_not_passed(self, mock_acompletion, llm_client, sample_atom):
        mock_acompletion.return_value = mock_completion_response(["Q1?"])
        gen = ClientQuestionGenerator(llm_client=llm_client, temperature=None)

        gen.generate(sample_atom)

        call_kwargs = mock_acompletion.call_args.kwargs
        assert "temperature" not in call_kwargs
        assert call_kwargs["drop_params"] is True


class TestMultiAtomBatching:
    """Tests for multi-atom batching behavior."""

    @pytest.fixture
    def multi_atom_response(self):
        """Create a mock multi-atom response."""
        return json.dumps(
            {
                "0": ["Who created Python?", "When was Python created?"],
                "1": ["What is indentation?", "Why use indentation?"],
            }
        )

    @patch("isotope.providers.litellm.client.litellm.acompletion")
    def test_batch_size_1_uses_single_atom_prompt(self, mock_acompletion, generator, sample_atom):
        """With batch_size=1, should use single-atom prompt format."""
        mock_acompletion.return_value = mock_completion_response(["Q1?", "Q2?"])

        config = BatchConfig(batch_size=1, max_concurrent=10)
        generator.generate_batch([sample_atom], config=config)

        # Single-atom prompt should NOT contain "ATOM [0]:"
        call_args = mock_acompletion.call_args
        prompt = call_args.kwargs["messages"][0]["content"]
        assert "ATOM [0]:" not in prompt

    @patch("isotope.providers.litellm.client.litellm.acompletion")
    def test_batch_size_greater_than_1_uses_multi_atom_prompt(
        self, mock_acompletion, generator, multi_atom_response
    ):
        """With batch_size>1, should use multi-atom prompt format."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = multi_atom_response
        mock_acompletion.return_value = mock_response

        atoms = [
            Atom(content="Python was created by Guido.", chunk_id="c1"),
            Atom(content="Python uses indentation.", chunk_id="c1"),
        ]
        config = BatchConfig(batch_size=5, max_concurrent=2)
        questions = generator.generate_batch(atoms, config=config)

        # Multi-atom prompt should contain "ATOM [0]:" and "ATOM [1]:"
        call_args = mock_acompletion.call_args
        prompt = call_args.kwargs["messages"][0]["content"]
        assert "ATOM [0]:" in prompt
        assert "ATOM [1]:" in prompt

        # Should have questions for both atoms
        assert len(questions) == 4
        atom_ids = {q.atom_id for q in questions}
        assert atoms[0].id in atom_ids
        assert atoms[1].id in atom_ids

    @patch("isotope.providers.litellm.client.litellm.acompletion")
    def test_multi_atom_parses_json_object_response(
        self, mock_acompletion, generator, multi_atom_response
    ):
        """Should parse JSON object response {"0": [...], "1": [...]}."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = multi_atom_response
        mock_acompletion.return_value = mock_response

        atoms = [
            Atom(content="Fact 1", chunk_id="c1"),
            Atom(content="Fact 2", chunk_id="c1"),
        ]
        config = BatchConfig(batch_size=5, max_concurrent=2)
        questions = generator.generate_batch(atoms, config=config)

        assert len(questions) == 4
        # First two questions should be for atom 0
        assert questions[0].atom_id == atoms[0].id
        assert questions[1].atom_id == atoms[0].id
        # Last two for atom 1
        assert questions[2].atom_id == atoms[1].id
        assert questions[3].atom_id == atoms[1].id

    @patch("isotope.providers.litellm.client.litellm.acompletion")
    def test_multi_atom_parses_json_array_response(self, mock_acompletion, generator):
        """Should fall back to JSON array [[...], [...]] format."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            [
                ["Q1a?", "Q1b?"],
                ["Q2a?", "Q2b?"],
            ]
        )
        mock_acompletion.return_value = mock_response

        atoms = [
            Atom(content="Fact 1", chunk_id="c1"),
            Atom(content="Fact 2", chunk_id="c1"),
        ]
        config = BatchConfig(batch_size=5, max_concurrent=2)
        questions = generator.generate_batch(atoms, config=config)

        assert len(questions) == 4

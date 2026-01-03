# tests/atomizer/test_llm.py
"""Tests for LiteLLM-based atomizer."""

import json
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("litellm", reason="Tests require litellm package")

from isotopedb.atomizer.base import Atomizer
from isotopedb.litellm import LiteLLMAtomizer
from isotopedb.models import Chunk


def mock_completion_response(facts: list[str]):
    """Create a mock LiteLLM completion response."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps(facts)
    return mock_response


@pytest.fixture
def atomizer():
    """Create a LiteLLMAtomizer instance."""
    return LiteLLMAtomizer()


class TestLiteLLMAtomizer:
    def test_is_atomizer(self, atomizer):
        assert isinstance(atomizer, Atomizer)

    @patch("isotopedb.litellm.atomizer.litellm.completion")
    def test_extracts_atomic_facts(self, mock_completion, atomizer):
        mock_completion.return_value = mock_completion_response(
            [
                "Python was created by Guido.",
                "Python is interpreted.",
            ]
        )

        chunk = Chunk(
            content="Python was created by Guido. It's an interpreted language.",
            source="test.md",
        )
        atoms = atomizer.atomize(chunk)

        assert len(atoms) == 2
        assert atoms[0].content == "Python was created by Guido."
        assert atoms[1].content == "Python is interpreted."
        mock_completion.assert_called_once()

    @patch("isotopedb.litellm.atomizer.litellm.completion")
    def test_assigns_correct_chunk_id(self, mock_completion, atomizer):
        mock_completion.return_value = mock_completion_response(["Fact 1", "Fact 2"])

        chunk = Chunk(content="Some content", source="test.md")
        atoms = atomizer.atomize(chunk)

        for atom in atoms:
            assert atom.chunk_id == chunk.id

    @patch("isotopedb.litellm.atomizer.litellm.completion")
    def test_assigns_sequential_index(self, mock_completion, atomizer):
        mock_completion.return_value = mock_completion_response(
            [
                "First fact",
                "Second fact",
                "Third fact",
            ]
        )

        chunk = Chunk(content="Content", source="test.md")
        atoms = atomizer.atomize(chunk)

        assert atoms[0].index == 0
        assert atoms[1].index == 1
        assert atoms[2].index == 2

    def test_handles_empty_content(self, atomizer):
        chunk = Chunk(content="", source="test.md")
        atoms = atomizer.atomize(chunk)
        assert atoms == []

    def test_handles_whitespace_only(self, atomizer):
        chunk = Chunk(content="   \n\t  ", source="test.md")
        atoms = atomizer.atomize(chunk)
        assert atoms == []

    @patch("isotopedb.litellm.atomizer.litellm.completion")
    def test_handles_code_block_response(self, mock_completion, atomizer):
        # Some LLMs wrap JSON in code blocks
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '```json\n["Fact 1", "Fact 2"]\n```'
        mock_completion.return_value = mock_response

        chunk = Chunk(content="Content", source="test.md")
        atoms = atomizer.atomize(chunk)

        assert len(atoms) == 2
        assert atoms[0].content == "Fact 1"

    @patch("isotopedb.litellm.atomizer.litellm.completion")
    def test_fallback_on_invalid_json(self, mock_completion, atomizer):
        # If LLM returns non-JSON, fall back to line splitting
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Fact 1\nFact 2\nFact 3"
        mock_completion.return_value = mock_response

        chunk = Chunk(content="Content", source="test.md")
        atoms = atomizer.atomize(chunk)

        assert len(atoms) == 3
        assert atoms[0].content == "Fact 1"

    @patch("isotopedb.litellm.atomizer.litellm.completion")
    def test_filters_empty_facts(self, mock_completion, atomizer):
        mock_completion.return_value = mock_completion_response(
            [
                "Valid fact",
                "",
                "  ",
                "Another valid fact",
            ]
        )

        chunk = Chunk(content="Content", source="test.md")
        atoms = atomizer.atomize(chunk)

        assert len(atoms) == 2
        assert atoms[0].content == "Valid fact"
        assert atoms[1].content == "Another valid fact"

    def test_custom_model(self):
        atomizer = LiteLLMAtomizer(model="openai/gpt-4")
        assert atomizer.model == "openai/gpt-4"

    def test_custom_prompt_template(self):
        custom_prompt = "Extract facts from: {content}"
        atomizer = LiteLLMAtomizer(prompt_template=custom_prompt)
        assert atomizer.prompt_template == custom_prompt

    def test_default_temperature(self, atomizer):
        assert atomizer.temperature == 0.0

    def test_custom_temperature(self):
        atomizer = LiteLLMAtomizer(temperature=0.3)
        assert atomizer.temperature == 0.3

    def test_temperature_none(self):
        atomizer = LiteLLMAtomizer(temperature=None)
        assert atomizer.temperature is None

    @patch("isotopedb.litellm.atomizer.litellm.completion")
    def test_temperature_passed_to_llm(self, mock_completion):
        mock_completion.return_value = mock_completion_response(["Fact 1"])
        atomizer = LiteLLMAtomizer(temperature=0.5)

        chunk = Chunk(content="Content", source="test.md")
        atomizer.atomize(chunk)

        call_kwargs = mock_completion.call_args.kwargs
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["drop_params"] is True

    @patch("isotopedb.litellm.atomizer.litellm.completion")
    def test_temperature_none_not_passed(self, mock_completion):
        mock_completion.return_value = mock_completion_response(["Fact 1"])
        atomizer = LiteLLMAtomizer(temperature=None)

        chunk = Chunk(content="Content", source="test.md")
        atomizer.atomize(chunk)

        call_kwargs = mock_completion.call_args.kwargs
        assert "temperature" not in call_kwargs
        assert call_kwargs["drop_params"] is True

    @patch("isotopedb.litellm.atomizer.litellm.completion")
    def test_each_atom_has_unique_id(self, mock_completion, atomizer):
        mock_completion.return_value = mock_completion_response(["Fact 1", "Fact 2"])

        chunk = Chunk(content="Content", source="test.md")
        atoms = atomizer.atomize(chunk)

        assert atoms[0].id != atoms[1].id

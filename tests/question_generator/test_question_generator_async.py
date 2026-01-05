# tests/question_generator/test_question_generator_async.py
"""Tests for async question generation."""

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("litellm", reason="Tests require litellm package")

from isotope.models import Atom, Question
from isotope.providers.litellm import LiteLLMClient
from isotope.question_generator import (
    BatchGenerationError,
    ClientQuestionGenerator,
    QuestionGenerator,
)


def mock_acompletion_response(questions: list[str]):
    """Create a mock LiteLLM async completion response."""
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
    """Create a ClientQuestionGenerator instance."""
    return ClientQuestionGenerator(llm_client=llm_client)


@pytest.fixture
def sample_atom():
    """Create a sample atom."""
    return Atom(
        content="Python was created by Guido van Rossum.",
        chunk_id="chunk-123",
    )


class TestAsyncLLMClient:
    """Tests for async LLM client methods."""

    @pytest.mark.asyncio
    @patch("isotope.providers.litellm.client.litellm.acompletion")
    async def test_acomplete(self, mock_acompletion, llm_client):
        """Test async completion using litellm.acompletion."""
        mock_acompletion.return_value = mock_acompletion_response(["Q1?"])

        result = await llm_client.acomplete([{"role": "user", "content": "Hello"}])

        assert result == '["Q1?"]'
        mock_acompletion.assert_called_once()

    @pytest.mark.asyncio
    @patch("isotope.providers.litellm.client.litellm.acompletion")
    async def test_acomplete_with_temperature(self, mock_acompletion, llm_client):
        """Test async completion passes temperature."""
        mock_acompletion.return_value = mock_acompletion_response(["Q1?"])

        await llm_client.acomplete([{"role": "user", "content": "Hello"}], temperature=0.5)

        call_kwargs = mock_acompletion.call_args.kwargs
        assert call_kwargs["temperature"] == 0.5


class TestAsyncQuestionGeneration:
    """Tests for async question generation."""

    @pytest.mark.asyncio
    @patch("isotope.providers.litellm.client.litellm.acompletion")
    async def test_agenerate_single_atom(self, mock_acompletion, generator, sample_atom):
        """Test async generation for a single atom."""
        mock_acompletion.return_value = mock_acompletion_response(
            ["Who created Python?", "What is Python?"]
        )

        questions = await generator.agenerate(sample_atom)

        assert len(questions) == 2
        assert all(isinstance(q, Question) for q in questions)
        assert questions[0].text == "Who created Python?"
        mock_acompletion.assert_called_once()

    @pytest.mark.asyncio
    @patch("isotope.providers.litellm.client.litellm.acompletion")
    async def test_agenerate_with_chunk_content(self, mock_acompletion, generator, sample_atom):
        """Test async generation with chunk context."""
        mock_acompletion.return_value = mock_acompletion_response(["Q1?"])

        await generator.agenerate(sample_atom, chunk_content="Full context here")

        call_args = mock_acompletion.call_args
        prompt = call_args.kwargs["messages"][0]["content"]
        assert "Full context here" in prompt


class TestAsyncBatchGeneration:
    """Tests for async batch question generation."""

    @pytest.mark.asyncio
    @patch("isotope.providers.litellm.client.litellm.acompletion")
    async def test_agenerate_batch_concurrent(self, mock_acompletion, generator):
        """Test concurrent batch generation."""
        mock_acompletion.return_value = mock_acompletion_response(["Q1?", "Q2?"])

        atoms = [Atom(content=f"Fact {i}", chunk_id=f"chunk-{i}", index=0) for i in range(5)]
        chunk_contents = [f"Content {i}" for i in range(5)]

        questions = await generator.agenerate_batch(atoms, chunk_contents, max_concurrent=3)

        # Should have 2 questions per atom = 10 total
        assert len(questions) == 10
        # Should call acompletion once per atom
        assert mock_acompletion.call_count == 5

    @pytest.mark.asyncio
    @patch("isotope.providers.litellm.client.litellm.acompletion")
    async def test_agenerate_batch_respects_concurrency_limit(self, mock_acompletion, generator):
        """Test that max_concurrent is respected."""
        active_count = 0
        max_active = 0

        async def mock_acomplete_impl(*args, **kwargs):
            nonlocal active_count, max_active
            active_count += 1
            max_active = max(max_active, active_count)
            await asyncio.sleep(0.01)  # Simulate network delay
            active_count -= 1
            return mock_acompletion_response(["Q?"])

        mock_acompletion.side_effect = mock_acomplete_impl

        atoms = [Atom(content=f"Fact {i}", chunk_id="chunk-1", index=i) for i in range(10)]

        await generator.agenerate_batch(atoms, max_concurrent=3)

        assert max_active <= 3

    @pytest.mark.asyncio
    @patch("isotope.providers.litellm.client.litellm.acompletion")
    async def test_agenerate_batch_partial_failure_continues(self, mock_acompletion, generator):
        """Test that partial failures don't stop the entire batch."""
        call_count = 0

        async def mock_acomplete_impl(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise Exception("Simulated failure")
            return mock_acompletion_response(["Q?"])

        mock_acompletion.side_effect = mock_acomplete_impl

        atoms = [Atom(content=f"Fact {i}", chunk_id="chunk-1", index=i) for i in range(5)]

        questions = await generator.agenerate_batch(atoms)

        # Should have 4 questions (1 failure out of 5)
        assert len(questions) == 4

    @pytest.mark.asyncio
    @patch("isotope.providers.litellm.client.litellm.acompletion")
    async def test_agenerate_batch_raises_on_majority_failure(self, mock_acompletion, generator):
        """Test that BatchGenerationError is raised when >50% fail."""
        call_count = 0

        async def mock_acomplete_impl(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 3:  # First 3 out of 5 fail = 60%
                raise Exception("Simulated failure")
            return mock_acompletion_response(["Q?"])

        mock_acompletion.side_effect = mock_acomplete_impl

        atoms = [Atom(content=f"Fact {i}", chunk_id="chunk-1", index=i) for i in range(5)]

        with pytest.raises(BatchGenerationError) as exc_info:
            await generator.agenerate_batch(atoms)

        assert "3/5 atoms failed" in str(exc_info.value)
        assert len(exc_info.value.partial_results) == 2  # 2 succeeded
        assert len(exc_info.value.errors) == 3  # 3 failed

    @pytest.mark.asyncio
    async def test_agenerate_batch_empty_atoms(self, generator):
        """Test that empty atoms list returns empty questions."""
        questions = await generator.agenerate_batch([])
        assert questions == []

    @pytest.mark.asyncio
    async def test_agenerate_batch_mismatched_lengths_raises(self, generator):
        """Test that mismatched atoms/chunk_contents raises ValueError."""
        atoms = [Atom(content="Fact", chunk_id="c1", index=0)]
        chunk_contents = ["a", "b"]  # Wrong length

        with pytest.raises(ValueError, match="must match atoms length"):
            await generator.agenerate_batch(atoms, chunk_contents)


class TestQuestionGeneratorBaseAsyncDefaults:
    """Tests for default async implementations in QuestionGenerator base class."""

    @pytest.mark.asyncio
    async def test_agenerate_default_calls_sync(self):
        """Test that default agenerate() calls sync generate()."""

        class TestGenerator(QuestionGenerator):
            def generate(self, atom, chunk_content=""):
                return [
                    Question(
                        text=f"Q about {atom.content}?",
                        chunk_id=atom.chunk_id,
                        atom_id=atom.id,
                    )
                ]

        gen = TestGenerator()
        atom = Atom(content="Test", chunk_id="c1", index=0)

        questions = await gen.agenerate(atom)

        assert len(questions) == 1
        assert "Test" in questions[0].text

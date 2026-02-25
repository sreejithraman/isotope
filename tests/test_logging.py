# tests/test_logging.py
"""Tests for ingestion logging."""

import logging
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
import pytest

from isotope.config import setup_file_logging, teardown_file_logging
from isotope.models import Atom, Chunk, Question


@pytest.fixture
def _empty_generator():
    """A question generator that returns no questions (simulates silent failure)."""
    from isotope.question_generator import QuestionGenerator
    from isotope.question_generator.base import BatchConfig, SyncOnlyGeneratorMixin

    class EmptyGenerator(SyncOnlyGeneratorMixin, QuestionGenerator):
        def generate_batch(
            self,
            atoms: list[Atom],
            chunk_contents: list[str] | None = None,
            config: BatchConfig | None = None,
        ) -> list[Question]:
            return []

    return EmptyGenerator()


# ---------------------------------------------------------------------------
# caplog tests: Ingestor logging
# ---------------------------------------------------------------------------


class TestIngestorLogging:
    """Verify log output from the Ingestor pipeline."""

    def test_zero_questions_warning(
        self, stores, mock_embedder, mock_atomizer, _empty_generator, caplog
    ):
        """When the generator returns [], a WARNING about 0 questions should appear."""
        from isotope.ingestor import Ingestor

        ingestor = Ingestor(
            embedded_question_store=stores["embedded_question_store"],
            chunk_store=stores["chunk_store"],
            atom_store=stores["atom_store"],
            atomizer=mock_atomizer,
            embedder=mock_embedder,
            question_generator=_empty_generator,
        )

        chunks = [Chunk(content="The sky is blue.", source="test.txt")]

        with caplog.at_level(logging.DEBUG, logger="isotope"):
            ingestor.ingest_chunks(chunks)

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("0 questions" in r.message for r in warnings), (
            f"Expected '0 questions' WARNING, got: {[r.message for r in warnings]}"
        )

    def test_stored_chunks_info(self, stores, mock_embedder, mock_atomizer, mock_generator, caplog):
        """INFO log for stored chunks count should appear."""
        from isotope.ingestor import Ingestor

        ingestor = Ingestor(
            embedded_question_store=stores["embedded_question_store"],
            chunk_store=stores["chunk_store"],
            atom_store=stores["atom_store"],
            atomizer=mock_atomizer,
            embedder=mock_embedder,
            question_generator=mock_generator,
        )

        chunks = [Chunk(content="The sky is blue.", source="test.txt")]

        with caplog.at_level(logging.DEBUG, logger="isotope"):
            ingestor.ingest_chunks(chunks)

        info_msgs = [r.message for r in caplog.records if r.levelno == logging.INFO]
        assert any("Stored 1 chunks" in m for m in info_msgs)
        assert any("Created 1 atoms from 1 chunks" in m for m in info_msgs)
        assert any("Indexed" in m and "(final)" in m for m in info_msgs)


# ---------------------------------------------------------------------------
# caplog tests: QuestionGenerator logging
# ---------------------------------------------------------------------------


class TestQuestionGeneratorLogging:
    """Verify log output from ClientQuestionGenerator."""

    def test_parse_failure_warning(self, caplog):
        """JSON parse failure in single-atom response should produce a WARNING."""
        from isotope.question_generator.client import ClientQuestionGenerator

        mock_client = MagicMock()
        mock_client.acomplete = AsyncMock(return_value="this is not json at all")
        generator = ClientQuestionGenerator(llm_client=mock_client)

        atom = Atom(content="The sky is blue.", chunk_id="c1", index=0)

        with caplog.at_level(logging.DEBUG, logger="isotope"):
            questions = generator.generate_batch([atom], ["The sky is blue."])

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("JSON parse failed" in r.message for r in warnings), (
            f"Expected 'JSON parse failed' WARNING, got: {[r.message for r in warnings]}"
        )
        # Line-parsing fallback should still produce questions
        assert len(questions) > 0

    def test_batch_failure_warning(self, caplog):
        """When a batch raises an exception, a WARNING should be logged."""
        from isotope.question_generator.base import BatchConfig
        from isotope.question_generator.client import ClientQuestionGenerator

        call_count = 0

        async def mock_acomplete(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("API timeout")
            return '["What is the sky?"]'

        mock_client = MagicMock()
        mock_client.acomplete = mock_acomplete
        generator = ClientQuestionGenerator(llm_client=mock_client)

        atoms = [
            Atom(content="The sky is blue.", chunk_id="c1", index=0),
            Atom(content="Water is wet.", chunk_id="c2", index=0),
        ]

        with caplog.at_level(logging.DEBUG, logger="isotope"):
            # batch_size=1 means 2 batches, 1 fails = 50%, which is at threshold
            generator.generate_batch(
                atoms, ["sky", "water"], config=BatchConfig(batch_size=1, max_concurrent=2)
            )

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("Batch 0 failed" in r.message for r in warnings), (
            f"Expected 'Batch 0 failed' WARNING, got: {[r.message for r in warnings]}"
        )
        assert any("batches failed" in r.message for r in warnings)


# ---------------------------------------------------------------------------
# caplog tests: commands/ingest.py per-file logging
# ---------------------------------------------------------------------------


class TestIngestCommandLogging:
    """Verify per-file log output from the ingest command layer."""

    def test_per_file_summary(self, temp_dir, mock_provider, caplog):
        """Running ingest should log per-file INFO with filepath."""
        from isotope.configuration import LocalStorage
        from isotope.isotope import Isotope

        iso = Isotope(
            provider=mock_provider,
            storage=LocalStorage(temp_dir),
        )

        # Create a test file
        test_file = os.path.join(temp_dir, "doc.txt")
        with open(test_file, "w") as f:
            f.write("The sky is blue and the grass is green.")

        from isotope.commands.ingest import _do_ingest

        with caplog.at_level(logging.DEBUG, logger="isotope"):
            _do_ingest(iso, Path(test_file))

        info_msgs = [r.message for r in caplog.records if r.levelno == logging.INFO]
        assert any("doc.txt" in m and "Ingesting" in m for m in info_msgs), (
            f"Expected 'Ingesting ...doc.txt' INFO, got: {info_msgs}"
        )
        assert any("doc.txt" in m and "Completed" in m for m in info_msgs), (
            f"Expected 'Completed ...doc.txt' INFO, got: {info_msgs}"
        )

    def test_zero_questions_per_file_warning(self, temp_dir, caplog):
        """When a file produces 0 questions, a per-file WARNING should appear."""
        from dataclasses import dataclass
        from typing import Any

        from isotope.atomizer import Atomizer
        from isotope.configuration import LocalStorage
        from isotope.embedder import Embedder
        from isotope.isotope import Isotope
        from isotope.models import Atom as AtomModel
        from isotope.models import EmbeddedQuestion
        from isotope.question_generator import QuestionGenerator
        from isotope.question_generator.base import SyncOnlyGeneratorMixin

        class EmptyGen(SyncOnlyGeneratorMixin, QuestionGenerator):
            def generate_batch(self, atoms, chunk_contents=None, config=None):
                return []

        class SimpleAtomizer(Atomizer):
            def atomize(self, chunk):
                return [AtomModel(content=chunk.content, chunk_id=chunk.id, index=0)]

        class SimpleEmbedder(Embedder):
            def embed_text(self, text):
                return [0.1] * 10

            def embed_texts(self, texts):
                return [[0.1] * 10 for _ in texts]

            def embed_question(self, question):
                return EmbeddedQuestion(question=question, embedding=[0.1] * 10)

            def embed_questions(self, questions):
                return [EmbeddedQuestion(question=q, embedding=[0.1] * 10) for q in questions]

        @dataclass(frozen=True)
        class EmptyProvider:
            _embedder: Any
            _atomizer: Any
            _question_generator: Any

            def build_embedder(self, settings):
                return self._embedder

            def build_atomizer(self, settings):
                return self._atomizer

            def build_question_generator(self, settings):
                return self._question_generator

        provider = EmptyProvider(
            _embedder=SimpleEmbedder(),
            _atomizer=SimpleAtomizer(),
            _question_generator=EmptyGen(),
        )

        iso = Isotope(
            provider=provider,
            storage=LocalStorage(temp_dir),
        )

        test_file = os.path.join(temp_dir, "empty_qs.txt")
        with open(test_file, "w") as f:
            f.write("Some content that will produce atoms but no questions.")

        from isotope.commands.ingest import _do_ingest

        with caplog.at_level(logging.DEBUG, logger="isotope"):
            _do_ingest(iso, Path(test_file))

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("0 questions" in r.message and "empty_qs.txt" in r.message for r in warnings), (
            f"Expected per-file '0 questions' WARNING, got: {[r.message for r in warnings]}"
        )


# ---------------------------------------------------------------------------
# setup_file_logging / teardown_file_logging
# ---------------------------------------------------------------------------


class TestFileLoggingSetupTeardown:
    """Tests for setup_file_logging() and teardown_file_logging()."""

    def test_creates_log_file(self):
        """setup_file_logging should create ingest.log in the data dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            handler, prev_level = setup_file_logging(tmpdir)
            try:
                log_path = os.path.join(tmpdir, "ingest.log")
                assert os.path.exists(log_path)
            finally:
                teardown_file_logging(handler, prev_level)

    def test_run_start_separator(self):
        """The log file should contain a run-start separator."""
        with tempfile.TemporaryDirectory() as tmpdir:
            handler, prev_level = setup_file_logging(tmpdir)
            try:
                handler.flush()
                log_path = os.path.join(tmpdir, "ingest.log")
                with open(log_path) as f:
                    content = f.read()
                assert "--- Ingest run:" in content
            finally:
                teardown_file_logging(handler, prev_level)

    def test_no_duplicate_handlers(self):
        """Calling setup_file_logging twice should add two handlers (caller's responsibility)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            isotope_logger = logging.getLogger("isotope")
            initial_count = len(isotope_logger.handlers)

            h1, lv1 = setup_file_logging(tmpdir)
            h2, lv2 = setup_file_logging(tmpdir)
            try:
                # Two calls = two handlers added
                assert len(isotope_logger.handlers) == initial_count + 2
            finally:
                teardown_file_logging(h2, lv2)
                teardown_file_logging(h1, lv1)

            # After teardown, handlers should be back to initial count
            assert len(isotope_logger.handlers) == initial_count

    def test_level_restoration(self):
        """After teardown, the isotope logger level should be restored."""
        isotope_logger = logging.getLogger("isotope")
        original_level = isotope_logger.level

        with tempfile.TemporaryDirectory() as tmpdir:
            handler, prev_level = setup_file_logging(tmpdir)
            # During setup, level should be DEBUG
            assert isotope_logger.level == logging.DEBUG

            teardown_file_logging(handler, prev_level)

            # After teardown, level should be restored
            assert isotope_logger.level == original_level

    def test_log_messages_written_to_file(self):
        """Log messages emitted during ingest should appear in the log file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            handler, prev_level = setup_file_logging(tmpdir)
            try:
                test_logger = logging.getLogger("isotope.test")
                test_logger.info("test message for file logging")
                handler.flush()

                log_path = os.path.join(tmpdir, "ingest.log")
                with open(log_path) as f:
                    content = f.read()
                assert "test message for file logging" in content
            finally:
                teardown_file_logging(handler, prev_level)

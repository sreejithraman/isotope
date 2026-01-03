"""Shared pytest fixtures."""

import os
import tempfile

import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for stores."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def stores(temp_dir):
    """Create store instances for testing (requires chromadb)."""
    pytest.importorskip("chromadb", reason="This fixture requires chromadb")
    from isotopedb.stores import ChromaVectorStore, SQLiteAtomStore, SQLiteChunkStore

    return {
        "vector_store": ChromaVectorStore(os.path.join(temp_dir, "chroma")),
        "chunk_store": SQLiteChunkStore(os.path.join(temp_dir, "chunks.db")),
        "atom_store": SQLiteAtomStore(os.path.join(temp_dir, "atoms.db")),
    }


@pytest.fixture
def mock_embedder():
    """Create a mock embedder for testing."""
    from isotopedb.embedder import Embedder
    from isotopedb.models import EmbeddedQuestion, Question

    class MockEmbedder(Embedder):
        """Mock embedder that returns fixed vectors."""

        def embed_text(self, text: str) -> list[float]:
            return [0.1] * 10

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            return [[0.1] * 10 for _ in texts]

        def embed_question(self, question: Question) -> EmbeddedQuestion:
            return EmbeddedQuestion(question=question, embedding=[0.1] * 10)

        def embed_questions(self, questions: list[Question]) -> list[EmbeddedQuestion]:
            return [EmbeddedQuestion(question=q, embedding=[0.1] * 10) for q in questions]

    return MockEmbedder()


@pytest.fixture
def mock_generator():
    """Create a mock question generator for testing."""
    from isotopedb.models import Atom, Question
    from isotopedb.question_generator import QuestionGenerator

    class MockGenerator(QuestionGenerator):
        """Mock generator that returns fixed questions."""

        def generate(self, atom: Atom, chunk_content: str = "") -> list[Question]:
            return [
                Question(
                    text=f"Question about {atom.content}?",
                    chunk_id=atom.chunk_id,
                    atom_id=atom.id,
                )
            ]

        def generate_batch(self, atoms: list[Atom], chunk_content: str = "") -> list[Question]:
            questions = []
            for atom in atoms:
                questions.extend(self.generate(atom, chunk_content))
            return questions

    return MockGenerator()


@pytest.fixture
def mock_atomizer():
    """Create a mock atomizer for testing."""
    from isotopedb.atomizer import Atomizer
    from isotopedb.models import Atom, Chunk

    class MockAtomizer(Atomizer):
        """Mock atomizer that returns one atom per chunk."""

        def atomize(self, chunk: Chunk) -> list[Atom]:
            return [Atom(content=chunk.content, chunk_id=chunk.id, index=0)]

    return MockAtomizer()

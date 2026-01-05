# tests/atomizer/test_sentence.py
"""Tests for sentence-based atomizer."""

import pytest

from isotope.atomizer.base import Atomizer
from isotope.atomizer.sentence import SentenceAtomizer
from isotope.models import Chunk


@pytest.fixture
def atomizer():
    """Create a SentenceAtomizer instance."""
    return SentenceAtomizer()


class TestSentenceAtomizer:
    def test_is_atomizer(self, atomizer):
        assert isinstance(atomizer, Atomizer)

    def test_splits_simple_sentences(self, atomizer):
        chunk = Chunk(
            content="Python is a programming language. It was created by Guido.",
            source="test.md",
        )
        atoms = atomizer.atomize(chunk)

        assert len(atoms) == 2
        assert atoms[0].content == "Python is a programming language."
        assert atoms[1].content == "It was created by Guido."

    def test_assigns_correct_chunk_id(self, atomizer):
        chunk = Chunk(content="First sentence. Second sentence.", source="test.md")
        atoms = atomizer.atomize(chunk)

        for atom in atoms:
            assert atom.chunk_id == chunk.id

    def test_assigns_sequential_index(self, atomizer):
        chunk = Chunk(
            content="First. Second. Third.",
            source="test.md",
        )
        atomizer_with_low_min = SentenceAtomizer(min_length=1)
        atoms = atomizer_with_low_min.atomize(chunk)

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

    def test_filters_short_sentences(self):
        atomizer = SentenceAtomizer(min_length=20)
        chunk = Chunk(
            content="Short. This is a longer sentence that will be kept.",
            source="test.md",
        )
        atoms = atomizer.atomize(chunk)

        assert len(atoms) == 1
        assert "longer sentence" in atoms[0].content

    def test_handles_question_marks(self, atomizer):
        chunk = Chunk(
            content="What is Python? It's a programming language.",
            source="test.md",
        )
        atoms = atomizer.atomize(chunk)

        assert len(atoms) == 2
        assert atoms[0].content == "What is Python?"

    def test_handles_exclamation_marks(self, atomizer):
        chunk = Chunk(
            content="Python is great! It's also easy to learn.",
            source="test.md",
        )
        atoms = atomizer.atomize(chunk)

        assert len(atoms) == 2
        assert atoms[0].content == "Python is great!"

    def test_single_sentence(self, atomizer):
        chunk = Chunk(
            content="This is just one sentence.",
            source="test.md",
        )
        atoms = atomizer.atomize(chunk)

        assert len(atoms) == 1
        assert atoms[0].content == "This is just one sentence."

    def test_each_atom_has_unique_id(self, atomizer):
        chunk = Chunk(content="First sentence. Second sentence.", source="test.md")
        atoms = atomizer.atomize(chunk)

        assert atoms[0].id != atoms[1].id

    def test_language_parameter(self):
        """Test that language parameter is accepted."""
        atomizer = SentenceAtomizer(language="en")
        chunk = Chunk(content="Hello world. Goodbye world.", source="test.md")
        atoms = atomizer.atomize(chunk)
        assert len(atoms) == 2


class TestSentenceAtomizerEdgeCases:
    """Tests for edge cases in sentence boundary detection."""

    @pytest.fixture
    def atomizer(self):
        return SentenceAtomizer(min_length=1)

    def test_preserves_honorific_abbreviations(self, atomizer):
        """Dr., Mr., Mrs., Ms. should not cause sentence splits."""
        cases = [
            ("Dr. Smith went to the store.", 1),
            ("Mr. Jones called yesterday.", 1),
            ("Mrs. Brown is here.", 1),
            ("Ms. Davis sent a letter.", 1),
        ]
        for text, expected_count in cases:
            chunk = Chunk(content=text, source="test.md")
            atoms = atomizer.atomize(chunk)
            assert len(atoms) == expected_count, f"Failed for: {text}"

    def test_preserves_country_abbreviations(self, atomizer):
        """U.S., U.K. etc. should not cause sentence splits."""
        chunk = Chunk(content="The U.S. Army is here.", source="test.md")
        atoms = atomizer.atomize(chunk)
        assert len(atoms) == 1
        assert atoms[0].content == "The U.S. Army is here."

    def test_handles_abbreviations_mid_text(self, atomizer):
        """Abbreviations in the middle of multi-sentence text."""
        chunk = Chunk(
            content="Hello world. This is Dr. Brown. He is nice.",
            source="test.md",
        )
        atoms = atomizer.atomize(chunk)
        assert len(atoms) == 3
        assert atoms[1].content == "This is Dr. Brown."

    def test_handles_ellipsis(self, atomizer):
        """Ellipsis (...) should be handled appropriately."""
        chunk = Chunk(content="Wait... What happened next?", source="test.md")
        atoms = atomizer.atomize(chunk)
        # pySBD treats ellipsis as sentence boundary in this context
        assert len(atoms) == 2
        assert atoms[0].content == "Wait..."
        assert atoms[1].content == "What happened next?"

    def test_handles_quoted_speech(self, atomizer):
        """Quoted speech with periods inside quotes."""
        chunk = Chunk(content='"Hello." She said goodbye.', source="test.md")
        atoms = atomizer.atomize(chunk)
        assert len(atoms) == 2
        assert atoms[0].content == '"Hello."'
        assert atoms[1].content == "She said goodbye."

    def test_preserves_decimal_numbers(self, atomizer):
        """Decimal numbers should not cause sentence splits."""
        chunk = Chunk(content="The price is $3.50 per unit.", source="test.md")
        atoms = atomizer.atomize(chunk)
        assert len(atoms) == 1

    def test_preserves_version_numbers(self, atomizer):
        """Version numbers like 2.0 should not cause splits."""
        chunk = Chunk(content="Version 2.0 was released today.", source="test.md")
        atoms = atomizer.atomize(chunk)
        assert len(atoms) == 1

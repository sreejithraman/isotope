# src/isotopedb/generator/base.py
"""QuestionGenerator abstract base class."""

from abc import ABC, abstractmethod

from isotopedb.models import Atom, Question


class QuestionGenerator(ABC):
    """Abstract base class for question generation.

    Implementations of this class generate synthetic questions from atoms.
    This is the core of the reverse-RAG approach: instead of embedding
    the content, we generate questions that the content answers, then
    embed those questions.

    The default implementation uses LiteLLM, but users can implement
    their own using any LLM provider.

    Example:
        class MyQuestionGenerator(QuestionGenerator):
            def generate(self, atom: Atom, chunk_content: str = "") -> list[Question]:
                questions = my_llm_api.generate_questions(atom.content)
                return [
                    Question(text=q, chunk_id=atom.chunk_id, atom_id=atom.id)
                    for q in questions
                ]

            def generate_batch(self, atoms: list[Atom], chunk_content: str = "") -> list[Question]:
                all_questions = []
                for atom in atoms:
                    all_questions.extend(self.generate(atom, chunk_content))
                return all_questions
    """

    @abstractmethod
    def generate(self, atom: Atom, chunk_content: str = "") -> list[Question]:
        """Generate questions for a single atom.

        Args:
            atom: The atom to generate questions for
            chunk_content: Optional context from the parent chunk

        Returns:
            List of Question objects
        """
        ...

    @abstractmethod
    def generate_batch(self, atoms: list[Atom], chunk_content: str = "") -> list[Question]:
        """Generate questions for multiple atoms.

        Args:
            atoms: List of atoms to generate questions for
            chunk_content: Optional context from the parent chunk

        Returns:
            Flat list of all generated Question objects
        """
        ...

"""Ingestion pipeline for Isotope."""

from isotopedb.atomizer import Atomizer
from isotopedb.dedup import Deduplicator
from isotopedb.embedder import Embedder
from isotopedb.generator import DiversityFilter, QuestionGenerator
from isotopedb.models import Chunk, EmbeddedQuestion
from isotopedb.stores import AtomStore, DocStore, VectorStore


class Ingestor:
    """Orchestrates the ingestion pipeline.

    Pipeline:
    1. Deduplicate (remove old chunks for re-ingested sources)
    2. Store chunks in DocStore
    3. Atomize chunks into atomic statements
    4. Store atoms in AtomStore
    5. Generate questions for each atom
    6. Embed questions
    7. Apply diversity filtering (optional)
    8. Store embedded questions in VectorStore
    """

    def __init__(
        self,
        vector_store: VectorStore,
        doc_store: DocStore,
        atom_store: AtomStore,
        atomizer: Atomizer,
        embedder: Embedder,
        generator: QuestionGenerator,
        deduplicator: Deduplicator,
        diversity_filter: DiversityFilter | None = None,
    ) -> None:
        """Initialize the ingestor with all required components."""
        self.vector_store = vector_store
        self.doc_store = doc_store
        self.atom_store = atom_store
        self.atomizer = atomizer
        self.embedder = embedder
        self.generator = generator
        self.deduplicator = deduplicator
        self.diversity_filter = diversity_filter

    def ingest_chunks(self, chunks: list[Chunk]) -> dict:
        """Ingest a list of chunks through the full pipeline.

        Args:
            chunks: List of Chunk objects to ingest

        Returns:
            Dict with ingestion statistics:
            - chunks: number of chunks ingested
            - chunks_removed: number of old chunks removed (dedup)
            - atoms: number of atoms created
            - questions: number of questions generated
            - questions_filtered: number of questions removed by diversity filter
        """
        if not chunks:
            return {
                "chunks": 0,
                "chunks_removed": 0,
                "atoms": 0,
                "questions": 0,
                "questions_filtered": 0,
            }

        # Step 1: Deduplication - remove old chunks from same sources
        chunk_ids_to_remove = self.deduplicator.get_chunks_to_remove(chunks, self.doc_store)
        chunks_removed = len(chunk_ids_to_remove)

        if chunk_ids_to_remove:
            # Remove from all stores
            self.vector_store.delete_by_chunk_ids(chunk_ids_to_remove)
            self.atom_store.delete_by_chunk_ids(chunk_ids_to_remove)
            for chunk_id in chunk_ids_to_remove:
                self.doc_store.delete(chunk_id)

        # Step 2: Store chunks
        for chunk in chunks:
            self.doc_store.put(chunk)

        # Step 3: Atomize and store atoms
        all_atoms = []
        for chunk in chunks:
            atoms = self.atomizer.atomize(chunk)
            for i, atom in enumerate(atoms):
                atom.index = i
            all_atoms.extend(atoms)
            self.atom_store.put_many(atoms)

        # Step 4: Generate questions
        all_questions = []
        for atom in all_atoms:
            # Get chunk content for context
            chunk = self.doc_store.get(atom.chunk_id)
            chunk_content = chunk.content if chunk else ""
            questions = self.generator.generate(atom, chunk_content)
            all_questions.extend(questions)

        # Step 5: Embed questions
        if all_questions:
            question_texts = [q.text for q in all_questions]
            embeddings = self.embedder.embed_texts(question_texts)

            # Create EmbeddedQuestion objects
            embedded_questions = [
                EmbeddedQuestion(question=q, embedding=e)
                for q, e in zip(all_questions, embeddings)
            ]

            # Step 6: Apply diversity filter (optional)
            questions_filtered = 0
            if self.diversity_filter:
                original_count = len(embedded_questions)
                embedded_questions = self.diversity_filter.filter(embedded_questions)
                questions_filtered = original_count - len(embedded_questions)

            # Step 7: Store in vector store
            self.vector_store.add(embedded_questions)
        else:
            embedded_questions = []
            questions_filtered = 0

        return {
            "chunks": len(chunks),
            "chunks_removed": chunks_removed,
            "atoms": len(all_atoms),
            "questions": len(embedded_questions),
            "questions_filtered": questions_filtered,
        }

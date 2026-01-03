"""Ingestion pipeline for Isotope."""

from collections.abc import Callable

from isotopedb.atomizer import Atomizer
from isotopedb.dedup import Deduplicator
from isotopedb.embedder import Embedder
from isotopedb.generator import DiversityFilter, QuestionGenerator
from isotopedb.models import Chunk, EmbeddedQuestion
from isotopedb.stores import AtomStore, DocStore, VectorStore

ProgressCallback = Callable[[str, int, int, str], None]


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

    def ingest_chunks(
        self,
        chunks: list[Chunk],
        on_progress: ProgressCallback | None = None,
    ) -> dict:
        """Ingest a list of chunks through the full pipeline.

        Args:
            chunks: List of Chunk objects to ingest
            on_progress: Optional callback(event, current, total, message)

        Returns:
            Dict with ingestion statistics:
            - chunks: number of chunks ingested
            - chunks_removed: number of old chunks removed (dedup)
            - atoms: number of atoms created
            - questions: number of questions generated
            - questions_filtered: number of questions removed by diversity filter
        """

        def progress(event: str, current: int, total: int, message: str = "") -> None:
            if on_progress:
                on_progress(event, current, total, message)

        if not chunks:
            return {
                "chunks": 0,
                "chunks_removed": 0,
                "atoms": 0,
                "questions": 0,
                "questions_filtered": 0,
            }

        # Step 1: Deduplication
        progress("deduplicating", 0, 1, "Checking for existing chunks...")
        chunk_ids_to_remove = self.deduplicator.get_chunks_to_remove(chunks, self.doc_store)
        chunks_removed = len(chunk_ids_to_remove)

        if chunk_ids_to_remove:
            self.vector_store.delete_by_chunk_ids(chunk_ids_to_remove)
            self.atom_store.delete_by_chunk_ids(chunk_ids_to_remove)
            self.doc_store.delete_many(chunk_ids_to_remove)
        progress("deduplicating", 1, 1, f"Removed {chunks_removed} old chunks")

        # Step 2: Store chunks
        progress("storing", 0, 1, f"Storing {len(chunks)} chunks...")
        self.doc_store.put_many(chunks)
        progress("storing", 1, 1, "Storing chunks complete")

        # Step 3: Atomize
        progress("atomizing", 0, len(chunks), "Atomizing chunks...")
        all_atoms = []
        for i, chunk in enumerate(chunks):
            atoms = self.atomizer.atomize(chunk)
            all_atoms.extend(atoms)
            self.atom_store.put_many(atoms)
            progress("atomizing", i + 1, len(chunks), f"Atomized {i + 1}/{len(chunks)} chunks")

        # Step 4: Generate questions
        progress("generating", 0, len(all_atoms), "Generating questions...")
        all_questions = []
        for i, atom in enumerate(all_atoms):
            parent_chunk = self.doc_store.get(atom.chunk_id)
            chunk_content = parent_chunk.content if parent_chunk else ""
            questions = self.generator.generate(atom, chunk_content)
            all_questions.extend(questions)
            progress(
                "generating",
                i + 1,
                len(all_atoms),
                f"Generated questions for {i + 1}/{len(all_atoms)} atoms",
            )

        # Step 5: Embed questions
        if all_questions:
            progress("embedding", 0, 1, f"Embedding {len(all_questions)} questions...")
            question_texts = [q.text for q in all_questions]
            embeddings = self.embedder.embed_texts(question_texts)
            progress("embedding", 1, 1, "Embedding complete")

            if len(all_questions) != len(embeddings):
                raise RuntimeError(
                    f"Embedding count mismatch: {len(all_questions)} questions, "
                    f"{len(embeddings)} embeddings"
                )
            embedded_questions = [
                EmbeddedQuestion(question=q, embedding=e)
                for q, e in zip(all_questions, embeddings, strict=True)
            ]

            # Step 6: Diversity filter
            questions_filtered = 0
            if self.diversity_filter:
                progress("filtering", 0, 1, "Applying diversity filter...")
                original_count = len(embedded_questions)
                embedded_questions = self.diversity_filter.filter(embedded_questions)
                questions_filtered = original_count - len(embedded_questions)
                progress("filtering", 1, 1, f"Filtered {questions_filtered} similar questions")

            # Step 7: Store
            progress("indexing", 0, 1, f"Indexing {len(embedded_questions)} questions...")
            self.vector_store.add(embedded_questions)
            progress("indexing", 1, 1, "Indexing complete")
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

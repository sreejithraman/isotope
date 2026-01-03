"""Ingestion pipeline for Isotope."""

from collections.abc import Callable

from isotopedb.atomizer import Atomizer
from isotopedb.embedder import Embedder
from isotopedb.models import Chunk, EmbeddedQuestion
from isotopedb.question_generator import DiversityFilter, FilterScope, QuestionGenerator
from isotopedb.stores import AtomStore, ChunkStore, VectorStore

ProgressCallback = Callable[[str, int, int, str], None]


class Ingestor:
    """Orchestrates the ingestion pipeline.

    Pipeline:
    1. Store chunks in ChunkStore
    2. Atomize chunks into atomic statements
    3. Store atoms in AtomStore
    4. Generate questions for each atom
    5. Embed questions
    6. Apply diversity filtering (optional)
    7. Store embedded questions in VectorStore

    Note: Source-level deduplication (re-ingestion) is handled by Isotope.ingest_file(),
    not by the Ingestor. The Ingestor simply processes whatever chunks it receives.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        chunk_store: ChunkStore,
        atom_store: AtomStore,
        atomizer: Atomizer,
        embedder: Embedder,
        question_generator: QuestionGenerator,
        diversity_filter: DiversityFilter | None = None,
        diversity_scope: FilterScope = "global",
    ) -> None:
        """Initialize the ingestor with all required components.

        Args:
            vector_store: Store for question embeddings
            chunk_store: Store for chunks
            atom_store: Store for atomic statements
            atomizer: Component to split chunks into atoms
            embedder: Component to embed questions
            question_generator: Component to generate questions from atoms
            diversity_filter: Optional filter to remove duplicate questions
            diversity_scope: Scope for diversity filtering. Options:
                - "global": Filter across all questions (default, paper-validated)
                - "per_chunk": Filter within each chunk (~100x faster)
                - "per_atom": Filter within each atom (~1000x faster)
        """
        self.vector_store = vector_store
        self.chunk_store = chunk_store
        self.atom_store = atom_store
        self.atomizer = atomizer
        self.embedder = embedder
        self.question_generator = question_generator
        self.diversity_filter = diversity_filter
        self.diversity_scope = diversity_scope

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
                "atoms": 0,
                "questions": 0,
                "questions_filtered": 0,
            }

        # Step 1: Store chunks
        progress("storing", 0, 1, f"Storing {len(chunks)} chunks...")
        self.chunk_store.put_many(chunks)
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

        # Build chunk lookup from input to avoid N database reads
        chunk_content_map = {chunk.id: chunk.content for chunk in chunks}

        all_questions = []
        for i, atom in enumerate(all_atoms):
            chunk_content = chunk_content_map.get(atom.chunk_id, "")
            questions = self.question_generator.generate(atom, chunk_content)
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
                scope = self.diversity_scope
                progress("filtering", 0, 1, f"Applying diversity filter ({scope})...")
                original_count = len(embedded_questions)
                embedded_questions = self.diversity_filter.filter_by_scope(
                    embedded_questions, self.diversity_scope
                )
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
            "atoms": len(all_atoms),
            "questions": len(embedded_questions),
            "questions_filtered": questions_filtered,
        }

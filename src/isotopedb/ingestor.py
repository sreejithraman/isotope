"""Ingestion pipeline for Isotope."""

from isotopedb.atomizer import Atomizer
from isotopedb.dedup import Deduplicator
from isotopedb.embedder import Embedder
from isotopedb.generator import DiversityFilter, QuestionGenerator
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

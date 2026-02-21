# Hybrid Retrieval Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add chunk-level vector search fallback so retrieval falls back to direct chunk search when question-match confidence is low.

**Architecture:** New `ChunkEmbeddingStore` ABC + ChromaDB implementation stores chunk embeddings at ingest time. `Retriever.get_context()` checks the best question-match score against a threshold; if below, it searches chunk embeddings and appends deduped results. Question collection renamed from `"isotope"` to `"isotope_questions"`.

**Tech Stack:** Python 3.11+, Pydantic 2.x, ChromaDB, pytest

**Design doc:** `docs/plans/2026-02-21-hybrid-retrieval-design.md`

---

### Task 1: ChunkEmbeddingStore ABC

**Files:**
- Modify: `src/isotope/stores/base.py`
- Test: `tests/stores/test_chunk_embedding_store.py`

**Step 1: Write the failing test**

Create `tests/stores/test_chunk_embedding_store.py`:

```python
"""Tests for ChunkEmbeddingStore ABC."""

from isotope.stores.base import ChunkEmbeddingStore


class TestChunkEmbeddingStoreABC:
    def test_cannot_instantiate(self):
        """ChunkEmbeddingStore is abstract and cannot be instantiated."""
        import pytest

        with pytest.raises(TypeError):
            ChunkEmbeddingStore()  # type: ignore[abstract]

    def test_has_required_methods(self):
        """ChunkEmbeddingStore defines the expected abstract methods."""
        assert hasattr(ChunkEmbeddingStore, "add")
        assert hasattr(ChunkEmbeddingStore, "search")
        assert hasattr(ChunkEmbeddingStore, "delete_by_chunk_ids")
        assert hasattr(ChunkEmbeddingStore, "count")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/stores/test_chunk_embedding_store.py -v`
Expected: FAIL with `ImportError` — `ChunkEmbeddingStore` doesn't exist yet

**Step 3: Write minimal implementation**

Add to `src/isotope/stores/base.py` after the `EmbeddedQuestionStore` class:

```python
class ChunkEmbeddingStore(ABC):
    """Abstract base class for chunk embedding storage.

    Stores chunk-level embeddings for hybrid retrieval fallback.
    When question-match confidence is low, the retriever searches
    this store directly for relevant chunks.
    """

    @abstractmethod
    def add(self, chunk_ids: list[str], embeddings: list[list[float]]) -> None:
        """Add chunk embeddings to the store."""
        ...

    @abstractmethod
    def search(self, embedding: list[float], k: int = 5) -> list[tuple[str, float]]:
        """Search for similar chunks. Returns (chunk_id, score) pairs ordered by relevance."""
        ...

    @abstractmethod
    def delete_by_chunk_ids(self, chunk_ids: list[str]) -> None:
        """Delete embeddings for the given chunk IDs."""
        ...

    @abstractmethod
    def count(self) -> int:
        """Count the total number of chunk embeddings in the store."""
        ...
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/stores/test_chunk_embedding_store.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/isotope/stores/base.py tests/stores/test_chunk_embedding_store.py
git commit -m "feat: add ChunkEmbeddingStore ABC"
```

---

### Task 2: ChromaChunkEmbeddingStore implementation

**Files:**
- Modify: `src/isotope/stores/chroma.py`
- Test: `tests/stores/test_chroma.py`

**Step 1: Write the failing tests**

Add to `tests/stores/test_chroma.py`:

```python
from isotope.stores.base import ChunkEmbeddingStore
from isotope.stores.chroma import ChromaChunkEmbeddingStore


@pytest.fixture
def chunk_embedding_store(temp_dir):
    """Create a ChromaChunkEmbeddingStore instance."""
    return ChromaChunkEmbeddingStore(temp_dir)


class TestChromaChunkEmbeddingStore:
    def test_is_chunk_embedding_store(self, chunk_embedding_store):
        assert isinstance(chunk_embedding_store, ChunkEmbeddingStore)

    def test_add_and_search(self, chunk_embedding_store):
        chunk_embedding_store.add(
            chunk_ids=["c1", "c2"],
            embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        )
        results = chunk_embedding_store.search([0.9, 0.1, 0.0], k=2)
        assert len(results) == 2
        # First result should be closest
        assert results[0][0] == "c1"
        assert results[0][1] > results[1][1]

    def test_search_returns_chunk_id_score_tuples(self, chunk_embedding_store):
        chunk_embedding_store.add(chunk_ids=["c1"], embeddings=[[1.0, 0.0, 0.0]])
        results = chunk_embedding_store.search([1.0, 0.0, 0.0], k=1)
        assert len(results) == 1
        chunk_id, score = results[0]
        assert chunk_id == "c1"
        assert isinstance(score, float)

    def test_delete_by_chunk_ids(self, chunk_embedding_store):
        chunk_embedding_store.add(
            chunk_ids=["c1", "c2", "c3"],
            embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        )
        chunk_embedding_store.delete_by_chunk_ids(["c1", "c2"])
        results = chunk_embedding_store.search([0.0, 0.0, 1.0], k=10)
        assert len(results) == 1
        assert results[0][0] == "c3"

    def test_count(self, chunk_embedding_store):
        assert chunk_embedding_store.count() == 0
        chunk_embedding_store.add(chunk_ids=["c1", "c2"], embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        assert chunk_embedding_store.count() == 2

    def test_search_empty_store(self, chunk_embedding_store):
        results = chunk_embedding_store.search([1.0, 0.0, 0.0], k=5)
        assert results == []

    def test_delete_empty_list(self, chunk_embedding_store):
        """delete_by_chunk_ids with empty list should not error."""
        chunk_embedding_store.delete_by_chunk_ids([])
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/stores/test_chroma.py::TestChromaChunkEmbeddingStore -v`
Expected: FAIL with `ImportError` — `ChromaChunkEmbeddingStore` doesn't exist yet

**Step 3: Write minimal implementation**

Add to `src/isotope/stores/chroma.py`:

```python
from isotope.stores.base import ChunkEmbeddingStore


class ChromaChunkEmbeddingStore(ChunkEmbeddingStore):
    """ChromaDB-based chunk embedding store for hybrid retrieval fallback."""

    def __init__(self, persist_dir: str, collection_name: str = "isotope_chunks") -> None:
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def close(self) -> None:
        """Close the store and release resources."""
        self._collection = None  # type: ignore[assignment]
        try:
            if self._client is not None and hasattr(self._client, "_system"):
                self._client._system.stop()
        except Exception:
            pass
        self._client = None  # type: ignore[assignment]

    def add(self, chunk_ids: list[str], embeddings: list[list[float]]) -> None:
        if not chunk_ids:
            return
        self._collection.add(
            ids=chunk_ids,
            embeddings=embeddings,  # type: ignore[arg-type]
        )

    def search(self, embedding: list[float], k: int = 5) -> list[tuple[str, float]]:
        if self._collection.count() == 0:
            return []
        results = self._collection.query(
            query_embeddings=[embedding],  # type: ignore[arg-type]
            n_results=min(k, self._collection.count()),
            include=["distances"],
        )
        ids = results["ids"][0]
        distances = results["distances"][0]  # type: ignore[index]
        return [(cid, 1.0 - dist) for cid, dist in zip(ids, distances, strict=True)]

    def delete_by_chunk_ids(self, chunk_ids: list[str]) -> None:
        if not chunk_ids:
            return
        self._collection.delete(ids=chunk_ids)

    def count(self) -> int:
        return self._collection.count()
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/stores/test_chroma.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/isotope/stores/chroma.py tests/stores/test_chroma.py
git commit -m "feat: add ChromaChunkEmbeddingStore"
```

---

### Task 3: Rename question collection and update exports

**Files:**
- Modify: `src/isotope/stores/chroma.py:16` — change default `collection_name` from `"isotope"` to `"isotope_questions"`
- Modify: `src/isotope/stores/__init__.py` — export `ChunkEmbeddingStore` and `ChromaChunkEmbeddingStore`
- Modify: `src/isotope/__init__.py` — export `ChunkEmbeddingStore` and `ChromaChunkEmbeddingStore`
- Test: existing tests in `tests/stores/test_chroma.py`

**Step 1: Update the default collection name**

In `src/isotope/stores/chroma.py`, `ChromaEmbeddedQuestionStore.__init__`:

Change: `collection_name: str = "isotope"`
To: `collection_name: str = "isotope_questions"`

**Step 2: Update `src/isotope/stores/__init__.py`**

Add imports for `ChunkEmbeddingStore` and `ChromaChunkEmbeddingStore`:

```python
from isotope.stores.base import AtomStore, ChunkEmbeddingStore, ChunkStore, EmbeddedQuestionStore, SourceRegistry

# In the try block for chroma imports, add:
from isotope.stores.chroma import ChromaChunkEmbeddingStore, ChromaEmbeddedQuestionStore

# Add to __all__:
"ChunkEmbeddingStore",
"ChromaChunkEmbeddingStore",
```

**Step 3: Update `src/isotope/__init__.py`**

Add `ChunkEmbeddingStore` and `ChromaChunkEmbeddingStore` to the stores import and `__all__`.

**Step 4: Run all existing tests to verify no regressions**

Run: `pytest tests/stores/test_chroma.py tests/test_isotope.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/isotope/stores/chroma.py src/isotope/stores/__init__.py src/isotope/__init__.py
git commit -m "refactor: rename question collection to isotope_questions, export chunk embedding store"
```

---

### Task 4: Make SearchResult fields optional

**Files:**
- Modify: `src/isotope/models/results.py`
- Modify: `tests/models/test_results.py`

**Step 1: Write the failing test**

Add to `tests/models/test_results.py`:

```python
class TestSearchResultOptionalFields:
    def test_create_search_result_without_question_and_atom(self):
        """Chunk-fallback results have no question or atom."""
        chunk = Chunk(content="Python is great", source="test.md")
        result = SearchResult(chunk=chunk, score=0.85, question=None, atom=None)
        assert result.question is None
        assert result.atom is None
        assert result.chunk == chunk
        assert result.score == 0.85

    def test_create_search_result_with_all_fields(self):
        """Question-match results still work with all fields."""
        chunk = Chunk(content="Python is great", source="test.md")
        atom = Atom(content="Python is great", chunk_id=chunk.id)
        question = Question(text="What is Python?", chunk_id=chunk.id, atom_id=atom.id)
        result = SearchResult(question=question, chunk=chunk, score=0.95, atom=atom)
        assert result.question == question
        assert result.atom == atom
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/models/test_results.py::TestSearchResultOptionalFields -v`
Expected: FAIL — `question` and `atom` are required

**Step 3: Update the model**

In `src/isotope/models/results.py`:

```python
class SearchResult(BaseModel):
    """A single search result — chunk with optional question/atom context."""

    question: Question | None = None
    chunk: Chunk
    score: float
    atom: Atom | None = None
```

**Step 4: Update the existing test that checks atom is required**

In `tests/models/test_results.py`, update `test_search_result_requires_atom`:
- Remove or rewrite this test since atom is now optional
- Replace with a test that verifies defaults are None:

```python
def test_search_result_defaults_to_none(self):
    """question and atom default to None."""
    chunk = Chunk(content="Content", source="test.md")
    result = SearchResult(chunk=chunk, score=0.5)
    assert result.question is None
    assert result.atom is None
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/models/test_results.py -v`
Expected: ALL PASS

**Step 6: Run full test suite to check for breakage**

Run: `pytest -v`
Expected: ALL PASS (existing code that passes `question=` and `atom=` as positional/keyword still works since they now have defaults)

**Step 7: Commit**

```bash
git add src/isotope/models/results.py tests/models/test_results.py
git commit -m "feat: make SearchResult question and atom optional for chunk fallback"
```

---

### Task 5: Add hybrid_confidence_threshold to Settings

**Files:**
- Modify: `src/isotope/settings.py`
- Modify: `tests/test_settings.py`

**Step 1: Write the failing test**

Add to `tests/test_settings.py`:

```python
class TestHybridSettings:
    def test_default_hybrid_threshold(self):
        settings = Settings()
        assert settings.hybrid_confidence_threshold == 0.7

    def test_custom_hybrid_threshold(self):
        settings = Settings(hybrid_confidence_threshold=0.5)
        assert settings.hybrid_confidence_threshold == 0.5

    def test_hybrid_disabled_with_zero(self):
        settings = Settings(hybrid_confidence_threshold=0)
        assert settings.hybrid_confidence_threshold == 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_settings.py::TestHybridSettings -v`
Expected: FAIL — field doesn't exist yet

**Step 3: Add the field**

In `src/isotope/settings.py`, in the `Settings` class, under the `# Retrieval` section:

```python
    # Hybrid retrieval fallback
    # When best question-match score is below this threshold, also search chunk embeddings.
    # 0 = disabled (pure question-matching), 1.0 = always include chunk fallback.
    hybrid_confidence_threshold: float = 0.7
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_settings.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/isotope/settings.py tests/test_settings.py
git commit -m "feat: add hybrid_confidence_threshold setting"
```

---

### Task 6: Hybrid retrieval logic in Retriever

**Files:**
- Modify: `src/isotope/retriever.py`
- Modify: `tests/test_retriever.py`

**Step 1: Write the failing tests**

Add to `tests/test_retriever.py`. These tests use the `stores` fixture from conftest and need a `chunk_embedding_store` fixture. Add a fixture first:

```python
# At the top of the file, add import:
from isotope.stores.chroma import ChromaChunkEmbeddingStore

# Add fixture (or add to conftest.py):
@pytest.fixture
def chunk_embedding_store(tmp_path):
    store = ChromaChunkEmbeddingStore(str(tmp_path / "chunk_embeddings"))
    yield store
    store.close()
```

Then add the test class:

```python
class TestRetrieverHybridFallback:
    @pytest.mark.mock_integration
    @patch("isotope.providers.litellm.client.litellm.embedding")
    def test_no_fallback_when_disabled(self, mock_embedding, stores, chunk_embedding_store):
        """threshold=0 disables hybrid fallback entirely."""
        mock_embedding.return_value = MagicMock(data=[{"embedding": [1.0, 0.0, 0.0], "index": 0}])
        chunk, atom, question = create_test_data(stores)

        # Add chunk to chunk_embedding_store
        chunk_embedding_store.add([chunk.id], [[1.0, 0.0, 0.0]])

        retriever = Retriever(
            embedded_question_store=stores["embedded_question_store"],
            chunk_store=stores["chunk_store"],
            atom_store=stores["atom_store"],
            embedder=ClientEmbedder(embedding_client=LiteLLMEmbeddingClient()),
            chunk_embedding_store=chunk_embedding_store,
            hybrid_confidence_threshold=0,
        )
        results = retriever.get_context("What is Python?")
        # Only question-match results, no fallback
        assert len(results) == 1
        assert results[0].question is not None

    @pytest.mark.mock_integration
    @patch("isotope.providers.litellm.client.litellm.embedding")
    def test_no_fallback_when_scores_above_threshold(self, mock_embedding, stores, chunk_embedding_store):
        """No fallback when best question score >= threshold."""
        mock_embedding.return_value = MagicMock(data=[{"embedding": [1.0, 0.0, 0.0], "index": 0}])
        chunk, atom, question = create_test_data(stores)
        chunk_embedding_store.add([chunk.id], [[1.0, 0.0, 0.0]])

        retriever = Retriever(
            embedded_question_store=stores["embedded_question_store"],
            chunk_store=stores["chunk_store"],
            atom_store=stores["atom_store"],
            embedder=ClientEmbedder(embedding_client=LiteLLMEmbeddingClient()),
            chunk_embedding_store=chunk_embedding_store,
            hybrid_confidence_threshold=0.5,  # Score will be ~1.0, above threshold
        )
        results = retriever.get_context("What is Python?")
        assert all(r.question is not None for r in results)

    @pytest.mark.mock_integration
    @patch("isotope.providers.litellm.client.litellm.embedding")
    def test_fallback_when_scores_below_threshold(self, mock_embedding, stores, chunk_embedding_store):
        """Fallback triggers when best question score < threshold."""
        mock_embedding.return_value = MagicMock(data=[{"embedding": [1.0, 0.0, 0.0], "index": 0}])

        # Add question with an embedding that won't match well
        chunk = Chunk(content="Python info.", source="test.md")
        stores["chunk_store"].put(chunk)
        atom = Atom(content="Python info.", chunk_id=chunk.id)
        stores["atom_store"].put(atom)
        q = Question(text="What is Python?", chunk_id=chunk.id, atom_id=atom.id)
        eq = EmbeddedQuestion(question=q, embedding=[0.0, 1.0, 0.0])  # Low match with [1,0,0]
        stores["embedded_question_store"].add([eq])

        # Add a different chunk to chunk embedding store that matches well
        chunk2 = Chunk(content="Extra Python info.", source="test2.md")
        stores["chunk_store"].put(chunk2)
        chunk_embedding_store.add([chunk2.id], [[0.95, 0.05, 0.0]])  # Close to query

        retriever = Retriever(
            embedded_question_store=stores["embedded_question_store"],
            chunk_store=stores["chunk_store"],
            atom_store=stores["atom_store"],
            embedder=ClientEmbedder(embedding_client=LiteLLMEmbeddingClient()),
            chunk_embedding_store=chunk_embedding_store,
            hybrid_confidence_threshold=0.9,  # Question score will be low, triggers fallback
        )
        results = retriever.get_context("What is Python?", k=5)
        # Should have results from both paths
        has_question_result = any(r.question is not None for r in results)
        has_chunk_result = any(r.question is None for r in results)
        assert has_question_result
        assert has_chunk_result

    @pytest.mark.mock_integration
    @patch("isotope.providers.litellm.client.litellm.embedding")
    def test_fallback_fills_all_slots_when_no_question_results(self, mock_embedding, stores, chunk_embedding_store):
        """When question store is empty, fallback fills all k slots."""
        mock_embedding.return_value = MagicMock(data=[{"embedding": [1.0, 0.0, 0.0], "index": 0}])

        # Don't add any questions — only chunk embeddings
        chunk = Chunk(content="Python info.", source="test.md")
        stores["chunk_store"].put(chunk)
        chunk_embedding_store.add([chunk.id], [[1.0, 0.0, 0.0]])

        retriever = Retriever(
            embedded_question_store=stores["embedded_question_store"],
            chunk_store=stores["chunk_store"],
            atom_store=stores["atom_store"],
            embedder=ClientEmbedder(embedding_client=LiteLLMEmbeddingClient()),
            chunk_embedding_store=chunk_embedding_store,
            hybrid_confidence_threshold=0.7,
        )
        results = retriever.get_context("What is Python?", k=5)
        assert len(results) == 1  # Only 1 chunk available
        assert results[0].question is None  # Came from chunk fallback
        assert results[0].chunk.id == chunk.id

    @pytest.mark.mock_integration
    @patch("isotope.providers.litellm.client.litellm.embedding")
    def test_fallback_deduplicates_by_chunk_id(self, mock_embedding, stores, chunk_embedding_store):
        """Same chunk from both paths appears only once."""
        mock_embedding.return_value = MagicMock(data=[{"embedding": [1.0, 0.0, 0.0], "index": 0}])

        chunk = Chunk(content="Python info.", source="test.md")
        stores["chunk_store"].put(chunk)
        atom = Atom(content="Python info.", chunk_id=chunk.id)
        stores["atom_store"].put(atom)

        # Same chunk in both stores — question won't match well
        q = Question(text="What is Python?", chunk_id=chunk.id, atom_id=atom.id)
        eq = EmbeddedQuestion(question=q, embedding=[0.0, 1.0, 0.0])
        stores["embedded_question_store"].add([eq])

        chunk_embedding_store.add([chunk.id], [[1.0, 0.0, 0.0]])

        retriever = Retriever(
            embedded_question_store=stores["embedded_question_store"],
            chunk_store=stores["chunk_store"],
            atom_store=stores["atom_store"],
            embedder=ClientEmbedder(embedding_client=LiteLLMEmbeddingClient()),
            chunk_embedding_store=chunk_embedding_store,
            hybrid_confidence_threshold=0.9,
        )
        results = retriever.get_context("What is Python?", k=5)
        chunk_ids = [r.chunk.id for r in results]
        assert len(chunk_ids) == len(set(chunk_ids))  # No duplicates

    @pytest.mark.mock_integration
    @patch("isotope.providers.litellm.client.litellm.embedding")
    def test_results_capped_at_k(self, mock_embedding, stores, chunk_embedding_store):
        """Combined results don't exceed k."""
        mock_embedding.return_value = MagicMock(data=[{"embedding": [1.0, 0.0, 0.0], "index": 0}])

        # Add many chunks to both stores
        for i in range(10):
            chunk = Chunk(content=f"Chunk {i}", source=f"test{i}.md")
            stores["chunk_store"].put(chunk)
            atom = Atom(content=f"Chunk {i}", chunk_id=chunk.id)
            stores["atom_store"].put(atom)
            q = Question(text=f"Q{i}?", chunk_id=chunk.id, atom_id=atom.id)
            eq = EmbeddedQuestion(question=q, embedding=[0.0, 1.0, 0.0])  # Low match
            stores["embedded_question_store"].add([eq])
            chunk_embedding_store.add([chunk.id], [[1.0, 0.0, 0.0]])

        retriever = Retriever(
            embedded_question_store=stores["embedded_question_store"],
            chunk_store=stores["chunk_store"],
            atom_store=stores["atom_store"],
            embedder=ClientEmbedder(embedding_client=LiteLLMEmbeddingClient()),
            chunk_embedding_store=chunk_embedding_store,
            hybrid_confidence_threshold=0.9,
            default_k=3,
        )
        results = retriever.get_context("query")
        assert len(results) <= 3
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_retriever.py::TestRetrieverHybridFallback -v`
Expected: FAIL — `Retriever` doesn't accept `chunk_embedding_store` yet

**Step 3: Implement hybrid retrieval in Retriever**

Modify `src/isotope/retriever.py`:

- Add `chunk_embedding_store` and `hybrid_confidence_threshold` params to `__init__`
- Modify `get_context()` to check threshold and run chunk fallback

Key logic for `get_context()`:

```python
def get_context(self, query: str, k: int | None = None) -> list[SearchResult]:
    k = self.default_k if k is None else k

    # Step 1: Embed query
    query_embedding = self.embedder.embed_text(query)

    # Step 2: Search question embeddings
    question_scores = self.embedded_question_store.search(query_embedding, k=k)

    # Step 3: Build question-match results
    results = []
    seen_chunk_ids: set[str] = set()
    for question, score in question_scores:
        chunk = self.chunk_store.get(question.chunk_id)
        atom = self.atom_store.get(question.atom_id)
        if chunk and atom:
            results.append(SearchResult(question=question, chunk=chunk, score=score, atom=atom))
            seen_chunk_ids.add(chunk.id)

    # Step 4: Hybrid fallback
    if self._should_fallback(results):
        remaining_k = k - len(results)
        if remaining_k > 0 and self.chunk_embedding_store is not None:
            chunk_results = self.chunk_embedding_store.search(query_embedding, k=remaining_k + len(seen_chunk_ids))
            for chunk_id, score in chunk_results:
                if chunk_id in seen_chunk_ids:
                    continue
                chunk = self.chunk_store.get(chunk_id)
                if chunk:
                    results.append(SearchResult(chunk=chunk, score=score))
                    seen_chunk_ids.add(chunk_id)
                if len(results) >= k:
                    break

    return results[:k]

def _should_fallback(self, results: list[SearchResult]) -> bool:
    """Check if hybrid fallback should run."""
    if self.hybrid_confidence_threshold <= 0:
        return False
    if not results:
        return True
    best_score = max(r.score for r in results)
    return best_score < self.hybrid_confidence_threshold
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_retriever.py -v`
Expected: ALL PASS (both old and new tests)

**Step 5: Commit**

```bash
git add src/isotope/retriever.py tests/test_retriever.py
git commit -m "feat: add hybrid retrieval fallback to Retriever"
```

---

### Task 7: Wire chunk embedding into Ingestor

**Files:**
- Modify: `src/isotope/ingestor.py`
- Modify: `tests/test_ingestor.py`

**Step 1: Write the failing test**

Add to `tests/test_ingestor.py`:

```python
from isotope.stores.chroma import ChromaChunkEmbeddingStore


class TestIngestorChunkEmbedding:
    @pytest.mark.mock_integration
    @patch("isotope.providers.litellm.client.litellm.embedding")
    @patch("isotope.providers.litellm.client.litellm.acompletion")
    def test_ingest_embeds_chunks(self, mock_acompletion, mock_embedding, stores, tmp_path):
        """Ingestor embeds chunks and stores them in chunk_embedding_store."""
        mock_acompletion.return_value = MagicMock(
            choices=[Choices(finish_reason="stop", index=0, message=Message(role="assistant", content='["Q1?"]'))]
        )

        def make_embeddings(*args, **kwargs):
            input_texts = kwargs.get("input", args[1] if len(args) > 1 else [])
            if isinstance(input_texts, str):
                input_texts = [input_texts]
            return MagicMock(
                data=[{"embedding": [0.1, 0.2, 0.3], "index": i} for i in range(len(input_texts))]
            )

        mock_embedding.side_effect = make_embeddings

        chunk_embedding_store = ChromaChunkEmbeddingStore(str(tmp_path / "chunk_emb"))

        ingestor = Ingestor(
            embedded_question_store=stores["embedded_question_store"],
            chunk_store=stores["chunk_store"],
            atom_store=stores["atom_store"],
            atomizer=SentenceAtomizer(),
            embedder=ClientEmbedder(embedding_client=LiteLLMEmbeddingClient()),
            question_generator=ClientQuestionGenerator(llm_client=LiteLLMClient()),
            chunk_embedding_store=chunk_embedding_store,
        )

        chunk = Chunk(content="Python is great.", source="test.md")
        ingestor.ingest_chunks([chunk])

        # Chunk embedding should be stored
        assert chunk_embedding_store.count() == 1

    @pytest.mark.mock_integration
    @patch("isotope.providers.litellm.client.litellm.embedding")
    @patch("isotope.providers.litellm.client.litellm.acompletion")
    def test_ingest_works_without_chunk_embedding_store(self, mock_acompletion, mock_embedding, stores):
        """Ingestor works without chunk_embedding_store (backward compat)."""
        mock_acompletion.return_value = MagicMock(
            choices=[Choices(finish_reason="stop", index=0, message=Message(role="assistant", content='["Q1?"]'))]
        )
        mock_embedding.return_value = MagicMock(data=[{"embedding": [0.1, 0.2, 0.3], "index": 0}])

        ingestor = Ingestor(
            embedded_question_store=stores["embedded_question_store"],
            chunk_store=stores["chunk_store"],
            atom_store=stores["atom_store"],
            atomizer=SentenceAtomizer(),
            embedder=ClientEmbedder(embedding_client=LiteLLMEmbeddingClient()),
            question_generator=ClientQuestionGenerator(llm_client=LiteLLMClient()),
            # No chunk_embedding_store
        )

        chunk = Chunk(content="Python is great.", source="test.md")
        result = ingestor.ingest_chunks([chunk])
        assert result["chunks"] == 1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ingestor.py::TestIngestorChunkEmbedding -v`
Expected: FAIL — `Ingestor` doesn't accept `chunk_embedding_store`

**Step 3: Implement chunk embedding in Ingestor**

Modify `src/isotope/ingestor.py`:

- Add `chunk_embedding_store: ChunkEmbeddingStore | None = None` to `__init__`
- In `_store_and_atomize`, after storing chunks, embed and store chunk embeddings if `chunk_embedding_store` is set

Add after `self.chunk_store.put_many(chunks)`:

```python
# Embed chunks and store in chunk embedding store
if self.chunk_embedding_store is not None:
    chunk_texts = [chunk.content for chunk in chunks]
    chunk_embeddings = self.embedder.embed_texts(chunk_texts)
    self.chunk_embedding_store.add(
        chunk_ids=[chunk.id for chunk in chunks],
        embeddings=chunk_embeddings,
    )
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ingestor.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/isotope/ingestor.py tests/test_ingestor.py
git commit -m "feat: embed chunks during ingestion for hybrid retrieval"
```

---

### Task 8: Wire through Isotope facade and LocalStorage

**Files:**
- Modify: `src/isotope/isotope.py`
- Modify: `src/isotope/configuration/base.py`
- Modify: `src/isotope/configuration/storage/local.py`
- Modify: `tests/test_isotope.py`
- Modify: `tests/conftest.py`

**Step 1: Write the failing tests**

Add to `tests/test_isotope.py`:

```python
class TestIsotopeHybridRetrieval:
    def test_retriever_gets_chunk_embedding_store(self, temp_dir, mock_provider):
        """Retriever receives chunk_embedding_store from Isotope."""
        iso = Isotope(provider=mock_provider, storage=LocalStorage(temp_dir))
        retriever = iso.retriever()
        assert retriever.chunk_embedding_store is not None

    def test_retriever_gets_hybrid_threshold_from_settings(self, temp_dir, mock_provider):
        """Retriever uses hybrid_confidence_threshold from Settings."""
        from isotope.settings import Settings
        iso = Isotope(
            provider=mock_provider,
            storage=LocalStorage(temp_dir),
            settings=Settings(hybrid_confidence_threshold=0.5),
        )
        retriever = iso.retriever()
        assert retriever.hybrid_confidence_threshold == 0.5

    def test_retriever_hybrid_threshold_override(self, temp_dir, mock_provider):
        """Retriever accepts hybrid_confidence_threshold override."""
        iso = Isotope(provider=mock_provider, storage=LocalStorage(temp_dir))
        retriever = iso.retriever(hybrid_confidence_threshold=0.3)
        assert retriever.hybrid_confidence_threshold == 0.3

    def test_ingestor_gets_chunk_embedding_store(self, temp_dir, mock_provider):
        """Ingestor receives chunk_embedding_store from Isotope."""
        iso = Isotope(provider=mock_provider, storage=LocalStorage(temp_dir))
        ingestor = iso.ingestor()
        assert ingestor.chunk_embedding_store is not None
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_isotope.py::TestIsotopeHybridRetrieval -v`
Expected: FAIL

**Step 3: Update LocalStorage**

In `src/isotope/configuration/storage/local.py`:
- Update `build_stores()` return type to include `ChunkEmbeddingStore`
- Create a `ChromaChunkEmbeddingStore` in the same chroma directory

In `src/isotope/configuration/base.py`:
- Update `StorageConfig.build_stores()` return type to 5-tuple

In `src/isotope/isotope.py`:
- Add `chunk_embedding_store` to `__init__`, `from_stores`, `retriever()`, `ingestor()`
- Wire `hybrid_confidence_threshold` through to `retriever()`
- Update `delete_source` and `_prepare_ingest_file` to clean up chunk embeddings

**Step 4: Update conftest.py**

Add `chunk_embedding_store` to the `stores` fixture:

```python
from isotope.stores.chroma import ChromaChunkEmbeddingStore

# In stores fixture, add:
chunk_embedding_store = ChromaChunkEmbeddingStore(os.path.join(temp_dir, "chunk_embeddings"))
stores_dict["chunk_embedding_store"] = chunk_embedding_store
# In cleanup, also close chunk_embedding_store
```

**Step 5: Run tests**

Run: `pytest tests/test_isotope.py -v`
Expected: ALL PASS

**Step 6: Run full test suite**

Run: `pytest -v`
Expected: ALL PASS

**Step 7: Commit**

```bash
git add src/isotope/isotope.py src/isotope/configuration/base.py src/isotope/configuration/storage/local.py tests/test_isotope.py tests/conftest.py
git commit -m "feat: wire hybrid retrieval through Isotope facade and LocalStorage"
```

---

### Task 9: Final verification

**Step 1: Run linting and type checking**

```bash
ruff format src tests
ruff check --fix src tests
mypy src
```

**Step 2: Run full test suite**

```bash
pytest -v
```

**Step 3: Fix any issues found**

**Step 4: Final commit if any fixes were needed**

```bash
git add -A
git commit -m "fix: address lint and type check issues"
```

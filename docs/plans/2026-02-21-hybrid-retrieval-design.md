# Hybrid Retrieval Mode

**Date:** 2026-02-21
**Status:** Approved

## Problem

Isotope's question-matching retrieval produces high-precision results when user queries match anticipated questions, but returns low scores (or nothing) for unanticipated queries. The README mentions hybrid mode as a mitigation strategy but it's not implemented.

## Solution

Add a chunk-level vector search fallback to the retrieval pipeline. When the best question-match score falls below a confidence threshold, the retriever also searches chunk embeddings directly and appends those results.

## Key Decisions

### Confidence Threshold
- **Type:** `float` (not optional)
- **Default:** `0.7`
- **`0`** = disabled (pure question-matching, current behavior)
- **`1.0`** = always include chunk fallback
- Configurable in `Settings.hybrid_confidence_threshold` and overridable at query time

### Merge Strategy
- Question-match results come first (higher signal)
- Chunk-fallback results fill remaining `k` slots
- Deduplicate by `chunk_id` — same chunk found by both paths appears only once
- If question matching returns results above threshold, those fill slots first; chunk fallback fills the rest up to `k`

### Storage
- New ABC: `ChunkEmbeddingStore` in `stores/base.py`
- New implementation: `ChromaChunkEmbeddingStore` — second ChromaDB collection `"isotope_chunks"`
- Existing question collection renamed from `"isotope"` to `"isotope_questions"`
- No backward compatibility migration needed
- Chunk content stays in SQLite; ChromaDB only stores embeddings + chunk_id metadata

### SearchResult Model
- `question: Question` becomes `question: Question | None`
- `atom: Atom` becomes `atom: Atom | None`
- Chunk-fallback results have `question=None, atom=None`

## Architecture

### Query-Time Flow

```
User query
    |
    v
Embed query (existing Embedder)
    |
    +--> Search question embeddings (existing)
    |        |
    |        v
    |    Question results with scores
    |        |
    |        v
    |    Best score < threshold? ----no----> Return question results
    |        |
    |       yes
    |        |
    |        v
    +--> Search chunk embeddings (new)
             |
             v
         Deduplicate by chunk_id
             |
             v
         Append chunk results after question results
             |
             v
         Cap at k, return
```

### Ingestion-Time Flow

Existing pipeline + one new step:

1. Store chunks in ChunkStore (existing)
2. **Embed chunks and store in ChunkEmbeddingStore (new)**
3. Atomize chunks (existing)
4. Generate questions (existing)
5. Embed questions and store in EmbeddedQuestionStore (existing)

### New ABC: ChunkEmbeddingStore

```python
class ChunkEmbeddingStore(ABC):
    def add(self, chunk_ids: list[str], embeddings: list[list[float]]) -> None: ...
    def search(self, embedding: list[float], k: int = 5) -> list[tuple[str, float]]: ...
    def delete_by_chunk_ids(self, chunk_ids: list[str]) -> None: ...
    def count(self) -> int: ...
```

## Components Modified

| Component | Change |
|-----------|--------|
| `Settings` | Add `hybrid_confidence_threshold: float = 0.7` |
| `stores/base.py` | Add `ChunkEmbeddingStore` ABC |
| `stores/chroma.py` | Add `ChromaChunkEmbeddingStore`; rename question collection to `"isotope_questions"` |
| `models/results.py` | Make `SearchResult.question` and `SearchResult.atom` optional |
| `retriever.py` | Add hybrid fallback logic in `get_context()` |
| `ingestor.py` | Embed chunks during ingestion, store in `ChunkEmbeddingStore` |
| `isotope.py` | Wire `chunk_embedding_store` through facade; pass threshold to retriever |
| `configuration/storage/local.py` | `build_stores()` returns `ChunkEmbeddingStore` as 5th store |
| `stores/__init__.py` | Export `ChunkEmbeddingStore` |
| `__init__.py` | Export `ChromaChunkEmbeddingStore` |

## Testing

- Retriever with hybrid disabled (threshold `0`) — current behavior unchanged
- Retriever with hybrid enabled, high question scores — no fallback triggered
- Retriever with hybrid enabled, low question scores — fallback runs, chunk results appended
- Retriever with no question results — fallback fills all `k` slots
- Dedup: same chunk from both paths appears once
- `k` budget respected: question + chunk results don't exceed `k`
- `ChromaChunkEmbeddingStore`: add/search round-trip, delete, count
- Ingestor with and without `chunk_embedding_store` (no regression)

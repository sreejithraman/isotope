# Isotope

Reverse RAG database - index questions, not chunks.

## How It Works

Traditional RAG indexes document chunks and hopes user queries semantically match those chunks. Isotope flips this:

1. **Ingest** → Break documents into chunks → Atomize into facts → Generate questions each fact answers
2. **Index** → Embed and store the *questions*, not the chunks
3. **Query** → User's question matches against pre-generated questions → Return the chunks that answer those questions

The insight: question-to-question similarity is tighter than question-to-chunk similarity.

## Limitations & Trade-offs

### The Coverage Gap

Isotope trades recall for precision. If a user asks a question you didn't anticipate generating, retrieval quality degrades.

**Example:**
- Chunk: "Python was created by Guido van Rossum in 1991 at CWI in the Netherlands."
- Generated questions: "Who created Python?", "When was Python created?", "Where was Python created?"
- User asks: "What programming languages were invented in the Netherlands?"
- Result: Low similarity scores. The chunk *might* surface, but with low confidence.

**What happens at query time:**
- Queries always return results with similarity scores
- High scores (>0.8): confident match
- Medium scores (0.5-0.8): uncertain, might be relevant
- Low scores (<0.5): likely no good match exists

The system doesn't "fail" - it returns best-effort matches. The scores encode confidence.

### When This Approach Works Well

- Question space is somewhat predictable
- High precision matters more than exhaustive recall
- Documents have clear, factual content that maps to natural questions

### When Traditional RAG May Be Better

- Exploratory queries with unpredictable phrasing
- Content that doesn't decompose into question-answerable facts
- Recall is critical (can't afford to miss relevant content)

### Mitigation Strategies

For production use, consider:
- **Hybrid retrieval**: Fall back to chunk embedding search when question scores are low
- **Query expansion**: Use LLM to rephrase user query in multiple ways before search
- **Lower thresholds + re-ranking**: Accept more candidates, then re-rank with a cross-encoder

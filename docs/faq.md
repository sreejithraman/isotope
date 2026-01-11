# Frequently Asked Questions

## General

### What is Reverse RAG?

Traditional RAG embeds document chunks and hopes user queries match them semantically. **Reverse RAG** flips this: we pre-generate questions each chunk can answer, embed those questions, and match user queries to questions. This gives tighter semantic alignment because we're matching question-to-question rather than query-to-chunk.

Based on the paper [arXiv:2405.12363](https://arxiv.org/abs/2405.12363).

### When should I use Isotope vs other RAG tools?

**Use Isotope when:**
- Your content has predictable questions (FAQ-style, technical docs, support content)
- You need high precision retrieval
- Meaningful confidence scores matter
- You can afford the upfront ingestion cost

**Use traditional RAG when:**
- Queries are exploratory or open-ended
- Content is narrative/creative
- You need maximum recall over precision
- Low-latency ingestion is critical

### Is Isotope production-ready?

Isotope is in early release (v0.1.x). It's suitable for:
- Prototypes and MVPs
- Internal tools
- Small-to-medium knowledge bases

For large-scale production, consider the [deployment guide](./guides/deployment.md) and test thoroughly.

### What's the minimum viable setup?

```bash
pip install "isotope-rag[cli,litellm,chroma]"
export OPENAI_API_KEY=sk-...
isotope init
isotope ingest ./your-docs/
isotope query "How does X work?"
```

---

## Setup & Installation

### Which Python version do I need?

Python 3.11 or newer. Check with `python --version`.

### What LLM provider should I start with?

- **Free:** Gemini (Google AI Studio) or Ollama (local)
- **Paid:** OpenAI GPT-5-mini offers good quality/cost ratio
- **Enterprise:** Anthropic Claude or Azure OpenAI

### Can I use multiple providers?

Yes, but not simultaneously in one Isotope instance. You can:
- Use different providers for different projects (separate `isotope.yaml`)
- Switch providers by updating your config

### How do I use a local model?

Install [Ollama](https://ollama.com), then:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text

isotope init --llm-model ollama/llama3.2 --embedding-model ollama/nomic-embed-text
```

---

## Usage

### How many questions should I generate per atom?

The default is 5 questions per atom. Adjust based on your needs:

| Setting | Effect |
|---------|--------|
| 3 | Faster, cheaper, lower recall |
| 5 | Balanced (default) |
| 10 | Higher recall, more expensive |

```yaml
settings:
  questions_per_atom: 5
```

### When should I use LLMAtomizer vs SentenceAtomizer?

| Atomizer | Best For | Trade-off |
|----------|----------|-----------|
| **SentenceAtomizer** | Speed, cost-sensitive | Faster, may miss context |
| **LLMAtomizer** | Quality, complex docs | Slower, better semantic units |

```yaml
settings:
  use_sentence_atomizer: true   # Fast
  use_sentence_atomizer: false  # Quality (default)
```

### How do I update indexed documents?

Isotope tracks content hashes. Just re-run ingest:

```bash
isotope ingest ./your-docs/
```

Unchanged files are skipped. Changed files are re-indexed.

To force re-indexing:
```bash
isotope delete path/to/file.md
isotope ingest path/to/file.md
```

### Can I delete specific sources?

Yes:

```bash
# List sources
isotope list

# Delete one
isotope delete /path/to/file.md
```

In TUI, add `--force`:
```
> delete /path/to/file.md --force
```

---

## Retrieval

### Why am I getting low scores?

Low confidence scores (< 0.7) indicate weak semantic matches. Try:

1. **Rephrase as a question:** "authentication" → "How does authentication work?"
2. **Be more specific:** Include relevant terms from your docs
3. **Check matched questions:**
   ```bash
   isotope query "How do I authenticate?" -q
   ```

### What's a good confidence threshold?

| Score | Interpretation |
|-------|----------------|
| > 0.9 | High confidence match |
| 0.7-0.9 | Good match |
| 0.5-0.7 | Weak match, review results |
| < 0.5 | Poor match |

### How do I improve retrieval quality?

1. **Increase questions per atom** for more entry points
2. **Use LLMAtomizer** for better semantic units
3. **Enable diversity filtering** to reduce redundant questions
4. **Review raw results** to understand what's being matched:
   ```bash
   isotope query "..." --raw
   ```

### Can I disable answer synthesis?

Yes, use `--raw` to get just the matched chunks:

```bash
isotope query "How does auth work?" --raw
```

---

## Performance

### Why is ingestion slow?

Isotope generates multiple LLM calls per chunk:
1. Atomization (if using LLMAtomizer)
2. Question generation (N questions per atom)
3. Embedding (for all questions)

**Speed up with:**
```yaml
settings:
  use_sentence_atomizer: true      # Skip LLM atomization
  questions_per_atom: 3            # Fewer questions
  max_concurrent_llm_calls: 10     # More parallelism (if not rate-limited)
```

### How much storage do I need?

Rough estimates per 1,000 chunks:
- SQLite metadata: ~1-5 MB
- Chroma embeddings: ~50-100 MB (depends on embedding dimension)

### What's the maximum corpus size?

Tested with:
- ~10,000 documents
- ~100,000 chunks
- ~500,000 questions

For larger corpora, consider external vector stores (Pinecone, Weaviate) instead of local Chroma.

### How do I handle rate limits?

Use conservative settings:

```yaml
settings:
  max_concurrent_llm_calls: 2
```

Or use a rate-limit profile:
```python
settings = Settings.with_profile("conservative")
```

---

## Troubleshooting

### My API key isn't being recognized

Check environment variable naming:
```bash
# OpenAI
export OPENAI_API_KEY=sk-...

# Anthropic
export ANTHROPIC_API_KEY=sk-ant-...

# Gemini
export GEMINI_API_KEY=...

# Generic (works with any provider)
export ISOTOPE_LLM_API_KEY=...
```

### Ingestion failed halfway through

Isotope tracks which files have been ingested. Re-run:
```bash
isotope ingest ./your-docs/
```

Successfully ingested files will be skipped.

### Results aren't matching my expectations

1. **Check what's indexed:** `isotope status`
2. **Review raw results:** `isotope query "..." --raw`
3. **See matched questions:** `isotope query "..." -q`
4. **Try different phrasing:** Questions work better than keywords

For more issues, see the [Troubleshooting Guide](./guides/troubleshooting.md).

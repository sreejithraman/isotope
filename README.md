# Isotope

**Your RAG is searching for answers. It should be matching questions.**

Traditional RAG embeds your documents and hopes user questions land nearby. But questions and statements live in different semantic spaces—you're matching apples to oranges.

Isotope breaks your documents into **atomic facts**, generates the **questions each fact answers**, then indexes *those questions*. When users ask something, they match question-to-question. Same semantic space. Tighter matches. Better retrieval.

## Why "Isotope"?

In chemistry, isotopes are variants of the same element—same core identity, different configurations. Isotope does the same for your knowledge: it takes each atomic fact and generates multiple question "isotopes"—different phrasings that all point back to the same truth.

One fact. Many questions. All paths lead to the right answer.

## The 30-Second Pitch

```
Traditional RAG:  "Who created Python?" → search chunks → hope for the best
                   ↓
                   Semantic gap between questions and statements

Isotope:          "Who created Python?" → search questions → get "Who created Python?"
                   ↓
                   Same semantic space = confident matches
```

## Installation

```bash
pip install isotopedb[local]
```

This installs Isotope with ChromaDB (local vector store), LiteLLM, and CLI tools—everything needed for local development.

## Quick Start

```python
from isotopedb import Isotope, Chunk

# Quick setup with LiteLLM
iso = Isotope.with_litellm(
    llm_model="openai/gpt-4o",
    embedding_model="openai/text-embedding-3-small",
    data_dir="./my_data",
)

# Ingest
ingestor = iso.ingestor()
chunks = [Chunk(content="Python was created by Guido van Rossum in 1991.", source="wiki")]
ingestor.ingest_chunks(chunks)

# Query (pass llm_model to enable synthesis)
retriever = iso.retriever(llm_model="openai/gpt-4o")
response = retriever.get_answer("Who invented Python?")
print(response.answer)  # LLM-synthesized answer
print(response.results)  # Source chunks with confidence scores
```

Or use the CLI:

```bash
# Ingest your docs
isotope ingest docs/

# Ask questions
isotope query "How do I authenticate?"

# See what's indexed
isotope status
```

## How It Works

```
┌──────────┐    ┌────────┐    ┌───────┐    ┌───────────┐    ┌─────────┐
│ Document │───▶│ Chunks │───▶│ Atoms │───▶│ Questions │───▶│  Index  │
└──────────┘    └────────┘    └───────┘    └───────────┘    └─────────┘
                                                                  │
┌──────────┐    ┌────────┐    ┌───────────────────────────────────┘
│  Answer  │◀───│ Chunks │◀───│ Match user query against questions
└──────────┘    └────────┘
```

1. **Atomize** → Break content into atomic facts
2. **Generate** → Create questions each fact answers (15 per atom by default)
3. **Embed & Index** → Store question embeddings
4. **Query** → User questions match against indexed questions
5. **Retrieve** → Return the chunks that answer matched questions

Based on ["Question-Based Retrieval using Atomic Units for Enterprise RAG"](https://arxiv.org/abs/2405.12363)

## Documentation

- **Concepts**
  - [Reverse RAG Explained](docs/concepts/reverse-rag.md) - The paper and core insight
  - [Architecture](docs/concepts/architecture.md) - System design and components
- **Guides**
  - [Configuration](docs/guides/configuration.md) - Settings and environment variables
  - [Atomization](docs/guides/atomization.md) - Sentence vs LLM atomization
  - [CLI Reference](docs/guides/cli.md) - Command-line usage
- **Tutorials**
  - [Getting Started](docs/tutorials/getting-started.md) - Your first 10 minutes

## When to Use Isotope

**Great fit:**
- FAQ-style content where questions are predictable
- Technical docs, knowledge bases, support content
- When precision matters more than recall
- When you want confidence scores you can trust

**Consider traditional RAG when:**
- Queries are exploratory and unpredictable
- You can't afford to miss any relevant content
- Content doesn't map naturally to Q&A format

## Trade-offs

Isotope trades recall for precision. If users ask questions you didn't anticipate, scores will be low. But you'll *know* they're low—the confidence scores are meaningful.

**Mitigation strategies:**
- Hybrid retrieval (fall back to chunk search for low scores)
- Query expansion (rephrase queries before search)
- Re-ranking with cross-encoders

See [Limitations](docs/concepts/reverse-rag.md#the-trade-off) for details.

## License

MIT

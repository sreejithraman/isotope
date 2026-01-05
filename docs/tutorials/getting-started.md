# Getting Started with Isotope

This tutorial walks you through ingesting your first document and querying it with Isotope. You'll be up and running in under 10 minutes.

## Prerequisites

- Python 3.11+
- An API key for your LLM provider (Gemini, OpenAI, Anthropic, etc.)

## Installation

```bash
pip install isotope-rag[all]
```

## Set Up Your API Key

Isotope uses LiteLLM under the hood, so it works with any major LLM provider. Set the API key for your provider:

```bash
# For Gemini (default)
export GOOGLE_API_KEY="your-api-key"

# For OpenAI
export OPENAI_API_KEY="your-api-key"

# For Anthropic
export ANTHROPIC_API_KEY="your-api-key"
```

See [Configuration Guide](../guides/configuration.md) for all supported providers.

## The Simple Way: LiteLLMProvider + LocalStorage

The simplest way to get started is to pair `LiteLLMProvider` with `LocalStorage`. This wires
up local stores and LiteLLM-backed components for you.

### Step 1: Create Some Content

Create a file called `python-intro.txt`:

```text
Python is a high-level programming language created by Guido van Rossum.
It was first released in 1991. Python emphasizes code readability and
uses significant indentation. It supports multiple programming paradigms
including procedural, object-oriented, and functional programming.
```

### Step 2: Ingest It

```python
from isotope import Isotope, Chunk, LiteLLMProvider, LocalStorage

# Create an Isotope instance (LiteLLM + local stores)
iso = Isotope(
    provider=LiteLLMProvider(
        llm="openai/gpt-4o",
        embedding="openai/text-embedding-3-small",
        atomizer_type="sentence",  # Faster, no LLM for atomization
    ),
    storage=LocalStorage("./my_isotope_data"),
)

# Load your content
with open("python-intro.txt") as f:
    content = f.read()

chunk = Chunk(content=content, source="python-intro.txt")

# Ingest it
ingestor = iso.ingestor()
result = ingestor.ingest_chunks([chunk])

print(f"Ingested {result['chunks']} chunks")
print(f"Created {result['atoms']} atoms")
print(f"Generated {result['questions']} questions")
```

That's it! Isotope automatically:
- Broke your content into atomic facts (sentences)
- Generated ~15 questions for each fact
- Embedded and indexed those questions
- Stored everything for retrieval

### Step 3: Query It

```python
# Create a retriever (pass llm_model to enable synthesis)
retriever = iso.retriever(llm_model="openai/gpt-4o")

# Ask a question
response = retriever.get_answer("Who invented Python?")

# Get the LLM-synthesized answer
print(f"Answer: {response.answer}")

# See the source chunks with scores
for result in response.results:
    print(f"  [{result.score:.2f}] {result.chunk.content[:80]}...")
```

Output:
```
Answer: Python was created by Guido van Rossum. It was first released in 1991.

  [0.92] Python is a high-level programming language created by Guido van Rossum...
```

### Step 4: Query Without LLM Synthesis (Optional)

If you just want the raw chunks without an LLM-generated answer, use `get_context()`:

```python
results = retriever.get_context("What paradigms does Python support?")

# Just the matching results, no LLM synthesis
for result in results:
    print(f"  [{result.score:.2f}] {result.question.text}")
    print(f"           → {result.chunk.content[:60]}...")
```

## Using the CLI

Isotope also has a command-line interface for quick workflows:

```bash
# One-time: create CLI config (or set ISOTOPE_LITELLM_* env vars)
isotope init --provider litellm --llm-model openai/gpt-4o --embedding-model openai/text-embedding-3-small

# Ingest a file or directory
isotope ingest python-intro.txt

# Query the database
isotope query "Who created Python?"

# Query without LLM synthesis
isotope query "Who created Python?" --raw

# See what's indexed
isotope status

# List all sources
isotope list

# Remove a source
isotope delete python-intro.txt
```

If you used a custom data directory above, add `--data-dir ./my_isotope_data` to the CLI commands
so they operate on the same database.

See [CLI Reference](../guides/cli.md) for all commands.

## Complete Example

Here's everything in one script:

```python
from isotope import Isotope, Chunk, LiteLLMProvider, LocalStorage

# 1. Setup
iso = Isotope(
    provider=LiteLLMProvider(
        llm="openai/gpt-4o",
        embedding="openai/text-embedding-3-small",
        atomizer_type="sentence",
    ),
    storage=LocalStorage("./my_data"),
)

# 2. Load and ingest content
with open("python-intro.txt") as f:
    content = f.read()

chunk = Chunk(content=content, source="python-intro.txt")

ingestor = iso.ingestor()
result = ingestor.ingest_chunks([chunk])
print(f"Indexed {result['questions']} questions from {result['atoms']} atoms")

# 3. Query
retriever = iso.retriever(llm_model="openai/gpt-4o")
response = retriever.get_answer("Who created Python?")
print(f"\nAnswer: {response.answer}")
```

## What Just Happened?

Under the hood, Isotope:

1. **Atomized** your content into individual facts (sentences)
2. **Generated questions** for each fact (~15 per atom by default)
3. **Embedded** those questions using your configured embedding model
4. **Filtered** near-duplicate questions (85% similarity threshold)
5. **Indexed** everything in a vector store (ChromaDB by default)

When you queried:

1. Your question was **embedded** using the same model
2. It was **matched** against the indexed questions (question-to-question!)
3. The matching chunks were **retrieved**
4. An LLM **synthesized** an answer from those chunks

This is the "Reverse RAG" approach—see [Reverse RAG Explained](../concepts/reverse-rag.md) for the theory.

## Customizing the Pipeline

The `Isotope` class uses sensible defaults, but you can customize everything:

```python
from isotope import Isotope, LiteLLMProvider, LocalStorage

iso = Isotope(
    provider=LiteLLMProvider(
        llm="openai/gpt-4o",                        # LLM for generation/synthesis
        embedding="openai/text-embedding-3-small",  # Embedding model
    ),
    storage=LocalStorage("./my_data"),
)

# Or customize the ingestor
ingestor = iso.ingestor(
    use_diversity_filter=False,  # Keep all questions
)

# Or customize the retriever
retriever = iso.retriever(
    llm_model=None,  # Disable synthesis entirely
    default_k=10,    # Return more results
)
```

See [Configuration Guide](../guides/configuration.md) for all options.

## Going Deeper

Want to understand or customize individual components?

<details>
<summary>Click to see the manual pipeline approach</summary>

If you need full control, you can use the components directly:

```python
from isotope import (
    Chunk,
    SentenceAtomizer,
    DiversityFilter,
    ChromaEmbeddedQuestionStore,
    SQLiteChunkStore,
    SQLiteAtomStore,
)
from isotope.embedder import ClientEmbedder
from isotope.providers.litellm import LiteLLMClient, LiteLLMEmbeddingClient
from isotope.question_generator import ClientQuestionGenerator

# Create stores
embedded_question_store = ChromaEmbeddedQuestionStore("./data/chroma")
chunk_store = SQLiteChunkStore("./data/chunks.db")
atom_store = SQLiteAtomStore("./data/atoms.db")

# Create components
atomizer = SentenceAtomizer()
embedding_client = LiteLLMEmbeddingClient(model="gemini/text-embedding-004")
llm_client = LiteLLMClient(model="gemini/gemini-3-flash-preview")
embedder = ClientEmbedder(embedding_client=embedding_client)
question_generator = ClientQuestionGenerator(llm_client=llm_client, num_questions=10)
diversity_filter = DiversityFilter(threshold=0.85)

# Load content
chunk = Chunk(content="Python was created by Guido van Rossum.", source="wiki")

# Atomize
atoms = atomizer.atomize(chunk)

# Generate questions (one atom at a time, or use async for speed)
questions = []
for atom in atoms:
    questions.extend(question_generator.generate(atom, chunk.content))

# Embed
embedded = embedder.embed_questions(questions)

# Filter
filtered = diversity_filter.filter(embedded)

# Store
chunk_store.put(chunk)
atom_store.put_many(atoms)
embedded_question_store.add(filtered)

# Query
query_embedding = embedder.embed_text("Who made Python?")
results = embedded_question_store.search(query_embedding, k=3)

for question, score in results:
    chunk = chunk_store.get(question.chunk_id)
    print(f"[{score:.2f}] {chunk.content}")
```

**Tip: Async for Large Documents**

For large documents with many atoms, use async methods to parallelize question generation:

```python
import asyncio

# Using the high-level API
async def ingest_large_file():
    result = await iso.aingest_file("large-document.pdf")
    return result

result = asyncio.run(ingest_large_file())

# Or with the Ingestor directly
ingestor = iso.ingestor(max_concurrent_questions=20)
result = asyncio.run(ingestor.aingest_chunks(chunks))
```

This can be 10-50x faster for documents with 100+ atoms.

</details>

## Next Steps

- [Configuration Guide](../guides/configuration.md) - All settings and environment variables
- [Atomization Guide](../guides/atomization.md) - Sentence vs LLM atomization strategies
- [CLI Reference](../guides/cli.md) - Command-line interface
- [Architecture](../concepts/architecture.md) - How the components fit together
- [Reverse RAG Explained](../concepts/reverse-rag.md) - The theory behind Isotope

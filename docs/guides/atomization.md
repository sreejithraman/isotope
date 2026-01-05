# Atomization Guide

Atomization is the process of breaking document chunks into **atomic facts** - single, self-contained statements that can each be matched to questions.

Isotope provides two atomization strategies:

## Strategy Comparison

| Aspect | SentenceAtomizer | LLMAtomizer |
|--------|------------------|-------------|
| **Method** | Split by sentence boundaries | LLM extracts facts |
| **Speed** | Fast (local processing) | Slow (API calls) |
| **Cost** | Free | LLM API costs |
| **Quality** | Structural | Semantic |
| **Best for** | Well-structured docs | Complex, dense content |

## SentenceAtomizer

Uses [pysbd](https://github.com/nipunsadvilkar/pySBD) for robust sentence boundary detection.

```python
from isotope import SentenceAtomizer, Chunk

atomizer = SentenceAtomizer(min_length=10, language="en")

chunk = Chunk(
    content="Python was created by Guido. It was released in 1991.",
    source="wiki"
)

atoms = atomizer.atomize(chunk)
# [Atom(content="Python was created by Guido.", ...),
#  Atom(content="It was released in 1991.", ...)]
```

### Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_length` | `10` | Minimum characters for an atom |
| `language` | `"en"` | Language for sentence detection |

### When to Use

- Documents with clear sentence structure
- Speed matters more than semantic precision
- Cost-sensitive applications
- Large document volumes

### Limitations

- Doesn't resolve coreferences ("It" → "Python")
- May split complex sentences poorly
- Treats each sentence as equal importance

## LLMAtomizer

Uses an LLM to extract atomic facts from chunks.

```python
from isotope.atomizer import LLMAtomizer
from isotope.providers.litellm import LiteLLMClient
from isotope import Chunk

client = LiteLLMClient(model="openai/gpt-4o")
atomizer = LLMAtomizer(llm_client=client)

chunk = Chunk(
    content="Python, created by Guido van Rossum in 1991, is known for readability.",
    source="wiki"
)

atoms = atomizer.atomize(chunk)
# [Atom(content="Python was created by Guido van Rossum.", ...),
#  Atom(content="Python was created in 1991.", ...),
#  Atom(content="Python is known for readability.", ...)]
```

### Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `llm_client` | (required) | Any `LLMClient` implementation |
| `prompt_template` | (default) | Custom extraction prompt |
| `temperature` | `0.0` | LLM temperature |

### When to Use

- Dense, complex content
- Quality matters more than speed
- Documents with implicit information
- Small document volumes (cost acceptable)

### Benefits

- Resolves coreferences
- Extracts implicit facts
- Normalizes statement structure
- Better semantic granularity

## Decision Tree

```
                    Start
                      │
                      ▼
              Is speed critical?
                 /         \
               Yes          No
                │            │
                ▼            ▼
        SentenceAtomizer   Is cost a concern?
                              /         \
                            Yes          No
                             │            │
                             ▼            ▼
                   SentenceAtomizer   LLMAtomizer
```

**Quick rules:**
- Prefer `SentenceAtomizer` when speed/cost matters
- Use `LLMAtomizer` when semantic quality matters more than cost

## Configuration

Configure atomization explicitly in code or via the CLI config file:

```python
from isotope import Isotope, LiteLLMProvider, LocalStorage

iso = Isotope(
    provider=LiteLLMProvider(
        llm="openai/gpt-4o",
        embedding="openai/text-embedding-3-small",
        atomizer_type="sentence",  # Use sentence-based atomizer
    ),
    storage=LocalStorage("./isotope_data"),
)
```

```yaml
# isotope.yaml
provider: litellm
llm_model: openai/gpt-4o
embedding_model: openai/text-embedding-3-small
use_sentence_atomizer: true
```

## Custom Atomizer

Implement the `Atomizer` ABC to create custom strategies:

```python
from isotope import Atomizer, Atom, Chunk

class CustomAtomizer(Atomizer):
    def atomize(self, chunk: Chunk) -> list[Atom]:
        # Your custom logic
        atoms = []

        # Example: split by paragraphs
        for i, para in enumerate(chunk.content.split("\n\n")):
            if para.strip():
                atoms.append(Atom(
                    content=para.strip(),
                    chunk_id=chunk.id,
                    index=i,
                ))

        return atoms
```

### ABC Contract

```python
class Atomizer(ABC):
    @abstractmethod
    def atomize(self, chunk: Chunk) -> list[Atom]:
        """Break a chunk into atomic facts."""
        ...
```

Requirements:
- Each `Atom` must have `chunk_id` set to the source chunk's ID
- Each `Atom` should have a unique `index` (position within chunk)
- Empty chunks should return empty list

## Hybrid Approach

You can combine strategies by routing chunks to the appropriate atomizer.

### Structure-based routing

Detect formatting signals to identify well-structured content:

```python
from isotope import SentenceAtomizer, Chunk, Atom
from isotope.atomizer import LLMAtomizer
from isotope.providers.litellm import LiteLLMClient
import re

sentence_atomizer = SentenceAtomizer()
llm_client = LiteLLMClient(model="openai/gpt-4o")
llm_atomizer = LLMAtomizer(llm_client=llm_client)

def smart_atomize(chunk: Chunk) -> list[Atom]:
    """Route to appropriate atomizer based on content structure."""

    # Detect well-structured content signals
    has_markdown_lists = bool(re.search(r'^[\s]*[-*]\s', chunk.content, re.MULTILINE))
    has_numbered_lists = bool(re.search(r'^[\s]*\d+\.\s', chunk.content, re.MULTILINE))
    has_headers = bool(re.search(r'^#+\s', chunk.content, re.MULTILINE))

    # Well-structured docs: clear formatting → fast sentence atomizer
    if has_markdown_lists or has_numbered_lists or has_headers:
        return sentence_atomizer.atomize(chunk)

    # Dense prose without structure → LLM for semantic extraction
    return llm_atomizer.atomize(chunk)
```

### Source-based routing

Simpler approach using document source metadata:

```python
from isotope import SentenceAtomizer, Chunk, Atom
from isotope.atomizer import LLMAtomizer
from isotope.providers.litellm import LiteLLMClient

sentence_atomizer = SentenceAtomizer()
llm_client = LiteLLMClient(model="openai/gpt-4o")
llm_atomizer = LLMAtomizer(llm_client=llm_client)

def smart_atomize(chunk: Chunk) -> list[Atom]:
    """Route based on document source type."""

    # Sources known to be well-structured
    structured_sources = {"api_docs", "faq", "changelog", "reference"}

    if chunk.source in structured_sources:
        return sentence_atomizer.atomize(chunk)

    # Prose, articles, technical content → LLM
    return llm_atomizer.atomize(chunk)
```

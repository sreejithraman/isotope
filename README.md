# Isotope

[![PyPI version](https://badge.fury.io/py/isotope-rag.svg)](https://pypi.org/project/isotope-rag/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2405.12363-b31b1b.svg)](https://arxiv.org/abs/2405.12363)

> ⚠️ **Alpha Release (v0.1.0)**: APIs are stabilizing but may change. Production use at your own risk.

**Your RAG is searching for answers. It should be matching questions.**

Traditional RAG embeds your documents and hopes user questions land nearby. But questions and statements live in different semantic spaces—you're matching apples to oranges.

Isotope breaks your documents into **atomic facts**, generates the **questions each fact answers**, then indexes *those questions*. When users ask something, they match question-to-question. Same semantic space. Tighter matches. Better retrieval.

## Why "Isotope"?

In chemistry, isotopes are variants of the same element—same core identity, different configurations. Isotope does the same for your knowledge: it takes each atomic fact and generates multiple question "isotopes"—different phrasings that all point back to the same truth.

One fact. Many questions. All paths lead to the right answer.

## Why Isotope?

- ✅ **Question-to-question matching** - Tighter semantic alignment than traditional RAG
- ✅ **Confidence scores you can trust** - Know when retrieval quality is low
- ✅ **Pluggable architecture** - Bring your own LLM provider, embeddings, vector store
- ✅ **CLI + Python API** - Use from command line or integrate into your app
- ✅ **Research-backed** - Implements peer-reviewed approach (arXiv:2405.12363)
- ✅ **Optional dependencies** - Install only what you need

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [Performance](#performance)
- [When to Use](#when-to-use-isotope)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

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

**Requirements:** Python 3.11+ and an LLM provider (OpenAI, Anthropic, etc. via LiteLLM)

**Quick start (recommended):**
```bash
pip install isotope-rag[all]
export OPENAI_API_KEY=your-key-here  # or other LiteLLM-compatible provider
```

**Minimal install:**
```bash
pip install isotope-rag  # Core only - bring your own provider/storage
pip install isotope-rag[litellm,chroma]  # Add LiteLLM + ChromaDB
```

**Optional extras:**
- `[all]` - Everything (recommended for new users)
- `[litellm]` - LiteLLM integration for 100+ LLM providers
- `[chroma]` - ChromaDB vector store
- `[cli]` - Command-line interface
- `[loaders]` - PDF/HTML document loaders
- `[dev]` - Development tools (pytest, ruff, mypy)

## Quick Start

### Option 1: Command Line (fastest)

```bash
# 0. Configure models (writes isotope.yaml)
isotope init --provider litellm --llm-model openai/gpt-4o --embedding-model openai/text-embedding-3-small

# 1. Ingest your docs
isotope ingest docs/

# 2. Ask questions
isotope query "How do I authenticate?"

# 3. See what's indexed
isotope status
```

**Expected output:**
```
📊 Indexed: 42 chunks → 156 atoms → 2,340 questions
🔍 Top result: "How to authenticate users" (confidence: 0.92)
📄 Source: docs/authentication.md
```

### Option 2: Python API (for integration)

```python
from isotope import Isotope, Chunk, LiteLLMProvider, LocalStorage

# Initialize
iso = Isotope(
    provider=LiteLLMProvider(
        llm="openai/gpt-4o",
        embedding="openai/text-embedding-3-small",
    ),
    storage=LocalStorage("./my_data"),
)

# Ingest
ingestor = iso.ingestor()
chunks = [Chunk(
    content="Python was created by Guido van Rossum in 1991.",
    source="wiki"
)]
ingestor.ingest_chunks(chunks)

# Query
retriever = iso.retriever(llm_model="openai/gpt-4o")
response = retriever.get_answer("Who invented Python?")

print(response.answer)   # "Python was created by Guido van Rossum."
print(response.results)  # [SearchResult(chunk=..., score=0.94)]
```

**What's happening here?**
1. Chunks are broken into atoms ("Python was created by Guido van Rossum" + "Python was created in 1991")
2. Questions are generated for each atom ("Who created Python?", "When was Python created?")
3. Questions are embedded and indexed in ChromaDB
4. Your query matches question-to-question
5. The LLM synthesizes an answer from matching chunks

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

## Performance

Based on [arXiv:2405.12363](https://arxiv.org/abs/2405.12363):

| Metric | Reverse RAG | Traditional RAG |
|--------|-------------|-----------------|
| Precision@5 | **Higher** | Baseline |
| MRR | **Higher** | Baseline |
| Score Calibration | **Meaningful** | Less calibrated |

**Key findings:**
- **Question diversity matters**: Generating diverse questions improves coverage
- **Deduplication helps**: 50% retention after deduplication maintains maximum retrieval performance
- **Confidence scores are meaningful**: Low scores genuinely indicate poor matches, unlike traditional RAG where scores can be misleadingly high

*See the paper for full benchmarks on MS MARCO and other datasets.*

## Documentation

**📚 Learn the Concepts**
- [Reverse RAG Explained](docs/concepts/reverse-rag.md) - The paper and core insight
- [Architecture](docs/concepts/architecture.md) - System design and components

**🛠️ Guides & How-Tos**
- [Configuration](docs/guides/configuration.md) - Settings and environment variables
- [Atomization Strategies](docs/guides/atomization.md) - Sentence vs LLM atomization
- [CLI Reference](docs/guides/cli.md) - Command-line usage

**🎓 Tutorials**
- [Getting Started](docs/tutorials/getting-started.md) - Your first 10 minutes
- *Coming soon: Building a FAQ bot, Hybrid retrieval, Custom providers*

**🔌 API Reference**
- *Coming soon: Full API documentation*

## When to Use Isotope

**Great fit:**
- ✅ FAQ-style content where questions are predictable (support docs, technical documentation)
- ✅ Knowledge bases with factual, Q&A-structured information
- ✅ When precision matters more than recall (legal, medical, compliance)
- ✅ When you want meaningful confidence scores (threshold-based filtering)

**Consider traditional RAG or hybrid approach when:**
- ❌ Queries are exploratory and unpredictable
- ❌ You can't afford to miss any relevant content (broad discovery)
- ❌ Content doesn't map naturally to Q&A format (narrative, creative writing)
- ❌ You need semantic search over unstructured brainstorming notes

**Trade-offs:**

Isotope trades recall for precision. If users ask questions you didn't anticipate, scores will be low. But you'll *know* they're low—the confidence scores are meaningful.

**Mitigation:** Isotope works great in **hybrid mode** - use question-matching for high-confidence results, fall back to chunk search for low scores.

**Other strategies:**
- Query expansion (rephrase queries before search)
- Re-ranking with cross-encoders

See [Limitations](docs/concepts/reverse-rag.md#the-trade-off) for details.

## Contributing

Isotope is in active development and we welcome contributions!

**Ways to contribute:**
- 🐛 [Report bugs](https://github.com/sreejithraman/isotope/issues)
- 💡 [Request features](https://github.com/sreejithraman/isotope/issues)
- 📖 Improve documentation
- 🔧 Submit pull requests

**Development setup:**
```bash
git clone https://github.com/sreejithraman/isotope.git
cd isotope
pip install -e ".[dev]"
pre-commit install
pytest  # Run tests
```

See `CONTRIBUTING.md` for detailed guidelines (coming soon).

## Citation

If you use Isotope in your research, please cite the paper:

```bibtex
@article{raina2024question,
  title={Question-Based Retrieval using Atomic Units for Enterprise RAG},
  author={Raina, Vatsal and Gales, Mark},
  journal={arXiv preprint arXiv:2405.12363},
  year={2024}
}
```

And consider citing this implementation:

```bibtex
@software{isotope_rag,
  title={Isotope: Reverse RAG Database},
  author={Raman, Sree},
  year={2026},
  url={https://github.com/sreejithraman/isotope}
}
```

## License

MIT

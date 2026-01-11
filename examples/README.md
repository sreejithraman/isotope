# Isotope Examples

Try Isotope in 5 minutes with these ready-to-use examples.

## Before You Start

You'll need:

1. **Python 3.11 or newer** - Check with `python --version` in your terminal
2. **A terminal app** - Terminal (Mac), Command Prompt or PowerShell (Windows)
3. **One of these AI providers:**

| Provider | Cost | Setup |
|----------|------|-------|
| **Gemini** | FREE tier! | [Get API key](https://aistudio.google.com/app/apikey) (Google account required) |
| **OpenAI** | ~$0.01-0.03 per query | [Get API key](https://platform.openai.com/api-keys) (payment required) |
| **Ollama** | FREE forever | [Download app](https://ollama.com) (runs locally on your computer) |

**Our recommendation:** Start with **Gemini** (free) or **Ollama** (local) if you don't want to pay.

---

## Quick Start

### Step 1: Open your terminal

- **Mac:** Press `Cmd + Space`, type "Terminal", press Enter
- **Windows:** Press `Win + R`, type "cmd", press Enter

### Step 2: Navigate to the isotope folder

If you cloned the repo to your Downloads folder:
```bash
cd ~/Downloads/isotope
```

### Step 3: Install Isotope

```bash
pip install -e ".[all]"
```

You should see a bunch of text scroll by, ending with "Successfully installed..."

> **Quick alternative:** If you just want to try the TUI immediately, run `make tui` - it auto-installs dependencies for you. For CLI: `make cli ARGS="--help"`

### Step 4: Choose your provider and initialize

Pick ONE of these options:

**Option A: Gemini (FREE tier)**
```bash
isotope init --llm-model gemini/gemini-3-flash-preview --embedding-model gemini/gemini-embedding-001
```

**Option B: OpenAI (paid)**
```bash
isotope init --llm-model openai/gpt-5-mini-2025-08-07 --embedding-model openai/text-embedding-3-small
```

**Option C: Ollama (local, FREE)**

First, [download Ollama](https://ollama.com) and install it. Then run:
```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

Then initialize Isotope:
```bash
isotope init --llm-model ollama/llama3.2 --embedding-model ollama/nomic-embed-text
```

### Step 5: Enter your API key

If using Gemini or OpenAI, you'll be prompted:
```
No GEMINI_API_KEY found in environment.
Get your API key from https://aistudio.google.com/app/apikey

Enter your API key (or press Enter to skip): <paste your key here>
Save to .env file for future use? [Y/n]: Y
```

(Ollama users skip this step - no API key needed!)

### Step 6: Load an example document

Let's try the Hacker Laws - a fun collection of programming wisdom:
```bash
isotope ingest examples/data/hacker-laws.pdf
```

You should see:
```
Ingested 1 files (X chunks)
Created X atoms
Indexed X questions
```

### Step 7: Ask a question!

```bash
isotope query "What happens when you add people to a late project?"
```

You should see an answer about Brooks' Law!

---

## Try More Examples

### Hacker Laws (Developer Culture)

Programming wisdom and "laws" like Conway's Law, the Peter Principle, and more.

```bash
isotope ingest examples/data/hacker-laws.pdf
isotope query "What happens when you add people to a late project?"
isotope query "What is the Pareto principle?"
isotope query "Why do software estimates always take longer?"
```

### Berkshire Hathaway 10-K (Financial)

Warren Buffett's annual report - a real SEC filing about insurance, railroads, and more.

```bash
isotope ingest "examples/data/Berkshire Hathaway - 10k - 2024 Annual Report.html"
isotope query "What are Berkshire's main business segments?"
isotope query "What risks does Berkshire face?"
```

### Isotope's Own Docs (Meta)

Use Isotope to search its own documentation. Very meta!

```bash
isotope ingest docs/
isotope query "How does reverse RAG work?"
isotope query "What's the difference between atoms and chunks?"
```

---

## Helpful Commands

| Command | What it does |
|---------|--------------|
| `isotope status` | See what's been indexed |
| `isotope list` | List all indexed documents |
| `isotope query "..." --raw` | See raw results without AI summary |
| `isotope query "..." -k 10` | Get more results (default is 5) |

---

## Troubleshooting

### "No OPENAI_API_KEY found" (or GEMINI_API_KEY)

Your API key isn't set. Either:
- Run `isotope init` again and enter your key when prompted
- Or set it manually: `export OPENAI_API_KEY=your-key-here`

### "No results found"

- Make sure you ran `isotope ingest` first
- Check `isotope status` to verify documents are indexed

### "Connection refused" (Ollama)

Ollama isn't running. Start it:
- **Mac:** Open the Ollama app from Applications
- **Windows:** Run `ollama serve` in a separate terminal

### Ingestion is really slow

This is normal for large documents - Isotope generates many questions per fact. For faster (but less precise) results, add this to your `isotope.yaml`:
```yaml
use_sentence_atomizer: true
```

### Still stuck?

Check the [main documentation](../docs/) or [open an issue](https://github.com/sreejithraman/isotope/issues).

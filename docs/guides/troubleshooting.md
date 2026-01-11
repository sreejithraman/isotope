# Troubleshooting Guide

Common issues and solutions when using Isotope.

## Installation Issues

### Python Version

Isotope requires Python 3.11 or newer.

```bash
# Check your Python version
python --version

# If too old, install a newer version
# macOS: brew install python@3.12
# Linux: sudo apt install python3.12
```

### Missing Dependencies

If you see `ModuleNotFoundError`, install the required extras:

```bash
# Install all dependencies
pip install isotope-rag[all]

# Or install specific extras
pip install "isotope-rag[cli,litellm,chroma]"
```

### LiteLLM Import Errors

If you see errors about `litellm`, ensure it's installed:

```bash
pip install "isotope-rag[litellm]"
```

---

## Configuration Issues

### Config File Not Found

Isotope looks for config files in this order:
1. `isotope.yaml` in current directory
2. `isotope.yml` in current directory
3. `.isotoperc` in current directory
4. Parent directories (searched upward)

**Solution:** Run `isotope init` to create a config file, or check your working directory.

### Environment Variables Not Working

Environment variables must be prefixed with `ISOTOPE_`:

```bash
# Correct
export ISOTOPE_LITELLM_LLM_MODEL=openai/gpt-5-mini-2025-08-07

# Wrong (missing prefix)
export LLM_MODEL=openai/gpt-5-mini-2025-08-07
```

Verify with `isotope config` to see where each setting comes from.

### Unknown Config Keys Warning

If you see warnings about unknown config keys, you may have a typo in your `isotope.yaml`. Check spelling against the [Configuration Guide](./configuration.md).

---

## Ingestion Issues

### Slow Ingestion

Ingestion can be slow for large documents because Isotope generates multiple questions per fact.

**Solutions:**

1. **Use sentence atomizer** (faster but less precise):
   ```yaml
   settings:
     use_sentence_atomizer: true
   ```

2. **Reduce questions per atom**:
   ```yaml
   settings:
     questions_per_atom: 3  # Default is 5
   ```

3. **Increase concurrency** (if not rate-limited):
   ```yaml
   settings:
     max_concurrent_llm_calls: 10  # Default is 10
   ```

### Out of Memory

Large files can consume significant memory during atomization.

**Solutions:**

1. Split large files into smaller chunks before ingesting
2. Reduce `questions_per_atom` to generate fewer embeddings
3. Process files one at a time instead of entire directories

### Unsupported File Format

Isotope supports:
- **Text:** `.txt`, `.md`
- **PDF:** `.pdf` (requires `pdfplumber`)
- **HTML:** `.html`, `.htm`

**Solution:** Convert unsupported formats to one of the above, or implement a custom `Loader`.

### Files Being Skipped

If files are skipped during re-ingestion, Isotope detected they haven't changed (based on content hash).

```bash
# Force re-ingestion by deleting and re-adding
isotope delete path/to/file.md
isotope ingest path/to/file.md
```

---

## Retrieval Issues

### No Results Found

If queries return no results:

1. **Check if data is indexed:**
   ```bash
   isotope status
   isotope list
   ```

2. **Try a different query:** Isotope matches query-to-question, so phrase your query as a question:
   ```bash
   # Less effective
   isotope query "authentication"

   # More effective
   isotope query "How does authentication work?"
   ```

3. **Increase result count:**
   ```bash
   isotope query "How does authentication work?" -k 10
   ```

### Low Confidence Scores

Low scores (< 0.7) indicate weak semantic matches.

**Solutions:**

1. **Increase questions per atom** during ingestion to create more entry points
2. **Try different query phrasing** - more specific questions work better
3. **Check matched questions** to understand what's being matched:
   ```bash
   isotope query "How do I authenticate?" --show-matched-questions
   ```

### Poor Quality Answers

If the synthesized answer is wrong or off-topic:

1. **Check raw results** to see what chunks are being retrieved:
   ```bash
   isotope query "How does auth work?" --raw
   ```

2. **Adjust retrieval** before changing synthesis - poor answers often come from poor retrieval

---

## Provider-Specific Issues

### OpenAI

**"Rate limit exceeded"**

```yaml
settings:
  max_concurrent_llm_calls: 2
```

Or use the rate-limit profile:
```python
settings = Settings.with_profile("conservative")
```

**"Invalid API key"**

Ensure your API key is set:
```bash
export OPENAI_API_KEY=sk-...
# Or
export ISOTOPE_LLM_API_KEY=sk-...
```

### Anthropic

**"Overloaded" errors**

Reduce concurrency:
```yaml
settings:
  max_concurrent_llm_calls: 2
```

**"Invalid API key"**

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### Google Gemini

**"API key not valid"**

Get an API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

```bash
export GEMINI_API_KEY=...
```

**"Resource exhausted"**

Free tier has strict rate limits:
```yaml
settings:
  max_concurrent_llm_calls: 1
```

### Ollama (Local)

**"Connection refused"**

Ollama isn't running. Start it:

```bash
# macOS: Open the Ollama app
# Linux/Windows:
ollama serve
```

**"Model not found"**

Pull the required models:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

**Slow responses**

Local models are CPU-bound by default. For faster inference:
- Use smaller models (`llama3.2` instead of `llama3.2:70b`)
- Enable GPU acceleration if available
- Reduce `questions_per_atom`

---

## Database Issues

### Corrupted Database

If you see SQLite or Chroma errors:

1. **Backup existing data** (if possible):
   ```bash
   cp -r isotope_data isotope_data.backup
   ```

2. **Delete and rebuild:**
   ```bash
   rm -rf isotope_data
   isotope ingest ./your-docs/
   ```

### Database Location

By default, Isotope stores data in `./isotope_data/`. Change with:

```yaml
data_dir: /path/to/custom/location
```

Or:
```bash
export ISOTOPE_DATA_DIR=/path/to/custom/location
```

---

## Getting Help

If these solutions don't work:

1. **Check verbose output:**
   ```bash
   isotope --verbose ingest ./docs
   ```

2. **Check the [FAQ](../faq.md)** for common questions

3. **Open an issue** at [GitHub Issues](https://github.com/sreejithraman/isotope/issues) with:
   - Isotope version (`pip show isotope-rag`)
   - Python version (`python --version`)
   - Full error message
   - Steps to reproduce

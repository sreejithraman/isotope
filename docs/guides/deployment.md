# Deployment Guide

Guidelines for deploying Isotope in production environments.

## Local Deployment

The default configuration uses local storage, suitable for single-machine deployments.

### Default Storage

```yaml
# isotope.yaml
provider: litellm
llm_model: openai/gpt-5-mini-2025-08-07
embedding_model: openai/text-embedding-3-small
data_dir: ./isotope_data
```

This creates:
- `isotope_data/chunks.db` - SQLite for chunk/atom/source metadata
- `isotope_data/questions/` - Chroma vector store for embeddings

### Storage Sizing

Rough estimates per 1,000 chunks:

| Component | Size |
|-----------|------|
| SQLite metadata | 1-5 MB |
| Chroma embeddings | 50-100 MB |
| Total | ~100 MB |

For 100,000 chunks, plan for ~10 GB storage.

### File System Considerations

- **SSD recommended** for Chroma queries
- **Ensure write permissions** for the `data_dir`
- **Avoid network mounts** (NFS, SMB) for Chroma - latency impacts query performance

---

## Security

### API Key Management

**Never commit API keys to version control.**

Options for managing keys:

1. **Environment variables** (recommended for production):
   ```bash
   export OPENAI_API_KEY=sk-...
   ```

2. **`.env` file** (development only):
   ```bash
   # .env
   ISOTOPE_LLM_API_KEY=sk-...
   ```
   Ensure `.env` is in `.gitignore`.

3. **Secrets manager** (enterprise):
   Load secrets at runtime from AWS Secrets Manager, HashiCorp Vault, etc.

### Input Validation

Isotope doesn't validate query content. If exposing to untrusted users:

- Sanitize inputs before passing to `isotope query`
- Implement rate limiting at the application layer
- Log queries for audit purposes

### Data Access

The SQLite and Chroma stores are unencrypted. For sensitive data:

- Use encrypted file systems
- Restrict file permissions:
  ```bash
  chmod 700 isotope_data
  ```
- Consider encrypting backups

---

## Backup & Recovery

### Backup Strategy

The entire `data_dir` should be backed up together:

```bash
# Stop any running Isotope processes first
tar -czf isotope_backup_$(date +%Y%m%d).tar.gz isotope_data/
```

### Backup Schedule

| Frequency | Scenario |
|-----------|----------|
| After each ingest | Critical data |
| Daily | Active development |
| Weekly | Stable content |

### Recovery

```bash
# Restore from backup
tar -xzf isotope_backup_20260111.tar.gz

# Verify
isotope status
```

### Rebuilding from Source

If backups fail, you can rebuild from original documents:

```bash
rm -rf isotope_data
isotope ingest ./your-docs/
```

This re-generates all questions and embeddings.

---

## Monitoring

### Health Checks

Basic health check:
```bash
isotope status
```

Programmatic check:
```python
from isotope import Isotope
from isotope.configuration import LiteLLMProvider, LocalStorage

iso = Isotope(provider=..., storage=...)
stats = iso.status()
print(f"Sources: {stats.source_count}, Questions: {stats.question_count}")
```

### Metrics to Track

| Metric | Description | Concern Threshold |
|--------|-------------|-------------------|
| Query latency | Time to return results | > 2s |
| Ingestion time | Time per document | > 60s/doc |
| Storage size | Disk usage | > 80% capacity |
| Error rate | Failed queries/ingests | > 1% |

### Logging

Isotope uses Python's standard logging:

```python
import logging
logging.basicConfig(level=logging.INFO)

# For debug output
logging.getLogger("isotope").setLevel(logging.DEBUG)
```

CLI verbose mode:
```bash
isotope --verbose ingest ./docs
```

---

## Scaling Considerations

### When to Scale

Consider scaling when:
- Corpus exceeds 100,000 chunks
- Query latency exceeds acceptable thresholds
- Multiple instances need shared access

### Horizontal Scaling Options

**Option 1: External Vector Store**

Replace Chroma with a hosted vector database:
- Pinecone
- Weaviate
- Milvus
- Qdrant

Implement custom `EmbeddedQuestionStore` for your chosen provider.

**Option 2: External Database**

Replace SQLite with PostgreSQL for:
- Better concurrent access
- Easier backups
- Replication support

Implement custom `ChunkStore`, `AtomStore`, and `SourceRegistry`.

**Option 3: Read Replicas**

For read-heavy workloads:
- Primary instance handles ingestion
- Read replicas serve queries
- Sync via file copy or database replication

---

## Docker Deployment

### Basic Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml .
RUN pip install isotope-rag[cli,litellm,chroma]

# Copy config and data
COPY isotope.yaml .
COPY isotope_data/ ./isotope_data/

# Set environment variables
ENV ISOTOPE_DATA_DIR=/app/isotope_data

# Default command
CMD ["isotope-tui"]
```

### Docker Compose

```yaml
version: '3.8'
services:
  isotope:
    build: .
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ISOTOPE_DATA_DIR=/data
    volumes:
      - isotope_data:/data
    ports:
      - "8000:8000"

volumes:
  isotope_data:
```

### Persistent Storage

Mount a volume for `data_dir` to persist between container restarts:

```bash
docker run -v isotope_data:/app/isotope_data isotope-image
```

---

## CI/CD Integration

### Automated Ingestion

```yaml
# .github/workflows/ingest.yml
name: Update Knowledge Base
on:
  push:
    paths:
      - 'docs/**'

jobs:
  ingest:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install Isotope
        run: pip install isotope-rag[cli,litellm,chroma]

      - name: Ingest docs
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: isotope ingest ./docs/

      - name: Upload artifacts
        uses: actions/upload-artifact@v4
        with:
          name: isotope-data
          path: isotope_data/
```

### Testing Retrieval Quality

```python
# tests/test_retrieval.py
def test_known_question():
    iso = Isotope(...)
    results = iso.retriever().search("How does authentication work?")

    # Verify we get relevant results
    assert len(results) > 0
    assert results[0].score > 0.7
    assert "auth" in results[0].chunk.content.lower()
```

---

## Cost Optimization

### Reduce Ingestion Costs

1. **Use sentence atomizer** for large corpora:
   ```yaml
   settings:
     use_sentence_atomizer: true
   ```

2. **Reduce questions per atom**:
   ```yaml
   settings:
     questions_per_atom: 3
   ```

3. **Use cheaper models for ingestion**:
   - Ingestion: GPT-5-mini or Gemini Flash
   - Queries: Higher quality model if needed

### Reduce Query Costs

1. **Cache frequent queries** at the application layer
2. **Use `--raw`** when synthesis isn't needed
3. **Reduce `k`** for fewer results when appropriate

### Model Cost Comparison

| Model | Relative Cost | Quality |
|-------|--------------|---------|
| GPT-5-nano | $ | Good |
| GPT-5-mini | $$ | Better |
| Claude Haiku | $$ | Better |
| Claude Sonnet | $$$ | Best |
| Gemini Flash | $ | Good |
| Ollama (local) | Free | Varies |

---

## Checklist

Before going to production:

- [ ] API keys stored securely (not in code)
- [ ] Backup strategy in place
- [ ] Monitoring/alerting configured
- [ ] Storage sized appropriately
- [ ] Rate limits configured for your API tier
- [ ] Error handling in application layer
- [ ] Logging enabled for debugging
- [ ] Recovery procedure documented and tested

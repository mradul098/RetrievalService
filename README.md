# Hybrid Retrieval Service

Read-path service that accepts a natural-language query and returns relevant document chunks, with tenant-level access control enforced.

## Architecture

- **Hybrid search:** lexical BM25 + kNN vector search, fused via application-side Reciprocal Rank Fusion (RRF)
- **Two independent search legs:** no native ES `rrf` retriever (Community/Basic licensing constraint)
- **Defense-in-depth:** tenant_id + is_deleted + legal_hold filters applied as pre-filters on both search legs, then re-checked in application code
- **Prompt-agnostic:** returns raw chunks — prompt assembly is RAG Orchestration Service's responsibility

## Environments

- `ENVIRONMENT=local`: Ollama embeddings + FAISS + SQLite FTS5. No Elasticsearch or GCP credentials needed.
- `ENVIRONMENT=production`: Vertex AI embeddings + Elasticsearch. ADC authentication.

## Running Locally

```bash
# 1. Pull the embedding model
ollama pull nomic-embed-text

# 2. Setup environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# 3. Seed test data (embeds chunks via Ollama → builds FAISS index + SQLite DB)
ENVIRONMENT=local python -m app.scripts.seed_local

# 4. Run
ENVIRONMENT=local uvicorn app.main:app --port 8002
```

## Testing

```bash
# Unit tests (no Ollama/FAISS needed)
python -m pytest tests/ -v

# End-to-end test (requires seeded data + running service)
curl http://localhost:8002/internal/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "tenantId": "00103",
    "requestingUser": {"userId": "test@example.com"},
    "query": {
      "text": "total assets Q2 balance sheet",
      "topK": 5,
      "searchMode": "hybrid"
    }
  }'
```

## Search Modes

| Mode | What it does |
|---|---|
| `hybrid` (default) | Both lexical + kNN legs, fused via RRF |
| `lexical` | BM25 only — useful for comparison/debugging |
| `semantic` | kNN only — requires embedding, skips BM25 |

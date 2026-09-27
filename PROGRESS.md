# Hybrid Retrieval Service — Progress

## Status: COMPLETE

## Completed
- [x] Project skeleton (pyproject.toml, requirements.txt, .env.example, config.py)
- [x] Pydantic models (requests.py, responses.py)
- [x] ABC interfaces (embedding_client.py, vector_store.py)
- [x] Ollama embedding adapter (local)
- [x] Vertex AI embedding adapter (production)
- [x] FAISS + SQLite FTS5 store adapter (local)
- [x] Elasticsearch store adapter (production)
- [x] RRF fusion logic (fusion.py)
- [x] Retrieval pipeline service (retrieval.py)
- [x] POST /internal/retrieve route
- [x] FastAPI app entry point (main.py)
- [x] Seed script for local test data
- [x] Tests
- [x] README + Dockerfile

## In Progress
- None

## Remaining
- None

## Open Decisions
- MMR diversification deferred per LLD §6.1a TBD-17 (ship without, instrument later)
- Reranking deferred per LLD §11
- Caching deferred per LLD §10

## How to run (once built)
```bash
# Seed test data first
ENVIRONMENT=local python -m app.scripts.seed_local

# Start service
ENVIRONMENT=local uvicorn app.main:app --port 8002
```

## How to test (once built)
```bash
curl http://localhost:8002/internal/retrieve \
  -H "Content-Type: application/json" \
  -d '{"tenantId":"00103","requestingUser":{"userId":"test@example.com"},"query":{"text":"total assets Q2 balance sheet","topK":5,"searchMode":"hybrid"}}'
```

"""
Hybrid Retrieval Service — FastAPI application entry point.

Startup:
  ENVIRONMENT=local uvicorn app.main:app --port 8002

Read-only service: queries embedded chunks from Elasticsearch (production)
or FAISS+SQLite (local). Never writes to any store.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from app.config import get_settings

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: create the embedding client and vector store adapters
    based on environment, then create the retrieval service.
    """
    settings = get_settings()

    # --- Create embedding client ---
    if settings.is_local:
        from app.adapters.local.ollama_embedding import OllamaEmbeddingClient

        embedding_client = OllamaEmbeddingClient(
            base_url=settings.ollama_base_url,
            model=settings.ollama_embedding_model,
        )
        logger.info(
            "embedding_client_initialized",
            adapter="ollama",
            model=settings.ollama_embedding_model,
        )
    else:
        from app.adapters.production.vertex_embedding import VertexEmbeddingClient

        embedding_client = VertexEmbeddingClient(
            project_id=settings.gcp_project_id,
            location=settings.gcp_location,
            model=settings.vertex_embedding_model,
        )
        logger.info(
            "embedding_client_initialized",
            adapter="vertex_ai",
            model=settings.vertex_embedding_model,
        )

    # --- Create vector store ---
    if settings.is_local:
        from app.adapters.local.faiss_sqlite_store import FaissSqliteStore

        vector_store = FaissSqliteStore(
            faiss_index_path=settings.local_faiss_index_path,
            sqlite_db_path=settings.local_sqlite_db_path,
        )
        logger.info(
            "vector_store_initialized",
            adapter="faiss_sqlite",
            faiss_path=settings.local_faiss_index_path,
            sqlite_path=settings.local_sqlite_db_path,
        )
    else:
        from app.adapters.production.elasticsearch_store import ElasticsearchStore

        vector_store = ElasticsearchStore(
            url=settings.elasticsearch_url,
            api_key=settings.elasticsearch_api_key,
            index_prefix=settings.elasticsearch_index_prefix,
        )
        logger.info(
            "vector_store_initialized",
            adapter="elasticsearch",
            url=settings.elasticsearch_url,
        )

    # --- Create retrieval service ---
    from app.services.retrieval import RetrievalService

    app.state.retrieval_service = RetrievalService(
        embedding_client=embedding_client,
        vector_store=vector_store,
        settings=settings,
    )

    logger.info("retrieval_service_started", environment=settings.environment)
    yield
    logger.info("retrieval_service_stopped")


app = FastAPI(
    title="Hybrid Retrieval Service",
    description=(
        "Read-path service that accepts a natural-language query and returns "
        "relevant chunks with access control enforced. Uses hybrid search "
        "(lexical BM25 + kNN vector) with application-side RRF fusion."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# --- Register routes ---
from app.routes.retrieve import router as retrieve_router  # noqa: E402

app.include_router(retrieve_router)


@app.get("/health")
async def health_check():
    """Liveness probe."""
    return {"status": "healthy", "service": "retrieval-service"}

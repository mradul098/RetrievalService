"""
Application configuration — loaded from environment variables / .env file.
Switches between local (Ollama/FAISS/SQLite) and production (Vertex AI/Elasticsearch).
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """All configuration for the Hybrid Retrieval Service."""

    # --- Environment toggle ---
    environment: str = "local"  # "local" or "production"

    # --- Local (Ollama) embedding ---
    ollama_base_url: str = "http://localhost:11434"
    ollama_embedding_model: str = "nomic-embed-text"

    # --- Production (Vertex AI / Gemini) embedding ---
    gcp_project_id: str = ""
    gcp_location: str = "us-central1"
    vertex_embedding_model: str = "text-embedding-004"

    # --- Production (Elasticsearch) ---
    elasticsearch_url: str = "https://localhost:9200"
    elasticsearch_api_key: str = ""
    elasticsearch_index_prefix: str = "rag-chunks"

    # --- Local store paths ---
    local_faiss_index_path: str = "data/faiss.index"
    local_sqlite_db_path: str = "data/chunks.db"

    # --- Retrieval settings ---
    top_k_max: int = 50
    top_k_default: int = 8
    rank_constant: int = 20  # RRF constant (k in 1/(k+rank+1))
    embedding_dims: int = 768

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 8002

    @property
    def is_local(self) -> bool:
        return self.environment.lower() == "local"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache()
def get_settings() -> Settings:
    """Cached singleton."""
    return Settings()

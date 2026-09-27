"""
Vertex AI embedding adapter — used for production on GCP.

Uses the same model/version as Embedding & Indexing Service
to ensure vectors are in the same embedding space (LLD §5.1).

Prerequisites:
  - Vertex AI API enabled
  - ADC configured
"""

from __future__ import annotations

from typing import List

from app.interfaces.embedding_client import EmbeddingClient, EmbeddingError


class VertexEmbeddingClient(EmbeddingClient):
    """Embedding client using Vertex AI's text-embedding models."""

    def __init__(self, project_id: str, location: str, model: str):
        self._project_id = project_id
        self._location = location
        self._model = model
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy-init Vertex AI SDK."""
        if self._initialized:
            return
        try:
            import vertexai
            vertexai.init(project=self._project_id, location=self._location)
            self._initialized = True
        except Exception as exc:
            raise EmbeddingError(
                f"Failed to initialize Vertex AI: {exc}",
                provider="vertex_ai",
            ) from exc

    async def embed(self, text: str) -> List[float]:
        """Embed text via Vertex AI's text-embedding model."""
        self._ensure_initialized()

        try:
            from vertexai.language_models import TextEmbeddingModel

            model = TextEmbeddingModel.from_pretrained(self._model)
            embeddings = model.get_embeddings([text])

            if not embeddings:
                raise EmbeddingError(
                    "Vertex AI returned empty embeddings",
                    provider="vertex_ai",
                )

            return embeddings[0].values

        except ImportError as exc:
            raise EmbeddingError(
                "google-cloud-aiplatform is not installed.",
                provider="vertex_ai",
            ) from exc
        except Exception as exc:
            if isinstance(exc, EmbeddingError):
                raise
            error_str = str(exc).lower()
            is_retryable = any(
                kw in error_str
                for kw in ["rate_limit", "quota", "503", "429", "timeout"]
            )
            raise EmbeddingError(
                f"Vertex AI embedding failed: {exc}",
                provider="vertex_ai",
                is_retryable=is_retryable,
            ) from exc

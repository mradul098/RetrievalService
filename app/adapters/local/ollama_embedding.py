"""
Ollama embedding adapter — used for local development.

Calls Ollama's /api/embed endpoint to produce dense vectors.
Must use the same model as what is configured for the seed script.

Prerequisites:
  ollama pull nomic-embed-text
"""

from __future__ import annotations

from typing import List

import httpx

from app.interfaces.embedding_client import EmbeddingClient, EmbeddingError


class OllamaEmbeddingClient(EmbeddingClient):
    """Embedding client that calls a local Ollama instance."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "nomic-embed-text"):
        self._base_url = base_url.rstrip("/")
        self._model = model

    async def embed(self, text: str) -> List[float]:
        """Embed text via Ollama's /api/embed endpoint."""
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
                response = await client.post(
                    f"{self._base_url}/api/embed",
                    json={
                        "model": self._model,
                        "input": text,
                    },
                )

                if response.status_code != 200:
                    raise EmbeddingError(
                        f"Ollama returned {response.status_code}: {response.text}",
                        provider="ollama",
                        is_retryable=response.status_code >= 500,
                    )

                data = response.json()
                # Ollama /api/embed returns {"embeddings": [[...], [...]]}
                embeddings = data.get("embeddings", [])
                if not embeddings:
                    raise EmbeddingError(
                        "Ollama returned empty embeddings",
                        provider="ollama",
                    )
                return embeddings[0]

        except httpx.TimeoutException as exc:
            raise EmbeddingError(
                "Ollama embedding request timed out",
                provider="ollama",
                is_retryable=True,
            ) from exc
        except httpx.ConnectError as exc:
            raise EmbeddingError(
                f"Could not connect to Ollama at {self._base_url}. Is it running?",
                provider="ollama",
                is_retryable=False,
            ) from exc

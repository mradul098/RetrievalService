"""
Abstract interface for embedding clients.

Implementations must produce vectors in the same embedding space
as the one used at index time by Embedding & Indexing Service.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class EmbeddingClient(ABC):
    """Embed text into a dense vector."""

    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """
        Embed a single text string into a dense vector.

        Args:
            text: The query text to embed.

        Returns:
            A list of floats (dimension must match the index's `embedding.dims`).

        Raises:
            EmbeddingError: If the embedding call fails.
        """
        ...


class EmbeddingError(Exception):
    """Raised when an embedding call fails."""

    def __init__(self, message: str, provider: str, is_retryable: bool = False):
        super().__init__(message)
        self.provider = provider
        self.is_retryable = is_retryable

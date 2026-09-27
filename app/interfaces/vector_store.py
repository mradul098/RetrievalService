"""
Abstract interface for the vector/lexical store.

Implementations:
  - Local: FAISS (kNN) + SQLite FTS5 (BM25)
  - Production: Elasticsearch (two plain _search calls)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SearchHit:
    """A single search result from one search leg."""

    chunk_id: str
    score: float
    chunk_text: str
    tenant_id: str
    document_id: Optional[str] = None
    version_series_id: Optional[str] = None
    document_name: Optional[str] = None
    chunk_type: Optional[str] = None
    page_no: Optional[int] = None
    heading: Optional[str] = None
    section: Optional[str] = None
    document_uri: Optional[str] = None
    document_model: Optional[str] = None
    is_deleted: bool = False
    legal_hold: bool = False
    classification: Optional[str] = None


@dataclass
class SearchFilters:
    """Filters applied as pre-filters on both search legs (LLD §9.4)."""

    tenant_id: str
    is_deleted: bool = False
    legal_hold: bool = False
    document_model: Optional[List[str]] = None
    version_series_ids: Optional[List[str]] = None
    classification: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None


class VectorStore(ABC):
    """
    Abstract store supporting both lexical and kNN search.

    Implementations must apply all SearchFilters as pre-filters —
    never as post-filters (LLD §9.4 explains why).
    """

    @abstractmethod
    async def search_lexical(
        self,
        query_text: str,
        filters: SearchFilters,
        size: int = 50,
    ) -> List[SearchHit]:
        """
        BM25/lexical search on chunk_text.

        Args:
            query_text: Raw user query text.
            filters: Pre-filters (tenant, is_deleted, legal_hold, etc.)
            size: Max results to return from this leg.

        Returns:
            Ranked list of SearchHits.
        """
        ...

    @abstractmethod
    async def search_knn(
        self,
        query_vector: List[float],
        filters: SearchFilters,
        k: int = 50,
        num_candidates: int = 200,
    ) -> List[SearchHit]:
        """
        kNN / vector search on embedding field.

        Args:
            query_vector: Dense vector from embedding client.
            filters: Pre-filters (same as lexical).
            k: Number of nearest neighbors to return.
            num_candidates: Candidates considered during ANN search (ES-specific).

        Returns:
            Ranked list of SearchHits.
        """
        ...

"""
Elasticsearch store adapter — used for production.

Issues two independent plain _search calls (lexical + kNN) as per
LLD §6.1, with mandatory access filters as pre-filters on both legs.
Does NOT use native rrf/retriever-tree features (licensing concern).
"""

from __future__ import annotations

from typing import List, Optional

from app.interfaces.vector_store import SearchFilters, SearchHit, VectorStore


class ElasticsearchStore(VectorStore):
    """
    Production vector store backed by Elasticsearch.

    Uses two plain _search calls: one for BM25 lexical, one for kNN vector.
    Both apply the same filter clauses as pre-filters.
    """

    def __init__(self, url: str, api_key: str, index_prefix: str):
        self._url = url
        self._api_key = api_key
        self._index_prefix = index_prefix
        self._client = None

    def _ensure_client(self):
        """Lazy-init the Elasticsearch client."""
        if self._client is not None:
            return

        try:
            from elasticsearch import AsyncElasticsearch

            self._client = AsyncElasticsearch(
                self._url,
                api_key=self._api_key,
                verify_certs=True,
            )
        except ImportError as exc:
            raise RuntimeError(
                "elasticsearch package is required. Install with: pip install elasticsearch"
            ) from exc

    def _get_index_name(self, tenant_id: str) -> str:
        """Resolve the per-tenant index name: rag-chunks-{tenantId}."""
        return f"{self._index_prefix}-{tenant_id}"

    def _build_filter_clauses(self, filters: SearchFilters) -> List[dict]:
        """Build the ES filter clause array from SearchFilters."""
        clauses = [
            {"term": {"tenant_id": filters.tenant_id}},
            {"term": {"is_deleted": filters.is_deleted}},
            {"term": {"legal_hold": filters.legal_hold}},
        ]

        if filters.document_model:
            clauses.append({"terms": {"document_model": filters.document_model}})

        if filters.version_series_ids:
            clauses.append({"terms": {"version_series_id": filters.version_series_ids}})

        if filters.classification:
            clauses.append({"term": {"classification": filters.classification}})

        if filters.date_from or filters.date_to:
            range_clause = {}
            if filters.date_from:
                range_clause["gte"] = filters.date_from
            if filters.date_to:
                range_clause["lte"] = filters.date_to
            clauses.append({"range": {"created_at": range_clause}})

        return clauses

    def _hit_to_search_hit(self, hit: dict) -> SearchHit:
        """Convert an ES _search hit to a SearchHit."""
        source = hit["_source"]
        return SearchHit(
            chunk_id=source.get("chunk_id", hit["_id"]),
            score=hit.get("_score", 0.0),
            chunk_text=source.get("chunk_text", ""),
            tenant_id=source.get("tenant_id", ""),
            document_id=source.get("document_id"),
            version_series_id=source.get("version_series_id"),
            document_name=source.get("document_name"),
            chunk_type=source.get("chunk_type"),
            page_no=source.get("page_no"),
            heading=source.get("heading"),
            section=source.get("section"),
            document_uri=source.get("document_uri"),
            document_model=source.get("document_model"),
            is_deleted=source.get("is_deleted", False),
            legal_hold=source.get("legal_hold", False),
            classification=source.get("classification"),
        )

    async def search_lexical(
        self,
        query_text: str,
        filters: SearchFilters,
        size: int = 50,
    ) -> List[SearchHit]:
        """
        BM25 lexical search — plain _search with match query + filters.
        Matches LLD §6.1 Step 2a.
        """
        self._ensure_client()
        index = self._get_index_name(filters.tenant_id)
        filter_clauses = self._build_filter_clauses(filters)

        body = {
            "query": {
                "bool": {
                    "must": [
                        {"match": {"chunk_text": query_text}}
                    ],
                    "filter": filter_clauses,
                }
            },
            "size": size,
            "_source": True,
        }

        response = await self._client.search(index=index, body=body)
        hits = response.get("hits", {}).get("hits", [])
        return [self._hit_to_search_hit(h) for h in hits]

    async def search_knn(
        self,
        query_vector: List[float],
        filters: SearchFilters,
        k: int = 50,
        num_candidates: int = 200,
    ) -> List[SearchHit]:
        """
        kNN vector search — plain _search with knn clause + filters.
        Matches LLD §6.1 Step 2b.
        """
        self._ensure_client()
        index = self._get_index_name(filters.tenant_id)
        filter_clauses = self._build_filter_clauses(filters)

        body = {
            "knn": {
                "field": "embedding",
                "query_vector": query_vector,
                "k": k,
                "num_candidates": num_candidates,
                "filter": filter_clauses,
            },
            "_source": True,
        }

        response = await self._client.search(index=index, body=body)
        hits = response.get("hits", {}).get("hits", [])
        return [self._hit_to_search_hit(h) for h in hits]

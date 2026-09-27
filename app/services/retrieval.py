"""
Retrieval pipeline — the core business logic of the Hybrid Retrieval Service.

Implements the full flow from LLD §6.2:
  1. Embed query text
  2. Fire two search legs (lexical + kNN) — or one, based on searchMode
  3. Fuse results via application-side RRF
  4. Defense-in-depth tenant re-check
  5. Return ranked results

Handles degraded modes:
  - Embedding failure → fall back to lexical-only search + warning
  - One search leg fails → use the other + warning
"""

from __future__ import annotations

import asyncio
import time
from typing import List, Optional

import structlog

from app.config import Settings
from app.interfaces.embedding_client import EmbeddingClient, EmbeddingError
from app.interfaces.vector_store import SearchFilters, SearchHit, VectorStore
from app.models.requests import RetrievalRequest
from app.models.responses import (
    ChunkResult,
    LatencyBreakdown,
    RetrievalResponse,
    ScoreBreakdown,
)
from app.services.fusion import fuse_results

logger = structlog.get_logger()


class RetrievalService:
    """Orchestrates the embed → search → fuse → filter pipeline."""

    def __init__(
        self,
        embedding_client: EmbeddingClient,
        vector_store: VectorStore,
        settings: Settings,
    ):
        self._embedding = embedding_client
        self._store = vector_store
        self._settings = settings

    def _build_filters(self, request: RetrievalRequest) -> SearchFilters:
        """Build SearchFilters from the request, always enforcing tenant_id."""
        filters = request.query.filters
        return SearchFilters(
            tenant_id=request.tenant_id,
            is_deleted=filters.is_deleted if filters else False,
            legal_hold=filters.legal_hold if filters else False,
            document_model=filters.document_model if filters else None,
            version_series_ids=filters.version_series_ids if filters else None,
            classification=filters.classification if filters else None,
            date_from=filters.date_range.from_date if filters and filters.date_range else None,
            date_to=filters.date_range.to_date if filters and filters.date_range else None,
        )

    def _defense_in_depth_check(
        self, hits: List[SearchHit], tenant_id: str,
        allowed_document_models: Optional[List[str]] = None,
    ) -> List[SearchHit]:
        """
        Application-side re-check of returned hits (LLD §9.4).

        Enforces:
          1. tenant_id must match (primary tenant isolation)
          2. is_deleted must be False
          3. legal_hold must be False
          4. document_model must be in allowed_document_models if supplied
             (LLD §9.2a — MVP ACL: tenantId + documentModel)

        This is a safety net — the primary enforcement is the ES pre-filter.
        """
        safe = []
        for hit in hits:
            if hit.tenant_id != tenant_id:
                logger.error(
                    "defense_in_depth_tenant_mismatch",
                    expected_tenant=tenant_id,
                    actual_tenant=hit.tenant_id,
                    chunk_id=hit.chunk_id,
                )
                continue
            if hit.is_deleted:
                logger.error(
                    "defense_in_depth_deleted_chunk",
                    chunk_id=hit.chunk_id,
                )
                continue
            if hit.legal_hold:
                logger.error(
                    "defense_in_depth_legal_hold_chunk",
                    chunk_id=hit.chunk_id,
                )
                continue
            # MVP ACL: documentModel must match if the caller scoped to a folder
            if allowed_document_models and hit.document_model:
                if hit.document_model not in allowed_document_models:
                    logger.warning(
                        "defense_in_depth_document_model_mismatch",
                        chunk_id=hit.chunk_id,
                        chunk_model=hit.document_model,
                        allowed_models=allowed_document_models,
                    )
                    continue
            safe.append(hit)
        return safe

    async def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        """
        Execute the full retrieval pipeline.

        Returns a RetrievalResponse — empty results is a valid 200, not a 404.
        """
        total_start = time.monotonic()
        warnings: List[str] = []
        search_filters = self._build_filters(request)
        top_k = min(request.query.top_k, self._settings.top_k_max)
        search_mode = request.query.search_mode

        # --- Step 1: Embed query text ---
        embed_start = time.monotonic()
        query_vector: Optional[List[float]] = None

        if search_mode in ("hybrid", "semantic"):
            try:
                query_vector = await self._embedding.embed(request.query.text)
                logger.info(
                    "query_embedded",
                    tenant_id=request.tenant_id,
                    dims=len(query_vector),
                )
            except EmbeddingError as exc:
                logger.warning(
                    "embedding_failed_fallback_to_lexical",
                    tenant_id=request.tenant_id,
                    error=str(exc),
                )
                warnings.append(f"Embedding failed ({exc.provider}): falling back to lexical-only search.")
                if search_mode == "semantic":
                    # Can't do semantic without a vector
                    embed_ms = int((time.monotonic() - embed_start) * 1000)
                    return RetrievalResponse(
                        request_id=request.request_id,
                        tenant_id=request.tenant_id,
                        turn_id=request.turn_id,
                        results=[],
                        total_candidates_considered=0,
                        latency_ms=LatencyBreakdown(
                            embed=embed_ms,
                            search=0,
                            total=int((time.monotonic() - total_start) * 1000),
                        ),
                        warnings=warnings + ["Semantic search unavailable: embedding failed."],
                    )
                search_mode = "lexical"  # Degrade hybrid → lexical

        embed_ms = int((time.monotonic() - embed_start) * 1000)

        # --- Step 2: Fire search legs ---
        search_start = time.monotonic()
        lexical_hits: List[SearchHit] = []
        knn_hits: List[SearchHit] = []
        # Fetch more candidates than topK for fusion quality
        search_size = min(top_k * 5, self._settings.top_k_max)

        if search_mode == "hybrid":
            # Fire both legs in parallel
            try:
                lexical_task = asyncio.create_task(
                    self._store.search_lexical(
                        query_text=request.query.text,
                        filters=search_filters,
                        size=search_size,
                    )
                )
                knn_task = asyncio.create_task(
                    self._store.search_knn(
                        query_vector=query_vector,
                        filters=search_filters,
                        k=search_size,
                    )
                )

                results = await asyncio.gather(lexical_task, knn_task, return_exceptions=True)

                if isinstance(results[0], Exception):
                    logger.warning("lexical_search_failed", error=str(results[0]))
                    warnings.append(f"Lexical search failed: {results[0]}")
                else:
                    lexical_hits = results[0]

                if isinstance(results[1], Exception):
                    logger.warning("knn_search_failed", error=str(results[1]))
                    warnings.append(f"kNN search failed: {results[1]}")
                else:
                    knn_hits = results[1]

            except Exception as exc:
                logger.error("search_failed", error=str(exc))
                warnings.append(f"Search failed: {exc}")

        elif search_mode == "lexical":
            try:
                lexical_hits = await self._store.search_lexical(
                    query_text=request.query.text,
                    filters=search_filters,
                    size=search_size,
                )
            except Exception as exc:
                logger.error("lexical_search_failed", error=str(exc))
                warnings.append(f"Lexical search failed: {exc}")

        elif search_mode == "semantic":
            try:
                knn_hits = await self._store.search_knn(
                    query_vector=query_vector,
                    filters=search_filters,
                    k=search_size,
                )
            except Exception as exc:
                logger.error("knn_search_failed", error=str(exc))
                warnings.append(f"kNN search failed: {exc}")

        search_ms = int((time.monotonic() - search_start) * 1000)
        total_candidates = len(lexical_hits) + len(knn_hits)

        # --- Step 3: Fuse results ---
        if lexical_hits and knn_hits:
            # Full hybrid fusion
            fused = fuse_results(
                lexical_hits=lexical_hits,
                knn_hits=knn_hits,
                rank_constant=self._settings.rank_constant,
                top_k=top_k,
            )
        elif lexical_hits:
            # Lexical only (either by mode or degraded)
            fused = [
                {
                    "chunk_id": h.chunk_id,
                    "fused_score": h.score,
                    "lexical_score": h.score,
                    "semantic_score": None,
                    "hit": h,
                }
                for h in lexical_hits[:top_k]
            ]
        elif knn_hits:
            # Semantic only
            fused = [
                {
                    "chunk_id": h.chunk_id,
                    "fused_score": h.score,
                    "lexical_score": None,
                    "semantic_score": h.score,
                    "hit": h,
                }
                for h in knn_hits[:top_k]
            ]
        else:
            fused = []

        # --- Step 4: Defense-in-depth re-check ---
        safe_hits = self._defense_in_depth_check(
            [f["hit"] for f in fused],
            request.tenant_id,
            allowed_document_models=search_filters.document_model,
        )
        # Rebuild fused list to only include safe hits
        safe_chunk_ids = {h.chunk_id for h in safe_hits}
        fused = [f for f in fused if f["chunk_id"] in safe_chunk_ids]

        # --- Step 5: Build response ---
        chunk_results = []
        for f in fused:
            hit: SearchHit = f["hit"]
            chunk_results.append(
                ChunkResult(
                    chunk_id=hit.chunk_id,
                    version_series_id=hit.version_series_id,
                    document_id=hit.document_id,
                    document_name=hit.document_name,
                    score=round(f["fused_score"], 4),
                    score_breakdown=ScoreBreakdown(
                        lexical=round(f["lexical_score"], 4) if f["lexical_score"] is not None else None,
                        semantic=round(f["semantic_score"], 4) if f["semantic_score"] is not None else None,
                    ),
                    chunk_text=hit.chunk_text,
                    chunk_type=hit.chunk_type,
                    page_no=hit.page_no,
                    heading=hit.heading,
                    section=hit.section,
                    document_uri=hit.document_uri,
                    access_decision="ALLOWED_TENANT_ONLY",
                )
            )

        total_ms = int((time.monotonic() - total_start) * 1000)

        logger.info(
            "retrieval_complete",
            tenant_id=request.tenant_id,
            search_mode=request.query.search_mode,
            results_count=len(chunk_results),
            total_candidates=total_candidates,
            embed_ms=embed_ms,
            search_ms=search_ms,
            total_ms=total_ms,
        )

        return RetrievalResponse(
            request_id=request.request_id,
            tenant_id=request.tenant_id,
            turn_id=request.turn_id,
            results=chunk_results,
            total_candidates_considered=total_candidates,
            latency_ms=LatencyBreakdown(
                embed=embed_ms,
                search=search_ms,
                total=total_ms,
            ),
            warnings=warnings,
        )

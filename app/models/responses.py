"""
Outbound response models — matches Retrieval Service LLD §4.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ScoreBreakdown(BaseModel):
    """Per-leg score for auditability."""

    lexical: Optional[float] = None
    semantic: Optional[float] = None


class ChunkResult(BaseModel):
    """A single retrieved chunk with its metadata and scores."""

    chunk_id: str = Field(alias="chunkId")
    version_series_id: Optional[str] = Field(default=None, alias="versionSeriesId")
    document_id: Optional[str] = Field(default=None, alias="documentId")
    document_name: Optional[str] = Field(default=None, alias="documentName")
    score: float
    score_breakdown: Optional[ScoreBreakdown] = Field(default=None, alias="scoreBreakdown")
    chunk_text: str = Field(alias="chunkText")
    chunk_type: Optional[str] = Field(default=None, alias="chunkType")
    page_no: Optional[int] = Field(default=None, alias="pageNo")
    heading: Optional[str] = None
    section: Optional[str] = None
    document_uri: Optional[str] = Field(default=None, alias="documentUri")
    access_decision: str = Field(
        default="ALLOWED_TENANT_ONLY",
        alias="accessDecision",
        description="Per-result access audit tag (LLD §9).",
    )

    model_config = {"populate_by_name": True}


class LatencyBreakdown(BaseModel):
    """Timing breakdown for observability."""

    embed: int = Field(description="Gemini/Ollama embedding call time in ms.")
    search: int = Field(description="Both ES search legs + app-side fusion time in ms.")
    total: int = Field(description="Total end-to-end retrieval time in ms.")


class RetrievalResponse(BaseModel):
    """
    Response from the Retrieval Service — matches LLD §4.
    Empty results is a valid 200, not a 404.
    """

    request_id: Optional[str] = Field(default=None, alias="requestId")
    tenant_id: str = Field(alias="tenantId")
    turn_id: Optional[str] = Field(default=None, alias="turnId")
    access_policy_version: str = Field(
        default="v0-tenant-only",
        alias="accessPolicyVersion",
    )
    results: List[ChunkResult]
    total_candidates_considered: int = Field(default=0, alias="totalCandidatesConsidered")
    latency_ms: LatencyBreakdown = Field(alias="latencyMs")
    warnings: List[str] = Field(default_factory=list)

    model_config = {"populate_by_name": True}

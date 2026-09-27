"""
Inbound request models — matches Retrieval Service LLD §3.1 / §3.2.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class RequestingUser(BaseModel):
    """Identity of the requesting user — already resolved by RAG Orchestration Service."""

    user_id: str = Field(..., alias="userId")
    roles: Optional[List[str]] = None
    groups: Optional[List[str]] = None

    model_config = {"populate_by_name": True}


class DateRange(BaseModel):
    """Optional date filter for document creation time."""

    from_date: Optional[str] = Field(default=None, alias="from")
    to_date: Optional[str] = Field(default=None, alias="to")

    model_config = {"populate_by_name": True}


class QueryFilters(BaseModel):
    """
    Optional filters applied as ES pre-filters on both search legs.
    Server always ANDs tenantId + safety filters regardless (LLD §3.2).
    """

    document_model: Optional[List[str]] = Field(default=None, alias="documentModel")
    version_series_ids: Optional[List[str]] = Field(default=None, alias="versionSeriesIds")
    legal_hold: Optional[bool] = Field(default=False, alias="legalHold")
    is_deleted: Optional[bool] = Field(default=False, alias="isDeleted")
    date_range: Optional[DateRange] = Field(default=None, alias="dateRange")
    classification: Optional[str] = None

    model_config = {"populate_by_name": True}


class QueryParams(BaseModel):
    """The query block within a retrieval request."""

    text: str = Field(..., min_length=1, description="User query text — must be non-empty.")
    retrieval_unit: str = Field(default="chunk", alias="retrievalUnit")
    top_k: int = Field(default=8, alias="topK", ge=1, le=50)
    search_mode: str = Field(
        default="hybrid",
        alias="searchMode",
        description="'hybrid' (both legs), 'lexical' (BM25 only), 'semantic' (kNN only).",
    )
    filters: Optional[QueryFilters] = None
    rerank: bool = Field(default=False, description="Reserved, no-op in MVP (LLD §11).")

    model_config = {"populate_by_name": True}


class RetrievalRequest(BaseModel):
    """
    Request from RAG Orchestration Service to retrieve relevant chunks.
    Matches LLD §3.1 exactly.
    """

    schema_version: str = Field(default="1.0", alias="schemaVersion")
    request_id: Optional[str] = Field(default=None, alias="requestId")
    tenant_id: str = Field(..., alias="tenantId")
    turn_id: Optional[str] = Field(
        default=None,
        alias="turnId",
        description="Opaque passthrough from IQS — echoed unchanged in response.",
    )
    requesting_user: RequestingUser = Field(..., alias="requestingUser")
    query: QueryParams
    traceparent: Optional[str] = None

    model_config = {"populate_by_name": True}

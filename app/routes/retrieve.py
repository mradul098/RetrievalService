"""
POST /internal/retrieve — the single endpoint for the Retrieval Service.

This is an internal endpoint called service-to-service by RAG Orchestration
Service. It does NOT validate JWTs — identity arrives already-resolved.
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.models.requests import RetrievalRequest
from app.services.retrieval import RetrievalService

logger = structlog.get_logger()

router = APIRouter()


@router.post("/internal/retrieve")
async def retrieve(request_body: RetrievalRequest, request: Request):
    """
    Retrieve relevant chunks for a query.

    Empty results is a valid 200, not a 404 — "no matches" is normal.
    """
    retrieval_service: RetrievalService = request.app.state.retrieval_service

    try:
        response = await retrieval_service.retrieve(request_body)
        return JSONResponse(
            content=response.model_dump(by_alias=True, exclude_none=True),
        )
    except Exception as exc:
        logger.error(
            "retrieval_error",
            tenant_id=request_body.tenant_id,
            request_id=request_body.request_id,
            error=str(exc),
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": True,
                "message": f"Internal retrieval error: {exc}",
                "requestId": request_body.request_id,
            },
        )

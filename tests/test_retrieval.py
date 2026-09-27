"""
Tests for the Hybrid Retrieval Service.

Uses mock embedding client and in-memory stores — no Ollama or FAISS needed.
"""

from __future__ import annotations

import json
from typing import List, Optional
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import Settings
from app.interfaces.embedding_client import EmbeddingClient, EmbeddingError
from app.interfaces.vector_store import SearchFilters, SearchHit, VectorStore
from app.services.fusion import fuse_results
from app.services.retrieval import RetrievalService


# ---------- Mock Embedding Client ----------


class MockEmbeddingClient(EmbeddingClient):
    """Returns a fixed vector — no real embedding needed."""

    def __init__(self, should_fail: bool = False):
        self._should_fail = should_fail

    async def embed(self, text: str) -> List[float]:
        if self._should_fail:
            raise EmbeddingError("Mock embedding failure", provider="mock")
        return [0.1] * 768


# ---------- Mock Vector Store ----------


class MockVectorStore(VectorStore):
    """Returns canned search results."""

    def __init__(
        self,
        lexical_results: Optional[List[SearchHit]] = None,
        knn_results: Optional[List[SearchHit]] = None,
    ):
        self._lexical = lexical_results or []
        self._knn = knn_results or []

    async def search_lexical(
        self, query_text: str, filters: SearchFilters, size: int = 50
    ) -> List[SearchHit]:
        # Apply tenant filter
        return [h for h in self._lexical if h.tenant_id == filters.tenant_id]

    async def search_knn(
        self, query_vector: List[float], filters: SearchFilters, k: int = 50, num_candidates: int = 200
    ) -> List[SearchHit]:
        return [h for h in self._knn if h.tenant_id == filters.tenant_id]


# ---------- Helpers ----------


def _make_hit(chunk_id: str, tenant_id: str = "00103", score: float = 1.0, text: str = "test") -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        score=score,
        chunk_text=text,
        tenant_id=tenant_id,
    )


def _create_test_app(
    embedding_client: EmbeddingClient,
    vector_store: VectorStore,
) -> TestClient:
    from app.routes.retrieve import router

    app = FastAPI()
    app.include_router(router)
    settings = Settings(environment="local")
    app.state.retrieval_service = RetrievalService(
        embedding_client=embedding_client,
        vector_store=vector_store,
        settings=settings,
    )
    return TestClient(app)


VALID_REQUEST = {
    "tenantId": "00103",
    "requestingUser": {"userId": "test@example.com"},
    "query": {
        "text": "total assets Q2 balance sheet",
        "topK": 5,
        "searchMode": "hybrid",
    },
}


# ---------- Fusion Tests ----------


class TestFusion:
    """Tests for the RRF fusion logic."""

    def test_basic_fusion(self):
        lexical = [_make_hit("c1", score=10.0), _make_hit("c2", score=8.0)]
        knn = [_make_hit("c2", score=0.95), _make_hit("c3", score=0.90)]

        results = fuse_results(lexical, knn, rank_constant=20, top_k=5)

        # c2 appears in both lists, so it should have the highest fused score
        assert results[0]["chunk_id"] == "c2"
        assert results[0]["fused_score"] > results[1]["fused_score"]

    def test_fusion_respects_top_k(self):
        lexical = [_make_hit(f"c{i}") for i in range(20)]
        knn = [_make_hit(f"c{i}") for i in range(20)]

        results = fuse_results(lexical, knn, top_k=3)
        assert len(results) == 3

    def test_fusion_single_leg(self):
        lexical = [_make_hit("c1", score=10.0)]
        knn = []

        results = fuse_results(lexical, knn, top_k=5)
        assert len(results) == 1
        assert results[0]["chunk_id"] == "c1"

    def test_fusion_empty(self):
        results = fuse_results([], [], top_k=5)
        assert results == []

    def test_score_breakdown_populated(self):
        lexical = [_make_hit("c1", score=10.0)]
        knn = [_make_hit("c1", score=0.95)]

        results = fuse_results(lexical, knn, top_k=5)
        assert results[0]["lexical_score"] == 10.0
        assert results[0]["semantic_score"] == 0.95


# ---------- Pipeline Tests ----------


class TestRetrievalPipeline:
    """Tests for the end-to-end retrieval pipeline."""

    def test_hybrid_search(self):
        lexical_hits = [_make_hit("c1", score=10.0, text="total assets balance")]
        knn_hits = [_make_hit("c1", score=0.9, text="total assets balance")]

        client = _create_test_app(
            MockEmbeddingClient(),
            MockVectorStore(lexical_results=lexical_hits, knn_results=knn_hits),
        )
        response = client.post("/internal/retrieve", json=VALID_REQUEST)

        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 1
        assert data["tenantId"] == "00103"
        assert data["results"][0]["chunkId"] == "c1"
        assert data["results"][0]["accessDecision"] == "ALLOWED_TENANT_ONLY"

    def test_empty_results_is_200(self):
        client = _create_test_app(MockEmbeddingClient(), MockVectorStore())
        response = client.post("/internal/retrieve", json=VALID_REQUEST)

        assert response.status_code == 200
        data = response.json()
        assert data["results"] == []

    def test_turn_id_passthrough(self):
        request = {**VALID_REQUEST, "turnId": "turn-abc-123"}
        client = _create_test_app(MockEmbeddingClient(), MockVectorStore())
        response = client.post("/internal/retrieve", json=request)

        assert response.status_code == 200
        assert response.json()["turnId"] == "turn-abc-123"

    def test_embedding_failure_degrades_to_lexical(self):
        lexical_hits = [_make_hit("c1", score=5.0, text="some text")]

        client = _create_test_app(
            MockEmbeddingClient(should_fail=True),
            MockVectorStore(lexical_results=lexical_hits),
        )
        response = client.post("/internal/retrieve", json=VALID_REQUEST)

        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 1
        assert len(data["warnings"]) > 0
        assert "Embedding failed" in data["warnings"][0]

    def test_tenant_isolation(self):
        """A hit from another tenant should be filtered out."""
        hits = [
            _make_hit("c1", tenant_id="00103", score=10.0, text="correct tenant"),
            _make_hit("c2", tenant_id="00999", score=20.0, text="wrong tenant"),
        ]

        client = _create_test_app(
            MockEmbeddingClient(),
            MockVectorStore(lexical_results=hits, knn_results=hits),
        )
        response = client.post("/internal/retrieve", json=VALID_REQUEST)

        assert response.status_code == 200
        data = response.json()
        chunk_ids = [r["chunkId"] for r in data["results"]]
        assert "c1" in chunk_ids
        assert "c2" not in chunk_ids  # wrong tenant, filtered

    def test_lexical_only_mode(self):
        request = {**VALID_REQUEST, "query": {**VALID_REQUEST["query"], "searchMode": "lexical"}}
        lexical_hits = [_make_hit("c1", score=5.0, text="text")]

        client = _create_test_app(
            MockEmbeddingClient(),
            MockVectorStore(lexical_results=lexical_hits),
        )
        response = client.post("/internal/retrieve", json=request)

        assert response.status_code == 200
        assert len(response.json()["results"]) == 1

    def test_latency_breakdown_present(self):
        client = _create_test_app(MockEmbeddingClient(), MockVectorStore())
        response = client.post("/internal/retrieve", json=VALID_REQUEST)

        data = response.json()
        assert "latencyMs" in data
        assert "embed" in data["latencyMs"]
        assert "search" in data["latencyMs"]
        assert "total" in data["latencyMs"]


# ---------- Validation Tests ----------


class TestValidation:
    def test_missing_tenant_id(self):
        client = _create_test_app(MockEmbeddingClient(), MockVectorStore())
        bad_request = {
            "requestingUser": {"userId": "test@example.com"},
            "query": {"text": "test"},
        }
        response = client.post("/internal/retrieve", json=bad_request)
        assert response.status_code == 422

    def test_empty_query_text(self):
        client = _create_test_app(MockEmbeddingClient(), MockVectorStore())
        bad_request = {
            "tenantId": "00103",
            "requestingUser": {"userId": "test@example.com"},
            "query": {"text": ""},
        }
        response = client.post("/internal/retrieve", json=bad_request)
        assert response.status_code == 422

    def test_top_k_exceeds_max(self):
        client = _create_test_app(MockEmbeddingClient(), MockVectorStore())
        bad_request = {
            "tenantId": "00103",
            "requestingUser": {"userId": "test@example.com"},
            "query": {"text": "test", "topK": 999},
        }
        response = client.post("/internal/retrieve", json=bad_request)
        assert response.status_code == 422

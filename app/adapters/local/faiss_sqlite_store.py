"""
Local vector store: FAISS (kNN) + SQLite FTS5 (BM25 lexical search).

Replaces Elasticsearch for local development. No external dependencies
beyond the local filesystem.

Data layout:
  - data/faiss.index  — FAISS flat index (IndexFlatIP for cosine similarity)
  - data/chunks.db    — SQLite database with:
      - `chunks` table (all metadata)
      - `chunks_fts` FTS5 virtual table on chunk_text for BM25 ranking
      - `chunk_index_map` table mapping FAISS index position → chunk_id
"""

from __future__ import annotations

import os
import sqlite3
from typing import List, Optional

import numpy as np

from app.interfaces.vector_store import SearchFilters, SearchHit, VectorStore


class FaissSqliteStore(VectorStore):
    """
    Local vector store using FAISS for kNN and SQLite FTS5 for BM25.

    FAISS index and SQLite DB are loaded from disk at init time.
    The seed script (seed_local.py) creates both.
    """

    def __init__(self, faiss_index_path: str, sqlite_db_path: str):
        self._faiss_index_path = faiss_index_path
        self._sqlite_db_path = sqlite_db_path
        self._index = None  # Lazy-loaded
        self._index_map: List[str] = []  # position → chunk_id

    def _ensure_loaded(self) -> None:
        """Lazy-load FAISS index and chunk_index_map from disk."""
        if self._index is not None:
            return

        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError(
                "faiss-cpu is required. Install with: pip install faiss-cpu"
            ) from exc

        if not os.path.exists(self._faiss_index_path):
            raise FileNotFoundError(
                f"FAISS index not found at {self._faiss_index_path}. "
                "Run 'python -m app.scripts.seed_local' first."
            )

        self._index = faiss.read_index(self._faiss_index_path)

        # Load the index map (FAISS position → chunk_id)
        conn = sqlite3.connect(self._sqlite_db_path)
        try:
            cursor = conn.execute(
                "SELECT chunk_id FROM chunk_index_map ORDER BY faiss_position"
            )
            self._index_map = [row[0] for row in cursor.fetchall()]
        finally:
            conn.close()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a SQLite connection with row factory."""
        conn = sqlite3.connect(self._sqlite_db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _build_where_clause(self, filters: SearchFilters) -> tuple:
        """Build WHERE clause + params from SearchFilters."""
        conditions = ["tenant_id = ?"]
        params: list = [filters.tenant_id]

        # Safety filters — always applied
        conditions.append("is_deleted = ?")
        params.append(1 if filters.is_deleted else 0)

        conditions.append("legal_hold = ?")
        params.append(1 if filters.legal_hold else 0)

        if filters.document_model:
            placeholders = ",".join("?" for _ in filters.document_model)
            conditions.append(f"document_model IN ({placeholders})")
            params.extend(filters.document_model)

        if filters.version_series_ids:
            placeholders = ",".join("?" for _ in filters.version_series_ids)
            conditions.append(f"version_series_id IN ({placeholders})")
            params.extend(filters.version_series_ids)

        if filters.classification:
            conditions.append("classification = ?")
            params.append(filters.classification)

        return " AND ".join(conditions), params

    def _row_to_hit(self, row: sqlite3.Row, score: float) -> SearchHit:
        """Convert a SQLite row to a SearchHit."""
        return SearchHit(
            chunk_id=row["chunk_id"],
            score=score,
            chunk_text=row["chunk_text"],
            tenant_id=row["tenant_id"],
            document_id=row["document_id"],
            version_series_id=row["version_series_id"],
            document_name=row["document_name"],
            chunk_type=row["chunk_type"],
            page_no=row["page_no"],
            heading=row["heading"],
            section=row["section"],
            document_model=row["document_model"],
            is_deleted=bool(row["is_deleted"]),
            legal_hold=bool(row["legal_hold"]),
            classification=row["classification"],
        )

    async def search_lexical(
        self,
        query_text: str,
        filters: SearchFilters,
        size: int = 50,
    ) -> List[SearchHit]:
        """
        BM25 search via SQLite FTS5.

        Uses the `chunks_fts` virtual table with the MATCH operator.
        The FTS5 `bm25()` function returns negative scores (more negative = better),
        so we negate them for consistency with kNN scores.
        """
        where_clause, params = self._build_where_clause(filters)

        # Tokenize query for FTS5: strip non-alphanumeric chars, join with OR
        import re
        words = re.findall(r'\w+', query_text)
        if not words:
            return []
        # Quote each word and join with OR for broad matching
        fts_query = " OR ".join(f'"{w}"' for w in words)

        conn = self._get_connection()
        try:
            # FTS5 MATCH query joined with chunks table for filtering
            sql = f"""
                SELECT c.*, -chunks_fts.rank AS fts_score
                FROM chunks_fts
                JOIN chunks c ON c.chunk_id = chunks_fts.chunk_id
                WHERE chunks_fts MATCH ?
                  AND {where_clause}
                ORDER BY chunks_fts.rank
                LIMIT ?
            """
            cursor = conn.execute(sql, [fts_query] + params + [size])
            rows = cursor.fetchall()

            results = []
            for row in rows:
                score = float(row["fts_score"])
                results.append(self._row_to_hit(row, score))

            return results
        except Exception:
            # If FTS match fails (e.g. bad query syntax), return empty
            return []
        finally:
            conn.close()

    async def search_knn(
        self,
        query_vector: List[float],
        filters: SearchFilters,
        k: int = 50,
        num_candidates: int = 200,
    ) -> List[SearchHit]:
        """
        kNN search via FAISS.

        Since FAISS doesn't support pre-filtering, we fetch a wider
        candidate set and post-filter. This is acceptable for local
        testing with small datasets. In production, Elasticsearch
        handles pre-filtering natively.
        """
        self._ensure_loaded()

        if self._index is None or self._index.ntotal == 0:
            return []

        # Search wider to account for post-filtering
        search_k = min(num_candidates, self._index.ntotal)
        query_np = np.array([query_vector], dtype=np.float32)

        # Normalize for cosine similarity (IndexFlatIP)
        norm = np.linalg.norm(query_np)
        if norm > 0:
            query_np = query_np / norm

        scores, indices = self._index.search(query_np, search_k)

        # Map FAISS indices to chunk_ids and look up metadata
        candidate_ids = []
        score_map = {}
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._index_map):
                continue
            chunk_id = self._index_map[idx]
            candidate_ids.append(chunk_id)
            score_map[chunk_id] = float(score)

        if not candidate_ids:
            return []

        # Fetch metadata + apply filters in SQL
        where_clause, params = self._build_where_clause(filters)
        placeholders = ",".join("?" for _ in candidate_ids)

        conn = self._get_connection()
        try:
            sql = f"""
                SELECT * FROM chunks
                WHERE chunk_id IN ({placeholders})
                  AND {where_clause}
            """
            cursor = conn.execute(sql, candidate_ids + params)
            rows = cursor.fetchall()

            results = []
            for row in rows:
                chunk_id = row["chunk_id"]
                score = score_map.get(chunk_id, 0.0)
                results.append(self._row_to_hit(row, score))

            # Sort by score descending and limit to k
            results.sort(key=lambda h: h.score, reverse=True)
            return results[:k]
        finally:
            conn.close()

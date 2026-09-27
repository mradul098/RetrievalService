"""
Application-side Reciprocal Rank Fusion (RRF) — matches LLD §6.1 Step 3.

Since the licensing status of Elasticsearch's native `rrf` retriever
is not confirmed for Community/Basic edition, this service implements
the same logic in application code.

The formula: for each chunk across both result sets,
  fused_score = sum( 1 / (rank_constant + rank + 1) )
  where rank is the 0-based position in each result list.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from app.interfaces.vector_store import SearchHit


def fuse_results(
    lexical_hits: List[SearchHit],
    knn_hits: List[SearchHit],
    rank_constant: int = 20,
    top_k: int = 8,
) -> List[dict]:
    """
    Fuse two ranked hit lists using Reciprocal Rank Fusion.

    Args:
        lexical_hits: Ranked results from BM25/lexical search leg.
        knn_hits: Ranked results from kNN/vector search leg.
        rank_constant: The `k` constant in 1/(k+rank+1). Default 20
                       (matches the typical ES rrf default of 60, but
                       our candidate sets are smaller).
        top_k: Number of final results to return.

    Returns:
        List of dicts with:
          - chunk_id
          - fused_score
          - lexical_score (raw score from lexical leg, or None)
          - semantic_score (raw score from kNN leg, or None)
          - hit (the full SearchHit object, from whichever leg had it)
    """
    # Track scores and source hits
    fused_scores: Dict[str, float] = {}
    lexical_scores: Dict[str, float] = {}
    semantic_scores: Dict[str, float] = {}
    hit_lookup: Dict[str, SearchHit] = {}

    # Score from lexical leg
    for rank, hit in enumerate(lexical_hits):
        rrf_contribution = 1.0 / (rank_constant + rank + 1)
        fused_scores[hit.chunk_id] = fused_scores.get(hit.chunk_id, 0.0) + rrf_contribution
        lexical_scores[hit.chunk_id] = hit.score
        hit_lookup[hit.chunk_id] = hit

    # Score from kNN leg
    for rank, hit in enumerate(knn_hits):
        rrf_contribution = 1.0 / (rank_constant + rank + 1)
        fused_scores[hit.chunk_id] = fused_scores.get(hit.chunk_id, 0.0) + rrf_contribution
        semantic_scores[hit.chunk_id] = hit.score
        # Prefer kNN hit object if available (it has the vector-search score)
        if hit.chunk_id not in hit_lookup:
            hit_lookup[hit.chunk_id] = hit

    # Sort by fused score descending, take top_k
    sorted_ids = sorted(fused_scores.keys(), key=lambda cid: fused_scores[cid], reverse=True)
    top_ids = sorted_ids[:top_k]

    results = []
    for chunk_id in top_ids:
        results.append({
            "chunk_id": chunk_id,
            "fused_score": fused_scores[chunk_id],
            "lexical_score": lexical_scores.get(chunk_id),
            "semantic_score": semantic_scores.get(chunk_id),
            "hit": hit_lookup[chunk_id],
        })

    return results

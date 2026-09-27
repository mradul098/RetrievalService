"""
Seed script — populates the local FAISS index + SQLite database
with sample financial document chunks for testing.

Usage:
  ENVIRONMENT=local python -m app.scripts.seed_local

This creates:
  data/faiss.index  — FAISS IndexFlatIP with normalized vectors
  data/chunks.db    — SQLite with chunks table, FTS5 index, and chunk_index_map
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
import sys
from typing import List

import numpy as np

# Sample chunks simulating a financial document corpus
SAMPLE_CHUNKS = [
    {
        "chunk_id": "DOC001#chunk-1",
        "tenant_id": "00103",
        "document_id": "DOC001",
        "version_series_id": "VS001",
        "document_name": "Q2_2026_Balance_Sheet.pdf",
        "chunk_text": "Total assets for the quarter ended June 30, 2026 were $482.3M, an increase of 8.2% from the prior quarter. Current assets comprised $198.7M in cash and cash equivalents, $45.2M in accounts receivable, and $12.8M in prepaid expenses.",
        "chunk_type": "table",
        "page_no": 3,
        "heading": "Balance Sheet",
        "section": "Balance Sheet",
        "document_model": "sampledocs",
        "is_deleted": False,
        "legal_hold": False,
        "classification": None,
    },
    {
        "chunk_id": "DOC001#chunk-2",
        "tenant_id": "00103",
        "document_id": "DOC001",
        "version_series_id": "VS001",
        "document_name": "Q2_2026_Balance_Sheet.pdf",
        "chunk_text": "Total liabilities stood at $215.6M, with long-term debt of $120.3M and current liabilities of $95.3M. The debt-to-equity ratio improved to 0.81 from 0.89 in Q1 2026.",
        "chunk_type": "paragraph",
        "page_no": 4,
        "heading": "Liabilities",
        "section": "Balance Sheet",
        "document_model": "sampledocs",
        "is_deleted": False,
        "legal_hold": False,
        "classification": None,
    },
    {
        "chunk_id": "DOC001#chunk-3",
        "tenant_id": "00103",
        "document_id": "DOC001",
        "version_series_id": "VS001",
        "document_name": "Q2_2026_Balance_Sheet.pdf",
        "chunk_text": "Shareholders' equity totaled $266.7M. Retained earnings increased by $18.4M during the quarter, reflecting strong net income performance. Book value per share rose to $14.82.",
        "chunk_type": "paragraph",
        "page_no": 5,
        "heading": "Shareholders Equity",
        "section": "Balance Sheet",
        "document_model": "sampledocs",
        "is_deleted": False,
        "legal_hold": False,
        "classification": None,
    },
    {
        "chunk_id": "DOC002#chunk-1",
        "tenant_id": "00103",
        "document_id": "DOC002",
        "version_series_id": "VS002",
        "document_name": "Q2_2026_Income_Statement.pdf",
        "chunk_text": "Revenue for Q2 2026 was $156.8M, up 12.3% year-over-year. Cost of goods sold was $89.2M, resulting in a gross margin of 43.1%. Operating expenses included $22.4M in R&D and $18.7M in sales and marketing.",
        "chunk_type": "table",
        "page_no": 2,
        "heading": "Income Statement Summary",
        "section": "Income Statement",
        "document_model": "sampledocs",
        "is_deleted": False,
        "legal_hold": False,
        "classification": None,
    },
    {
        "chunk_id": "DOC002#chunk-2",
        "tenant_id": "00103",
        "document_id": "DOC002",
        "version_series_id": "VS002",
        "document_name": "Q2_2026_Income_Statement.pdf",
        "chunk_text": "EBITDA for Q2 2026 was $42.1M with an EBITDA margin of 26.9%. Net income attributable to shareholders was $24.3M, or $1.35 per diluted share. This represents a 15.2% increase compared to Q2 2025.",
        "chunk_type": "paragraph",
        "page_no": 3,
        "heading": "Profitability Metrics",
        "section": "Income Statement",
        "document_model": "sampledocs",
        "is_deleted": False,
        "legal_hold": False,
        "classification": None,
    },
    {
        "chunk_id": "DOC003#chunk-1",
        "tenant_id": "00103",
        "document_id": "DOC003",
        "version_series_id": "VS003",
        "document_name": "Q2_2026_Cash_Flow.pdf",
        "chunk_text": "Operating cash flow for Q2 2026 was $38.9M. Capital expenditures totaled $12.1M, primarily related to data center expansion and equipment upgrades. Free cash flow was $26.8M.",
        "chunk_type": "paragraph",
        "page_no": 2,
        "heading": "Cash Flow Summary",
        "section": "Cash Flow Statement",
        "document_model": "sampledocs",
        "is_deleted": False,
        "legal_hold": False,
        "classification": None,
    },
    {
        "chunk_id": "DOC003#chunk-2",
        "tenant_id": "00103",
        "document_id": "DOC003",
        "version_series_id": "VS003",
        "document_name": "Q2_2026_Cash_Flow.pdf",
        "chunk_text": "The company repurchased 1.2 million shares for $18.6M during Q2 2026 under its existing share buyback program. Dividend payments of $5.4M were made to shareholders. Net change in cash position was positive $2.8M.",
        "chunk_type": "paragraph",
        "page_no": 3,
        "heading": "Financing Activities",
        "section": "Cash Flow Statement",
        "document_model": "sampledocs",
        "is_deleted": False,
        "legal_hold": False,
        "classification": None,
    },
    {
        "chunk_id": "DOC004#chunk-1",
        "tenant_id": "00103",
        "document_id": "DOC004",
        "version_series_id": "VS004",
        "document_name": "Annual_Risk_Assessment_2026.pdf",
        "chunk_text": "Key risk factors include exposure to foreign currency fluctuations, particularly EUR/USD and GBP/USD. The company maintains hedging positions covering approximately 60% of anticipated foreign currency revenues for the next 12 months.",
        "chunk_type": "paragraph",
        "page_no": 8,
        "heading": "Currency Risk",
        "section": "Risk Factors",
        "document_model": "sampledocs",
        "is_deleted": False,
        "legal_hold": False,
        "classification": None,
    },
    {
        "chunk_id": "DOC004#chunk-2",
        "tenant_id": "00103",
        "document_id": "DOC004",
        "version_series_id": "VS004",
        "document_name": "Annual_Risk_Assessment_2026.pdf",
        "chunk_text": "Regulatory compliance costs are projected to increase by 15-20% in FY2027 due to new data privacy regulations in the EU and APAC regions. The company has allocated $8.2M for compliance infrastructure upgrades.",
        "chunk_type": "paragraph",
        "page_no": 12,
        "heading": "Regulatory Risk",
        "section": "Risk Factors",
        "document_model": "sampledocs",
        "is_deleted": False,
        "legal_hold": False,
        "classification": None,
    },
    {
        "chunk_id": "DOC005#chunk-1",
        "tenant_id": "00103",
        "document_id": "DOC005",
        "version_series_id": "VS005",
        "document_name": "Q3_2026_Balance_Sheet.pdf",
        "chunk_text": "Total assets for Q3 2026 reached $501.8M, a 4.0% increase from Q2 2026. The growth was primarily driven by a $12.5M increase in property, plant and equipment related to the new Sydney data center.",
        "chunk_type": "table",
        "page_no": 3,
        "heading": "Balance Sheet",
        "section": "Balance Sheet",
        "document_model": "sampledocs",
        "is_deleted": False,
        "legal_hold": False,
        "classification": None,
    },
    # A chunk from a different tenant (should be filtered out)
    {
        "chunk_id": "DOC006#chunk-1",
        "tenant_id": "00999",
        "document_id": "DOC006",
        "version_series_id": "VS006",
        "document_name": "Other_Tenant_Doc.pdf",
        "chunk_text": "This is a document from a completely different tenant. Total assets were $999M. It should never appear in results for tenant 00103.",
        "chunk_type": "paragraph",
        "page_no": 1,
        "heading": "Summary",
        "section": "Summary",
        "document_model": "sampledocs",
        "is_deleted": False,
        "legal_hold": False,
        "classification": None,
    },
    # A deleted chunk (should be filtered out)
    {
        "chunk_id": "DOC007#chunk-1",
        "tenant_id": "00103",
        "document_id": "DOC007",
        "version_series_id": "VS007",
        "document_name": "Deleted_Report.pdf",
        "chunk_text": "This report has been marked as deleted and should not appear in search results. Total assets mentioned here are stale.",
        "chunk_type": "paragraph",
        "page_no": 1,
        "heading": "Deleted Report",
        "section": "Summary",
        "document_model": "sampledocs",
        "is_deleted": True,
        "legal_hold": False,
        "classification": None,
    },
]


def create_sqlite_schema(conn: sqlite3.Connection) -> None:
    """Create the chunks table, FTS5 virtual table, and chunk_index_map."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            document_id TEXT,
            version_series_id TEXT,
            document_name TEXT,
            chunk_text TEXT NOT NULL,
            chunk_type TEXT,
            page_no INTEGER,
            heading TEXT,
            section TEXT,
            document_model TEXT,
            is_deleted INTEGER DEFAULT 0,
            legal_hold INTEGER DEFAULT 0,
            classification TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            chunk_id,
            chunk_text,
            content='chunks',
            content_rowid='rowid'
        );

        CREATE TABLE IF NOT EXISTS chunk_index_map (
            faiss_position INTEGER PRIMARY KEY,
            chunk_id TEXT NOT NULL
        );
    """)


def insert_chunks(conn: sqlite3.Connection, chunks: list) -> None:
    """Insert sample chunks into the chunks table and FTS5 index."""
    for chunk in chunks:
        conn.execute(
            """INSERT OR REPLACE INTO chunks
               (chunk_id, tenant_id, document_id, version_series_id,
                document_name, chunk_text, chunk_type, page_no,
                heading, section, document_model, is_deleted, legal_hold, classification)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                chunk["chunk_id"],
                chunk["tenant_id"],
                chunk["document_id"],
                chunk["version_series_id"],
                chunk["document_name"],
                chunk["chunk_text"],
                chunk["chunk_type"],
                chunk["page_no"],
                chunk["heading"],
                chunk["section"],
                chunk["document_model"],
                1 if chunk["is_deleted"] else 0,
                1 if chunk["legal_hold"] else 0,
                chunk.get("classification"),
            ),
        )

        # Insert into FTS5 index
        conn.execute(
            "INSERT OR REPLACE INTO chunks_fts (chunk_id, chunk_text) VALUES (?, ?)",
            (chunk["chunk_id"], chunk["chunk_text"]),
        )

    conn.commit()


async def embed_chunks(chunks: list) -> List[List[float]]:
    """Embed all chunk texts via Ollama."""
    from app.adapters.local.ollama_embedding import OllamaEmbeddingClient
    from app.config import get_settings

    settings = get_settings()
    client = OllamaEmbeddingClient(
        base_url=settings.ollama_base_url,
        model=settings.ollama_embedding_model,
    )

    embeddings = []
    for i, chunk in enumerate(chunks):
        print(f"  Embedding chunk {i + 1}/{len(chunks)}: {chunk['chunk_id']}")
        vector = await client.embed(chunk["chunk_text"])
        embeddings.append(vector)

    return embeddings


def build_faiss_index(
    embeddings: List[List[float]], chunk_ids: List[str], faiss_path: str, db_path: str
) -> None:
    """Build FAISS IndexFlatIP and save the index + chunk_index_map."""
    import faiss

    # Convert to numpy, normalize for cosine similarity
    vectors = np.array(embeddings, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1  # avoid division by zero
    vectors = vectors / norms

    dims = vectors.shape[1]
    index = faiss.IndexFlatIP(dims)  # Inner product on normalized vectors = cosine similarity
    index.add(vectors)
    faiss.write_index(index, faiss_path)

    # Save chunk_index_map
    conn = sqlite3.connect(db_path)
    conn.execute("DELETE FROM chunk_index_map")
    for pos, chunk_id in enumerate(chunk_ids):
        conn.execute(
            "INSERT INTO chunk_index_map (faiss_position, chunk_id) VALUES (?, ?)",
            (pos, chunk_id),
        )
    conn.commit()
    conn.close()

    print(f"  FAISS index: {index.ntotal} vectors, {dims} dims → {faiss_path}")


async def main():
    """Main seed function."""
    print("=" * 60)
    print("Hybrid Retrieval Service — Local Data Seed")
    print("=" * 60)

    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    db_path = "data/chunks.db"
    faiss_path = "data/faiss.index"

    # Step 0: Remove old DB to prevent FTS5 rowid mismatch
    if os.path.exists(db_path):
        os.remove(db_path)

    # Step 1: Create SQLite schema and insert chunks
    print("\n1. Creating SQLite database + FTS5 index...")
    conn = sqlite3.connect(db_path)
    create_sqlite_schema(conn)
    insert_chunks(conn, SAMPLE_CHUNKS)
    conn.close()
    print(f"   Inserted {len(SAMPLE_CHUNKS)} chunks into {db_path}")

    # Step 2: Embed all chunks via Ollama
    print("\n2. Embedding chunks via Ollama...")
    print("   (Make sure Ollama is running: ollama serve)")
    embeddings = await embed_chunks(SAMPLE_CHUNKS)
    print(f"   Got {len(embeddings)} embeddings, dims={len(embeddings[0])}")

    # Step 3: Build FAISS index
    print("\n3. Building FAISS index...")
    chunk_ids = [c["chunk_id"] for c in SAMPLE_CHUNKS]
    build_faiss_index(embeddings, chunk_ids, faiss_path, db_path)

    print("\n" + "=" * 60)
    print("Seed complete! You can now start the retrieval service:")
    print("  ENVIRONMENT=local uvicorn app.main:app --port 8002")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

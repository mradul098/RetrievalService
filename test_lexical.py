import asyncio
from app.adapters.local.faiss_sqlite_store import FaissSqliteStore
from app.interfaces.vector_store import SearchFilters

async def main():
    store = FaissSqliteStore("data/faiss.index", "data/chunks.db")
    filters = SearchFilters(tenant_id="00103")
    
    where_clause, params = store._build_where_clause(filters)
    query_text = "total assets"
    
    import re
    words = re.findall(r'\w+', query_text)
    fts_query = " OR ".join(f'"{w}"' for w in words)
    print("FTS Query:", fts_query)
    
    sql = f"""
        SELECT c.*, -chunks_fts.rank AS fts_score
        FROM chunks_fts
        JOIN chunks c ON c.chunk_id = chunks_fts.chunk_id
        WHERE chunks_fts MATCH ?
          AND {where_clause}
        ORDER BY chunks_fts.rank
        LIMIT ?
    """
    print("SQL:", sql)
    print("Params:", [fts_query] + params + [50])
    
    conn = store._get_connection()
    cursor = conn.execute(sql, [fts_query] + params + [50])
    rows = cursor.fetchall()
    print("Rows:", rows)

asyncio.run(main())

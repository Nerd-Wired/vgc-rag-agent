"""
Online Hybrid Retriever module for the VGC RAG Agent.
Uses psycopg_pool for connection management and Reciprocal Rank Fusion (RRF)
in PostgreSQL to merge dense vector similarity and lexical BM25 scores.
"""
import os
from typing import Any, Dict, List
from dotenv import load_dotenv
import numpy as np
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RRF_CONSTANT = 60

# 1. Initialize Global Resources (Loaded once at server startup)
if not DATABASE_URL:
    raise ValueError("DATABASE_URL is missing from environment variables.")

print(f"[{__name__}] Initializing embedding model: {EMBEDDING_MODEL}...")
embedder = SentenceTransformer(EMBEDDING_MODEL)

# Define the adapter registration callback BEFORE creating the pool
def _configure_connection(conn):
    register_vector(conn)

print(f"[{__name__}] Initializing PostgreSQL connection pool...")
# Pass the configure callback directly into the constructor
pool = ConnectionPool(
    conninfo=DATABASE_URL,
    min_size=1,
    max_size=10,
    configure=_configure_connection, 
    kwargs={"autocommit": True}
)


def retrieve(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Executes a hybrid search (Dense HNSW + Lexical GIN) using RRF ranking.
    
    Args:
        query (str): The user's natural language question.
        top_k (int): Number of top documents to return after fusion.
        
    Returns:
        List[Dict[str, Any]]: Ranked list of metadata and text chunks.
    """
    if not query.strip():
        return []

    # 1. Generate query embedding (384-dimensional unit-normalized vector)
    query_vector = embedder.encode(query, normalize_embeddings=True)

    # 2. RRF SQL Query using Common Table Expressions (CTEs)
    # <#> is the negative inner product operator (ideal for normalized vectors in pgvector)
    rrf_sql = """
        WITH dense_search AS (
            SELECT id, category, title, text,
                   ROW_NUMBER() OVER (ORDER BY embedding <#> %s ASC) AS dense_rank
            FROM vgc_knowledge_base
            LIMIT 20
        ),
        lexical_search AS (
            SELECT id, category, title, text,
                   ROW_NUMBER() OVER (
                       ORDER BY ts_rank_cd(search_vector, plainto_tsquery('english', %s)) DESC
                   ) AS lexical_rank
            FROM vgc_knowledge_base
            WHERE search_vector @@ plainto_tsquery('english', %s)
            LIMIT 20
        )
        SELECT 
            COALESCE(d.id, l.id) AS id,
            COALESCE(d.category, l.category) AS category,
            COALESCE(d.title, l.title) AS title,
            COALESCE(d.text, l.text) AS text,
            (
                COALESCE(1.0 / (%s + d.dense_rank), 0.0) + 
                COALESCE(1.0 / (%s + l.lexical_rank), 0.0)
            ) AS rrf_score
        FROM dense_search d
        FULL OUTER JOIN lexical_search l ON d.id = l.id
        ORDER BY rrf_score DESC
        LIMIT %s;
    """

    results = []
    
    # 3. Check out a warm connection from the pool and execute
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                rrf_sql, 
                (
                    query_vector,       # For dense_search CTE
                    query,              # For lexical_search ranking
                    query,              # For lexical_search WHERE filter
                    RRF_CONSTANT,       # For RRF math (dense)
                    RRF_CONSTANT,       # For RRF math (lexical)
                    top_k               # For final limit
                )
            )
            rows = cur.fetchall()

            for row in rows:
                results.append({
                    "id": row[0],
                    "category": row[1],
                    "title": row[2],
                    "text": row[3],
                    "score": float(row[4])
                })

    return results


# Quick execution block for local debugging
if __name__ == "__main__":
    test_query = "How do I counter Dondozo and Tatsugiri cores?"
    print(f"\nRunning test query: '{test_query}'...")
    docs = retrieve(test_query, top_k=3)
    
    for idx, doc in enumerate(docs, 1):
        print(f"\n--- Rank {idx} (RRF Score: {doc['score']:.4f}) ---")
        print(f"Title: {doc['title']} [{doc['category']}]")
        print(f"Text: {doc['text'][:150]}...")
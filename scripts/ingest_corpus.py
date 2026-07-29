"""
Offline batch ingestion pipeline for VGC RAG knowledge base.
Reads data/corpus.json, computes embeddings in batches, and performs
idempotent UPSERTs into PostgreSQL/Supabase via psycopg3.
"""
import json
import os
from pathlib import Path

import numpy as np
import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

# Load environment variables from .env
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
CORPUS_PATH = Path("data/corpus.json")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 32


def load_chunks(file_path: Path) -> list[dict]:
    if not file_path.exists():
        raise FileNotFoundError(f"Corpus file not found at {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def ingest():
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL is not set in environment variables.")

    print(f"Loading corpus from {CORPUS_PATH}...")
    chunks = load_chunks(CORPUS_PATH)
    print(f"Loaded {len(chunks)} chunks.")

    # 1. Initialize Embedder
    print(f"Loading embedding model ({EMBEDDING_MODEL})...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    # 2. Extract texts and embed in batches
    # Combining title and text gives the dense vector semantic context of the header
    texts_to_embed = [f"{c['title']}. {c['text']}" for c in chunks]
    print("Generating embeddings in batches...")
    embeddings = embedder.encode(
        texts_to_embed, 
        batch_size=BATCH_SIZE, 
        normalize_embeddings=True, 
        show_progress_bar=True
    )

    # 3. Prepare data tuples for bulk database insertion
    records = []
    for i, chunk in enumerate(chunks):
        records.append((
            chunk["id"],
            chunk["category"],
            chunk["title"],
            chunk["text"],
            embeddings[i]  # Unit-normalized NumPy array of shape (384,)
        ))

    # 4. Idempotent UPSERT Query (skipping generated search_vector)
    upsert_query = """
        INSERT INTO vgc_knowledge_base (id, category, title, text, embedding)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (id) DO UPDATE SET
            category = EXCLUDED.category,
            title = EXCLUDED.title,
            text = EXCLUDED.text,
            embedding = EXCLUDED.embedding;
    """

    print("Connecting to Supabase...")
    # Open database connection
    with psycopg.connect(DATABASE_URL) as conn:
        # Register pgvector type adapter for this connection
        register_vector(conn)
        
        with conn.cursor() as cur:
            print(f"Executing bulk UPSERT for {len(records)} records...")
            cur.executemany(upsert_query, records)
        
        conn.commit()
    print("Ingestion complete!")


if __name__ == "__main__":
    ingest()
-- Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Create table with full-text search and vector columns
CREATE TABLE IF NOT EXISTS vgc_knowledge_base (
    id TEXT PRIMARY KEY,
    category TEXT NOT NULL,
    title TEXT NOT NULL,
    text TEXT NOT NULL,
    
    -- Auto-generated English tokens for BM25/Lexical Search
    search_vector TSVECTOR GENERATED ALWAYS AS (
        to_tsvector('english', coalesce(title, '') || ' ' || coalesce(text, ''))
    ) STORED,
    
    -- 384-dimensional dense vector
    embedding VECTOR(384)
);

-- Index for Fast Lexical Search
CREATE INDEX IF NOT EXISTS idx_vgc_lexical 
ON vgc_knowledge_base 
USING GIN (search_vector);

-- Index for Fast Dense Search (using inner product for normalized vectors)
CREATE INDEX IF NOT EXISTS idx_vgc_dense 
ON vgc_knowledge_base 
USING hnsw (embedding vector_ip_ops) 
WITH (m = 16, ef_construction = 64);
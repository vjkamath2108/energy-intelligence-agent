-- Document chunk store with pgvector embedding column.
-- Run once after CREATE EXTENSION IF NOT EXISTS vector;
-- Embeddings (vector(1024)) are populated in W2.1 by the embedding pipeline.

CREATE TABLE IF NOT EXISTS document_chunks (
    doc_id           TEXT    PRIMARY KEY,
    content          TEXT    NOT NULL,
    source           TEXT    NOT NULL,
    chunk_index      INTEGER NOT NULL,
    similarity_score FLOAT   DEFAULT 0.0,
    embedding        vector(1024),          -- populated later in W2.1
    metadata         JSONB   DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_chunks_source
    ON document_chunks (source);

CREATE INDEX IF NOT EXISTS idx_chunks_source_chunk
    ON document_chunks (source, chunk_index);

-- Placeholder IVFFlat index — rebuild with correct lists value after embeddings are loaded.
-- DO NOT use until embeddings are populated (W2.1).
-- CREATE INDEX idx_chunks_embedding
--     ON document_chunks USING ivfflat (embedding vector_cosine_ops)
--     WITH (lists = 100);

-- BM25 full-text search (W2.2)
-- Generated column keeps fts in sync with content automatically.
ALTER TABLE document_chunks
    ADD COLUMN IF NOT EXISTS fts tsvector
    GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;

CREATE INDEX IF NOT EXISTS idx_chunks_fts
    ON document_chunks USING gin(fts);

# Vector DB Analysis — pgvector vs Pinecone vs Chroma vs Weaviate

Decision: **pgvector on RDS PostgreSQL 15**.

---

## The Core Requirement

The EIA audit trail is non-negotiable: every dispatch recommendation must be traced to the exact document chunks that supported it. That means the retrieval event and the audit log entry must commit together or not at all. A vector store that is a separate service cannot provide this guarantee.

This constraint eliminates every managed vector database from the shortlist.

---

## Options Evaluated

### pgvector (chosen)

PostgreSQL 15 with the `pgvector` extension. The embedding column (`vector(1024)`) lives in the same `document_chunks` table as the content, source, and metadata. The HNSW index (`vector_cosine_ops`) provides sub-millisecond ANN recall.

**Pros**
- Atomic transactions: the retrieval result and audit log entry are in the same database — they commit together. No distributed consistency problem.
- Operational simplicity: one RDS instance, one connection pool, one backup schedule. No additional infrastructure to operate.
- PostgreSQL's tsvector/GIN index provides BM25-style keyword search in the same query path — hybrid retrieval without a second service.
- Exact match queries (SQL `WHERE` clauses) work natively — useful for filtering by source, category, or date.
- pgvector HNSW is production-grade: used at scale by Supabase, Neon, and others.

**Cons**
- Not purpose-built for vector search: filtering before ANN (pre-filtering) requires care — over-filtering can degrade HNSW recall.
- Scaling beyond ~10M vectors requires PostgreSQL tuning and potentially partitioning, whereas managed vector DBs scale with less effort.
- HNSW index rebuild on schema changes is slow for large corpora.

**Configuration used**
```sql
CREATE INDEX idx_chunks_embedding
    ON document_chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
```
`m=16` / `ef_construction=64` is the standard starting point for corpora < 1M vectors.
Titan V2 produces 1024-dimensional embeddings. The 1536-dim column from a prior iteration was migrated down to 1024 at first use.

---

### Pinecone

Fully managed vector DB. Sub-10ms query latency at any scale, automatic replication, namespaces for multi-tenancy.

**Why not**
- Pinecone is a separate service: retrieval events and audit log entries cannot be in the same transaction. The audit trail would be eventually consistent at best.
- Under NIS2, external API calls create a supply chain dependency. Pinecone falls outside the client VPC.
- Adds ~$70–$200/month in cost for the scale relevant to this system (< 500k vectors). Not justified when pgvector on RDS covers the use case.
- No native BM25 — hybrid search requires a separate keyword index.

---

### Chroma

Embedded Python vector database. Zero infrastructure, runs in-process.

**Why not**
- Embedded store cannot be accessed by multiple processes or services. The EIA RAG pipeline and the anomaly pipeline would each need their own copy of the index — writes are not shared.
- No transactional guarantees with the PostgreSQL audit log.
- Not designed for production deployments with durability requirements (NIS2 requires 6-year record retention).
- No native BM25. Hybrid search requires additional tooling.
- Persistent Chroma with a client-server split re-introduces the separate-service problem.

---

### Weaviate

Open-source vector database with built-in BM25 (BM25F) and hybrid search.

**Why not**
- Separate Kubernetes deployment (or managed Weaviate Cloud). Increases operational footprint and adds an external dependency for NIS2-critical infrastructure.
- No shared transaction with PostgreSQL audit log — same consistency problem as Pinecone.
- Built-in hybrid search is appealing, but we achieve the same via pgvector + tsvector in one system.
- Additional Kubernetes resources and a separate backup/recovery strategy.

---

## Decision Matrix

| Criterion | pgvector | Pinecone | Chroma | Weaviate |
|---|---|---|---|---|
| Atomic transaction with audit log | ✅ Same DB | ❌ Separate service | ❌ No audit log | ❌ Separate service |
| NIS2 data residency | ✅ VPC-local | ❌ External API | ✅ In-process | ✅ Self-hosted |
| Native BM25 (hybrid) | ✅ tsvector | ❌ Needs second index | ❌ | ✅ BM25F |
| Production durability | ✅ RDS | ✅ Managed | ⚠️ Embedded only | ✅ With k8s |
| Operational complexity | ✅ One DB | ⚠️ API key + SDK | ✅ Minimal | ❌ Cluster required |
| Cost at < 500k vectors | ✅ Included | ❌ $70–200/month | ✅ Free | ⚠️ Infra cost |

---

## Measured Retrieval Performance

Evaluated on 15 labeled queries spanning regulations, tariffs, and decisions.

| Strategy | Precision@5 | Notes |
|---|---|---|
| BM25 only | 0.58 | Exact acronyms handled; paraphrase queries miss |
| Hybrid (BM25 + pgvector + RRF) | 0.78 | Both signals captured; gate passed |

BM25 alone handles exact acronyms (BSUoS, TNUoS) but misses paraphrase queries.
Semantic search alone handles intent but misses rare domain acronyms absent from Titan V2's training corpus.
RRF fusion with `k=60` and `fetch_k_multiplier=3` captures both signals.

Gate: **hybrid Precision@5 ≥ 0.70** — verified by `eval/retrieval_evals.py`.

---

## HNSW Tuning Notes

Default `ef_search=40` (PostgreSQL) is sufficient for the current ~250-chunk corpus.
For corpora > 100k vectors, tune `hnsw.ef_search` at query time:
```sql
SET hnsw.ef_search = 100;
SELECT ... ORDER BY embedding <=> $1 LIMIT 5;
```
Higher `ef_search` improves recall at the cost of latency. At < 1k chunks the difference is negligible.

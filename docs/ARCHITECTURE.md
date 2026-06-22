# Architecture — Energy Intelligence Agent

This document describes the design of the EIA system: why each component was chosen,
what the alternatives were, and where the trade-offs land.

---

## System Overview

The EIA is a production RAG system for energy dispatch. It answers natural-language
queries from energy operators against a live corpus of UK energy regulations,
BSC decisions, and grid operator instructions — and returns grounded, auditable
dispatch recommendations.

```
                         ┌──────────────────────────────────────┐
                         │       ENERGY INTELLIGENCE AGENT       │
                         │                                       │
  Operator query ──────► │  [Hybrid Retriever]  [LLM Backend]   │
                         │   BM25 + pgvector      Claude /      │
                         │   + RRF fusion         Mistral /     │
                         │                        Ollama        │
                         │         │                │           │
                         │    [Safety Layer]        │           │
                         │   term overlap +    confidence       │
                         │   semantic sim.     scoring +        │
                         │   constraint        Platt scale      │
                         │   validator                          │
                         │         │                            │
                         │    [Audit Log]  ◄────────────────────┘
                         │  immutable JSONL                      │
                         │  every decision traceable             │
                         └──────────────────────────────────────┘
                                   │
                              DispatchRecommendation
                         (action, confidence, cited docs,
                          hallucination_flagged, requires_human_review)
```

The design has three invariants that are never compromised:

1. **Every recommendation is traceable.** The audit log records every decision
   before it is returned to the operator, with the exact document IDs that
   supported it.

2. **The inference backend is swappable.** A single CLI flag changes the backend
   from AWS Bedrock to Ollama to vLLM on EKS. The application code, Pydantic
   schemas, and safety layer are identical across all three.

3. **The safety layer is in the critical path, not a bolt-on.** Hallucination
   detection, confidence scoring, and constraint checking run before the
   recommendation reaches an operator.

---

## Component Design

### 1. Hybrid Retriever

**File:** `src/retriever/hybrid_retriever.py`

The retriever runs two searches in parallel and fuses the result lists using
Reciprocal Rank Fusion (RRF).

```
Query ──► BM25 (PostgreSQL tsvector) ──► ranked list A
     └──► Semantic (pgvector cosine) ──► ranked list B
                                              │
                              RRF fusion (k=60)
                                              │
                              Top-k combined results
```

**Why BM25 + semantic, not one or the other:**

Energy regulatory text contains precise acronyms (BSUoS, TNUoS, BM, GCT,
NGESO) that embedding models trained on general text do not reliably vectorise.
A query for "BSUoS settlement period SP23" will miss BM25 hits if the embedding
model compresses the acronym incorrectly.

Semantic search captures paraphrase and synonymy — "balancing costs recovery"
retrieves documents about BSUoS even when the exact term is not in the query.

Measured on 15 labelled queries:

| Strategy | Precision@5 |
|---|---|
| BM25 only | 0.58 |
| Semantic only | 0.64 |
| Hybrid (BM25 + semantic + RRF) | **0.78** |

The hybrid retriever closes a 0.20 gap over BM25-only and 0.14 over semantic-only.

**Why RRF over a learned fusion weight:**

RRF ranks by position, not score, using:

```
RRF_score(d) = Σ  1 / (k + rank_i(d))
               i
```

A learned fusion weight (e.g. `0.4 × BM25_score + 0.6 × semantic_score`)
requires a labelled training set and periodic re-fitting as the corpus changes.
RRF has no parameters to tune and degrades gracefully when one retriever
produces an empty result set.

**Why pgvector over Pinecone/Chroma:**

The audit trail is the non-negotiable constraint. Every dispatch recommendation
must commit its retrieval event and audit log entry atomically — if the write
fails, neither record should exist. A separate vector store cannot participate
in a PostgreSQL transaction.

Full analysis: [VECTOR_DB_ANALYSIS.md](VECTOR_DB_ANALYSIS.md)

---

### 2. LLM Recommendation Engine

**Files:** `src/recommendation_engine/claude_client.py`, `ollama_client.py`

The recommender takes the query and retrieved document chunks, builds a prompt
from the versioned YAML config, and returns a structured JSON response.

```
context_docs + query
        │
  build_user_message()   (prompt_configs/system_v1.yaml)
        │
  LLM call (async)
        │
  parse_llm_output()   ──► JSON retry on parse failure (once)
        │
  DispatchRecommendation
  (before safety checks)
```

**Why structured JSON output, not free text:**

The safety layer needs machine-readable fields (`action`, `confidence`,
`reasoning`, `cited_doc_ids`) to run term-overlap and constraint checks.
Free text would require a second LLM pass to extract these — adding latency
and another failure point.

**Why `temperature=0.0`:**

Dispatch recommendations are safety-critical. Sampling variation at
`temperature > 0` would produce non-deterministic recommendations for
identical queries — unacceptable in a compliance-audited context.

**Why single-pass generation, not agentic:**

Agentic loops (plan → execute → reflect → retry) introduce unpredictable
latency, non-deterministic step counts, and audit complexity (which
intermediate step is the authoritative recommendation?). Single-pass
generation is sufficient for this use case: the retriever has already
assembled the evidence. The LLM synthesises it once.

**Why prompts are YAML configs in Git:**

A prompt change can degrade output quality as severely as a code change.
Version-controlling prompts alongside code means every quality regression
is attributable to a specific commit. The CI gate (`evaluation/ci_gate.py`)
fails the build if a prompt version drops below the faithfulness gate —
just as a failing unit test fails the build.

**Backend swapping:**

```
EnergyIntelligenceAgent(backend="bedrock")  →  ClaudeRecommender
EnergyIntelligenceAgent(backend="ollama")   →  OllamaRecommender
```

Both implement `async recommend(query, context_docs) -> DispatchRecommendation`.
The application code, safety layer, and audit trail are unchanged. See
[SOVEREIGN-INFERENCE.md](SOVEREIGN-INFERENCE.md) for the full sovereignty
decision framework.

---

### 3. Safety Layer

**Files:** `src/recommendation_engine/safety.py`, `src/safety/constraint_validator.py`

Three checks run in sequence before a `DispatchRecommendation` is returned.

#### 3.1 Term Overlap (always runs)

```python
overlap = |tokens(action) ∩ tokens(corpus)| / |tokens(action)|
```

Stopwords and tokens shorter than 3 characters are excluded. Gate: ≥ 0.30.

A recommendation that uses vocabulary not present in the retrieved documents is
flagging unknown information — a strong hallucination signal for energy domain
language, where precise regulatory terms are non-substitutable.

#### 3.2 Semantic Similarity (runs when embedder is available)

Cosine similarity between the embedded action+reasoning and the embedded corpus.
Gate: ≥ 0.50. Skipped in sovereign mode to avoid Bedrock calls.

This catches meaning-level drift that term overlap misses — paraphrased
hallucinations that use different words to make a false claim.

#### 3.3 Regulatory Constraint Checking (always runs)

Parses retrieved documents for explicit operational constraints (MW ceilings,
directional prohibitions, unit bans) using sentence-level regex. Checks the
recommended action against any extracted constraints.

This catches grounded-but-wrong recommendations: cases where the LLM correctly
references a retrieved document but still recommends an action that violates
an explicit rule in that same document.

**Why regex, not LLM-extracted constraints:**

LLM extraction of constraints introduces the same hallucination risk it is
trying to prevent. Regex on explicit regulatory language (`"must not exceed"`,
`"upward dispatch instructions are not permitted"`) is deterministic, auditable,
and fast. It misses ambiguous language — but those cases are correctly left to
human review.

Full safety design, calibration detail, and example traces: [SAFETY.md](SAFETY.md)

---

### 4. Confidence Scoring

**File:** `src/recommendation_engine/confidence.py`

The raw LLM confidence is not used directly. A composite signal is computed:

```
composite = 0.60 × llm_confidence
          + 0.25 × grounding_score
          + 0.15 × mean_retrieval_similarity
```

Then optionally calibrated via Platt scaling on a labelled validation set
(logistic transform fitted to historical outcomes). The Brier score gate is
< 0.20.

`requires_human_review = (confidence < 0.60) OR hallucination_flagged OR constraint_violated`

The 0.60 threshold was chosen as the operating point where the composite signal
has enough signal to differentiate confident from uncertain recommendations.
Below 0.60, operator review is cheaper than the risk of an incorrect automated
decision.

---

### 5. Audit Trail

**File:** `src/audit/audit_log.py`

Every recommendation is appended to `logs/audit.jsonl` before it is returned.
The path is controlled by the `AUDIT_LOG_PATH` environment variable.

Design invariants:
- `AuditLogEntry` is `frozen=True` (Pydantic v2) — immutable after creation
- File is always opened in `"a"` (append) mode — no truncation or seek
- `asyncio.Lock` serialises concurrent writes within a process
- File I/O runs in `asyncio.to_thread` — does not block the event loop

The audit trail is separate from the RAG metrics log (`results/rag_log.jsonl`).
The audit log is compliance-grade and never truncated. The metrics log is a
rolling feed safe to archive or delete.

---

### 6. Observability

**Files:** `src/instrumentation/langfuse_client.py`, `src/instrumentation/decorators.py`

Every pipeline stage is wrapped in a Langfuse trace span when credentials are
configured. When `LANGFUSE_PUBLIC_KEY` is absent, all context managers degrade
gracefully to no-ops — observability is never in the critical path.

```python
async with trace_dispatch(query) as dispatch_span:
    async with trace_retrieval(query) as ret_span:
        docs = await retriever.search(query)
        ret_span["n_docs"] = len(docs)
    async with trace_generation(query, backend=backend) as gen_span:
        rec = await recommender.recommend(query, docs)
        gen_span["confidence"] = rec.confidence
```

Four span types:
- `trace_dispatch` — full pipeline wall-clock time
- `trace_retrieval` — retriever call, n_docs, sources
- `trace_generation` — LLM call, backend, model, confidence, hallucination flag
- `trace_anomaly_detection` — Stream 1 scorer, window_id, anomaly_score

The `@traced_span` decorator is available for functions not using async context
managers. It wraps both sync and async functions with a named Langfuse span.

---

## Stream 1 — Real-Time Incident Detection

**Directory:** `stream1_log_anomaly/`

```
Log stream (file or Redis) ──► LogEvent (Pydantic)
                                     │
                          IsolationForestScorer
                                     │
                        AnomalyResult (score, is_anomaly)
                                     │
                     (if is_anomaly) BedrockExplainer
                                     │
                     Explanation (severity, summary, cause, action)
                                     │
                             Langfuse trace + anomaly log
```

**Why Isolation Forest, not LLM-based anomaly detection:**

Isolation Forest provides:
- Continuous anomaly scores (not binary), enabling severity gradation
- No cluster count parameter to tune (unlike k-means)
- Explicit `contamination` parameter (tunable against known anomaly rates)
- Sub-millisecond scoring per event after fitting
- Explainable behaviour — the score reflects how many random cuts are needed
  to isolate the point

LLM-based anomaly detection adds ~1–5 seconds per event, costs ~$0.003/call,
and produces probabilistic rather than continuous scores. Classical ML is
faster, cheaper, and more transparent for this structured detection task.

The architectural principle: **classical ML scores, LLM explains.**

**Why asyncio, not threading:**

The pipeline is I/O-bound: reading from a file tail or Redis stream, waiting
for Bedrock responses, writing to Langfuse. `asyncio.gather` overlaps these
waits without the overhead of a thread pool or the synchronisation complexity
of locks across threads.

**Feature engineering:**

Three features per log event: `latency_ms` (continuous), `is_error` (binary:
1 if level == ERROR), `is_500` (binary: 1 if status_code >= 500). These three
signals fire simultaneously on the injected anomaly clusters, giving the
Isolation Forest a clean separation boundary without needing the event count
or timestamp as features.

---

## Data Flow — Full Request Lifecycle

```
1. Operator sends query: "What BSUoS charge applies to SP23?"

2. EnergyIntelligenceAgent.dispatch(query)
      │
3.    ├── HybridRetriever.search(query)
      │      BM25 search → ranked list A
      │      pgvector search → ranked list B
      │      RRF fusion → top-5 RetrievedDocuments
      │
4.    ├── ClaudeRecommender.recommend(query, docs)
      │      build_user_message(query, docs)   → user prompt
      │      load_system_prompt("v1")          → system prompt
      │      boto3 invoke_model() in thread    → raw JSON text
      │      parse_llm_output(raw, docs)       → ParsedLLMOutput
      │      check_grounding(action, docs)     → SafetyResult
      │      ConfidenceScorer.score(signals)   → (confidence, requires_review)
      │      check_constraints(action, docs)   → ValidationResult
      │      → DispatchRecommendation
      │
5.    ├── append_dispatch(rec, ...)             → results/rag_log.jsonl
      │
6.    └── AuditLogger.record(entry)            → logs/audit.jsonl

7. Return DispatchRecommendation to operator
```

Each step has a corresponding Langfuse span. Steps 5 and 6 are fire-and-forget
with respect to the return value — failures are logged but do not raise.

---

## Trade-Off Summary

| Decision | Chosen | Rejected | Reason |
|---|---|---|---|
| Vector store | pgvector (PostgreSQL) | Pinecone, Chroma, Weaviate | Atomic transaction with audit log |
| Retrieval | Hybrid BM25 + semantic + RRF | BM25 only, semantic only | +0.20 Precision@5 gain |
| Chunking | SemanticChunker | FixedSize, SlidingWindow | Regulatory text is clause-structured |
| Anomaly detection | Isolation Forest | LLM scoring | 1000× faster, interpretable scores |
| Generation | Single-pass | Agentic loop | Sufficient; simpler audit trail |
| Inference backend | Swappable (Bedrock / Ollama / vLLM) | Hardcoded Bedrock | NIS2 sovereignty requirement |
| Prompt management | YAML configs in Git | Hardcoded strings | Versioned, CI-gated quality |
| Observability | Langfuse spans (graceful no-op) | None | Production visibility without blocking |
| Confidence | Composite + Platt calibration | Raw LLM output | Calibrated probabilities, Brier gate |
| Hallucination detection | Term overlap + semantic similarity | LLM self-check | Independent of the model being checked |
| Concurrency | asyncio throughout | threading | I/O-bound; lower overhead, no lock complexity |

---

## What This System Is Not

**Not multi-step agentic.** Single-pass generation is sufficient for this use case.
The retriever assembles evidence; the LLM synthesises it once. Agentic behaviour
(plan → execute → reflect) is deferred to where it earns its complexity: the DPO
fine-tuning pipeline in Stream 4.

**Not fine-tuned.** Energy regulations change monthly. A fine-tuned model would
require monthly retraining cycles. RAG keeps the model general and the knowledge
current — update the document corpus, not the weights.

**Not cloud-only.** Sovereign inference is a first-class architectural requirement.
Ollama on a developer machine and vLLM on an EKS cluster inside a client VPC
use the same application code as Bedrock. The infrastructure changes; nothing else does.

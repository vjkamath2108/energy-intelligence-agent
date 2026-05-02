# CLAUDE.md — Energy Intelligence Agent

A production RAG system for energy dispatch with swappable inference backends,
real-time anomaly detection, and full observability instrumentation.

---

## Repo Structure

```
energy-intelligence-agent/
├── src/                          # EIA core — RAG pipeline
│   ├── agent/                    # EnergyIntelligenceAgent orchestrator
│   ├── audit/                    # Immutable audit trail
│   ├── data_pipeline/            # Ingestion, chunking
│   ├── instrumentation/          # Langfuse client, trace decorators
│   ├── models/                   # Pydantic v2 schemas
│   ├── recommendation_engine/    # Claude + Ollama integration, safety layer
│   ├── retriever/                # BM25, semantic, hybrid (RRF)
│   └── db.py                     # RDS PostgreSQL connection module
├── stream1_log_anomaly/          # Real-time log anomaly detection pipeline
│   ├── src/
│   │   ├── streaming/            # log_generator, tail_consumer, redis_consumer
│   │   └── detection/            # Isolation Forest anomaly scorer
│   ├── logs/                     # sample.jsonl (synthetic EKS logs)
│   └── tests/
├── stream2_localslm/             # Edge inference + sovereign architecture
│   ├── src/inference/            # ollama_client, vllm_client, benchmarks
│   └── results/                  # bedrock_vs_ollama.md
├── stream3_observability/        # AI observability layer
│   └── src/instrumentation/      # Langfuse, dashboards, CI eval gate
├── stream4_finetuning/           # SFT + DPO fine-tuning
│   ├── data/                     # DPO pairs, SFT dataset
│   ├── training/                 # LoRA/QLoRA scripts
│   └── results/
├── shared/                       # Cross-stream shared code
├── results/                      # Benchmark outputs
├── eval/                         # Retrieval eval framework, CI gate
└── docs/                         # Architecture decision records
```

---

## Active Build State

**Current sprint: Stream 1, Sprint 0 — gate Wednesday 6 May 2026**

- [x] S1.0.1 — Synthetic log generator (`stream1_log_anomaly/src/streaming/generate_logs.py`)
- [x] S1.0.2 — asyncio file tail consumer (`stream1_log_anomaly/src/streaming/tail_consumer.py`)
- [x] S1.0.3 — Isolation Forest anomaly scorer (`stream1_log_anomaly/src/detection/anomaly_scorer.py`)
- [x] S1.0.4 — Bedrock explainer (`stream1_log_anomaly/src/explanation/bedrock_explainer.py`)
- [x] S1.0.5 — Wire together in `stream1_log_anomaly/main.py`

Update checkboxes as tasks complete.

---

## Architecture Decisions — Do Not Revisit Without Strong Evidence

### Inference Backends
- **Default:** AWS Bedrock (`anthropic.claude-sonnet-4-5`)
- **Dev sovereign:** Ollama on localhost (`http://localhost:11434`)
- **Production sovereign:** vLLM on EKS inside client VPC
- Swappable by single CLI flag: `--explainer bedrock|ollama`
- **Same Pydantic output schema regardless of backend. This is non-negotiable.**
- Cost crossover: ~50M tokens/day where EKS self-hosting beats Bedrock on TCO
- Primary regulation driver: NIS2 (not just GDPR) for UK energy operators

### Retrieval
- Hybrid: BM25 (PostgreSQL tsvector) + semantic (pgvector) + Reciprocal Rank Fusion
- pgvector over Pinecone/Chroma: atomic transactions with audit log in same DB
- Evidence for hybrid: BM25-only Precision@5 = 0.58 → hybrid = 0.78

### ML vs LLM Division of Labour
- Classical ML does structured work: Prophet (forecasting), Isolation Forest (anomaly detection)
- LLM does explanation and natural language query only
- LLM never does anomaly scoring. Isolation Forest scores. LLM explains.

### Safety
- Hallucination detection: term overlap (≥30%) + semantic similarity check
- Confidence scoring calibrated on validation set, Brier score target <0.2
- Audit trail: immutable, append-only, Pydantic `frozen=True`
- CI gate: GitHub Actions fails if RAGAS faithfulness < 0.80

---

## Coding Standards

```
Language:       Python 3.11+
Type checking:  mypy --strict (must pass, no exceptions)
Linting:        ruff check src/ && ruff format src/
Testing:        pytest with ≥80% coverage
Docstrings:     Every public function — type hints + one-line purpose + args
Async:          asyncio throughout — no threading
```

**Never:**
- Hardcode credentials — use `.env` via `python-dotenv`
- Use `Optional[X]` — use `X | None`
- Skip type hints on any function
- Use `print()` in pipeline code — use `logging`
- Write a function longer than ~40 lines without splitting

**Always:**
- Handle malformed input without crashing — log and skip
- Use `asyncio.Queue` for inter-component communication
- Wrap external API calls (Bedrock, Ollama) in retry with exponential backoff
- Write the test alongside the implementation

---

## Pydantic v2 Conventions

- `model | None` not `Optional[model]`
- `frozen=True` on all audit and decision models
- `ConfigDict` not class-level `Config`
- No `Any` — all models fully typed

### Canonical Model Names

Do not rename these fields without updating all consumers:

```python
# Stream 1
LogEvent        # timestamp, level, message, latency_ms, status_code, service
AnomalyResult   # event, anomaly_score, is_anomaly, window_id
Explanation     # severity, summary, likely_cause, recommended_action

# EIA core
RetrievedDocument       # chunk, metadata, similarity_score
DispatchRecommendation  # action, confidence, citations, requires_human_review
AuditLogEntry           # full decision trace — frozen=True
```

---

## Environment Variables

```bash
# AWS
AWS_REGION=eu-west-2
AWS_PROFILE=vijay-dev

# Database (EIA)
DB_HOST=
DB_PORT=5432
DB_NAME=energy_db
DB_USER=energy_user
DB_PASSWORD=

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b

# Langfuse
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=https://cloud.langfuse.com
```

`.env` is in `.gitignore`. Never commit credentials.

---

## Quality Gates

| Metric | Target |
|---|---|
| Retrieval Precision@5 | ≥ 0.70 |
| Retrieval latency p95 | < 500ms |
| Full pipeline latency p95 | < 2s |
| RAGAS faithfulness (CI gate) | ≥ 0.80 |
| Hallucination detection rate | ≥ 80% of contradictions caught |
| Confidence calibration Brier score | < 0.2 |
| Test coverage | ≥ 80% |

---

## What This System Is NOT

- Not multi-step agentic (single-pass generation by design — sufficient for this use case)
- Not fine-tuned (RAG by design — energy regulations change monthly)
- Not cloud-only (sovereign inference is a first-class architectural requirement)

When considering adding complexity: does it hit a sprint gate or improve
a measurable quality metric? If neither, defer it.
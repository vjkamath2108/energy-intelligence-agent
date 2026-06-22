# Energy Intelligence Agent

A RAG system for energy dispatch optimization, built with production
engineering practices — auditable decisions, grounded recommendations,
and inference that can run entirely within a client's own network.

---

## The Problem

Energy dispatch carries three constraints that most AI systems ignore:

**Regulations change monthly.** UK energy market rules are updated
continuously. Fine-tuning an LLM on a monthly cycle is impractical and
expensive. The system needs to stay current without retraining.

**Wrong recommendations have regulatory and financial consequences.**
A dispatch decision that contradicts current regulation exposes the operator
to fines. Every recommendation must be traceable to the specific document
that supports it — not just plausible-sounding.

**Client data often cannot leave the network.** Energy operators under
NIS2 (Network and Information Security Directive) are classified as
critical infrastructure. An external API endpoint is a supply chain
dependency. Some clients need inference to run entirely within their
own VPC with no external calls.

RAG addresses the first constraint. The safety layer addresses the second.
The swappable inference backend addresses the third.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ENERGY INTELLIGENCE AGENT                        │
│                                                                     │
│  Query → [Hybrid Retriever] → [Claude Sonnet] → [Safety Layer]     │
│              BM25 + pgvector      structured        hallucination   │
│              + RRF fusion         JSON output       detection +     │
│                                                     confidence      │
│                    │                  │             scoring         │
│              [Audit Trail] ←──────────┴─────────────────┘          │
│         immutable append-only log — every decision traceable        │
└─────────────────────────────────────────────────────────────────────┘
```

### Four Streams — Each Closes a Specific Gap

The centrepiece alone leaves four questions unanswered. The streams answer them.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  QUESTION 1: How do you detect and explain operational incidents          │
│  before an engineer reads the logs?                                      │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  STREAM 1 — Real-Time Incident Detection                        │    │
│  │                                                                 │    │
│  │  Energy operators are fined for unplanned outages. A 15-minute │    │
│  │  detection-to-explanation gap costs more than the system.       │    │
│  │                                                                 │    │
│  │  Live telemetry → Isolation Forest (scores anomalies) →         │    │
│  │  Claude (generates human-readable incident summary)             │    │
│  │                                                                 │    │
│  │  Classical ML scores. LLM explains. Neither does the other's    │    │
│  │  job. Result: incident summaries in seconds, not minutes.       │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  QUESTION 2: What happens when the client's data cannot leave           │
│  their network — not as a preference, but as a legal requirement?        │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  STREAM 2 — Sovereign Inference                                 │    │
│  │                                                                 │    │
│  │  Calling Bedrock makes AWS a GDPR data processor. Under NIS2,    │    │
│  │  Under NIS2, an external inference endpoint is a supply chain   │    │
│  │  dependency. Some clients cannot accept either.                 │    │
│  │                                                                 │    │
│  │  The inference backend is swappable by a single flag:           │    │
│  │  --explainer bedrock  (default, Anthropic API)                  │    │
│  │  --explainer ollama   (self-hosted, zero external calls)        │    │
│  │                       vLLM on EKS in production sovereign mode  │    │
│  │                                                                 │    │
│  │  Same Pydantic output schema. Same application code.            │    │
│  │  Published benchmark: latency, cost, and quality trade-offs.    │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  QUESTION 3: How do you know the system is degrading before              │
│  a user complains?                                                       │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  STREAM 3 — AI Observability                                    │    │
│  │                                                                 │    │
│  │  A prompt change can degrade output quality as badly as a code  │    │
│  │  change — but it won't show up in unit tests.                   │    │
│  │                                                                 │    │
│  │  Langfuse distributed tracing across every LLM call.            │    │
│  │  RAGAS faithfulness score on every PR — build fails below 0.80. │    │
│  │  Prompts versioned in Git alongside code.                       │    │
│  │  Streamlit dashboard: latency, cost, citation coverage over time.│    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  QUESTION 4: How do you improve the system without starting over?        │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  STREAM 4 — Fine-Tuning from Pipeline Outputs                  │    │
│  │                                                                 │    │
│  │  The pipeline generates its own training signal. Bedrock and    │    │
│  │  Ollama produce different explanations for the same anomaly.    │    │
│  │  Those paired outputs become DPO preference pairs.              │    │
│  │                                                                 │    │
│  │  SFT + DPO on Qwen 3 8B using the pipeline's own outputs.       │    │
│  │  The system improves from its own production data — not a       │    │
│  │  separate dataset collected for the purpose.                    │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### Inference Backend

```
                         ┌─────────────────────┐
                         │   Application code  │
                         │   (unchanged)       │
                         └──────────┬──────────┘
                                    │
          ┌─────────────────────────┼──────────────────────────┐
          ▼                         ▼                          ▼
  FRONTIER MANAGED          OPEN-WEIGHT MANAGED         SOVEREIGN
  --explainer bedrock        --explainer                --explainer
                              mistral-bedrock            mistral-local
          │                         │                          │
  AWS Bedrock                AWS Bedrock                  Ollama /
  Claude Sonnet              Mistral 7B                   vLLM on EKS
  ~$0.003/call               ~$0.0001/call                ~$0/call
  Data → AWS                 Data → AWS                   Zero egress
          │                         │                          │
          └─────────────────────────┴──────────────────────────┘
                                    │
                         ┌──────────▼──────────┐
                         │  Pydantic schema    │
                         │  (identical output) │
                         └─────────────────────┘
```

Swappable by a single CLI flag. Same Pydantic output schema regardless of
backend — application code is unchanged. See
[results/bedrock_vs_ollama_vs_mistral.md](results/bedrock_vs_ollama_vs_mistral.md)
for the full benchmark and deployment decision framework.

---

## Repo Structure

```
energy-intelligence-agent/
├── src/                          # EIA core — RAG pipeline
│   ├── agent/                    # EnergyIntelligenceAgent orchestrator
│   ├── audit/                    # Immutable audit trail
│   ├── data_pipeline/            # Ingestion + semantic chunking
│   ├── instrumentation/          # Langfuse client + trace decorators
│   ├── models/                   # Pydantic v2 schemas
│   ├── recommendation_engine/    # Claude integration + safety layer
│   ├── retriever/                # BM25 + semantic + hybrid (RRF)
│   └── db.py                     # RDS PostgreSQL + pgvector
│
├── stream1_log_anomaly/          # Real-time incident detection pipeline
│   ├── src/streaming/            # Log generator + asyncio tail consumer
│   ├── src/detection/            # Isolation Forest anomaly scorer
│   ├── src/explanation/          # Bedrock + Ollama explainers
│   └── main.py                   # Pipeline entry point
│
├── stream2_localslm/             # Sovereign inference + benchmarks
│   ├── src/inference/            # Ollama + vLLM clients
│   └── results/                  # bedrock_vs_ollama.md
│
├── stream3_observability/        # AI observability layer
│   └── src/instrumentation/      # Langfuse + Streamlit dashboard
│
├── stream4_finetuning/           # SFT + DPO from pipeline outputs
│   ├── data/                     # DPO preference pairs
│   └── training/                 # LoRA/QLoRA scripts
│
├── shared/                       # Cross-stream shared code
├── eval/                         # Retrieval eval + CI gate
├── results/                      # Published benchmark studies
└── docs/                         # Architecture decision records
```

---

## Quick Start

### Stream 1 — Real-Time Incident Detection

```bash
git clone https://github.com/vijaykamath/energy-intelligence-agent
cd energy-intelligence-agent

python -m venv .venv && source .venv/bin/activate
pip install -r stream1_log_anomaly/requirements.txt

cp .env.example .env   # add AWS credentials

python -m stream1_log_anomaly.main                             # Claude Sonnet (Bedrock)
python -m stream1_log_anomaly.main --explainer mistral-bedrock # Mistral 7B (Bedrock, 30x cheaper)
python -m stream1_log_anomaly.main --explainer mistral-local   # Mistral 7B (sovereign, zero egress)
```

Example output:

```
INFO  — Isolation Forest fitted on 20 events (window 0)
INFO  — Anomaly detected (event 033, score=-0.7374)

🚨 ANOMALY [033] — auth-service | ERROR | 1624ms
============================================================
Severity:  HIGH
Summary:   Auth-service cascade failure — 4 services degraded within 5 seconds
Cause:     Database connection pool exhaustion triggering downstream timeouts
Action:    Check auth-service health endpoints and connection pool metrics.
           Review recent deployments. Consider circuit breaker activation.
```

### EIA Core

```bash
# Prerequisites: RDS PostgreSQL 15 + pgvector, .env configured
python -m src.agent.energy_agent --query "Optimal dispatch for peak demand"
```

### Demo Scripts — No AWS or DB Required

```bash
pip install -r requirements.txt

# RAG pipeline demo: 5 dispatch scenarios including hallucination detection
PYTHONPATH=. python examples/demo_queries.py

# Stream 1 anomaly detection demo: 300 synthetic EKS logs + Isolation Forest
PYTHONPATH=. python examples/demo_streaming.py
PYTHONPATH=. python examples/demo_streaming.py --max-anomalies 5

# CI eval gate: faithfulness + citation coverage on 5 golden eval cases
PYTHONPATH=. python evaluation/ci_gate.py --mock --prompt-version v1
```

Expected output from `demo_queries.py`:
```
SCENARIO 1: BSUoS Settlement Charge
  Confidence  : 0.867  |  Grounded: YES  |  Human review: NO

SCENARIO 5: HALLUCINATION DETECTION
  Confidence  : 0.262  |  Grounded: NO   |  Human review: YES ⚠

Result: ALL SCENARIOS PASSED — safety layer behaved as expected.
```

---

## Benchmarking Strategy

The system is designed to run across three deployment tiers. Every model runs
against the same anomaly events, same prompt template, and same ±5 context
window — making the comparison apples-to-apples.

### Three Tiers

**Tier 1 — Frontier managed (Claude Sonnet via Bedrock)**
Quality and reliability baseline. Best choice when output quality is
non-negotiable and the client has signed a DPA with Anthropic.

**Tier 2 — Open-weight managed (Mistral 7B via Bedrock)**
Same managed infrastructure as Tier 1 but ~30× cheaper per call. Accuracy
matched the frontier model on every injected anomaly pattern in benchmarking.
Best choice when volume makes Claude cost-prohibitive and the client is
comfortable with open-weight quality trade-offs.

**Tier 3 — Open-weight sovereign (Mistral 7B or phi4-mini via Ollama / vLLM)**
Zero data egress — satisfies NIS2 data residency requirements for UK energy
operators. No external API dependency. The latency trade-off is acceptable for
async anomaly pipelines; GPU hardware (g5.xlarge on EKS/vLLM) closes the gap
for real-time dashboards.

### Benchmark Results (Stream 1 — Log Anomaly Explanation)

Accuracy is measured only on injected pattern events with deterministic ground
truth. Isolated events are included in latency and JSON validity only.

| Model | Deployment | Severity Accuracy | Latency P50 | Cost/call | Data Egress |
|---|---|---|---|---|---|
| Claude Sonnet 4.5 | Bedrock (managed) | 82% | 4673ms | ~$0.003 | Yes (AWS) |
| Mistral 7B | Bedrock (managed) | **100%** | 1316ms | ~$0.0001 | Yes (AWS) |
| Mistral 7B | Ollama (sovereign) | **100%** | 3167ms | ~$0 | **None** |
| phi4-mini | Ollama (sovereign) | 96% | 1992ms | ~$0 | **None** |
| llama3.2:3b | Ollama (sovereign) | 6% | 1297ms | ~$0 | **None** |

Key findings:
- Mistral 7B (Bedrock) matches or exceeds frontier accuracy at 30× lower cost.
- Mistral 7B (Sovereign) delivers identical accuracy with zero data egress —
  the only production-ready NIS2-compliant option on current hardware.
- phi4-mini is the best-value sovereign fallback if Mistral is unavailable.
- llama3.2:3b is unsuitable for structured severity classification at 3B scale.

Full methodology, pattern-level breakdown, and deployment decision framework:
[results/bedrock_vs_ollama_vs_mistral.md](results/bedrock_vs_ollama_vs_mistral.md)

---

## Key Design Decisions

**Hybrid retrieval over pure semantic search**
Energy domain contains regulatory terminology that embedding models
don't reliably capture. Measured: BM25-only Precision@5 = 0.58,
hybrid (BM25 + pgvector + RRF) = 0.78.

**pgvector over Pinecone**
Atomic transactions between vector store and audit log in the same
database. When a decision is logged, retrieval event and recommendation
commit together or not at all. A separate vector store cannot guarantee this.

**Isolation Forest over LLM-based anomaly detection**
Continuous anomaly scores, no cluster count required, measurable
contamination parameter. LLM-based scoring adds latency and cost where
a classical scorer is faster, cheaper, and its behaviour is explainable.

**Single-pass generation over agentic loops**
Sufficient for this use case. Simpler failure modes, predictable latency,
easier to audit. Multi-step reasoning deferred to where it earns its
complexity: generating DPO preference pairs in Stream 4.

---

## Quality Gates

| Metric | Target |
|---|---|
| Retrieval Precision@5 | ≥ 0.70 |
| Retrieval latency p95 | < 500ms |
| Full pipeline latency p95 | < 2s |
| RAGAS faithfulness (CI gate) | ≥ 0.80 |
| Hallucination detection rate | ≥ 80% of contradictions caught |
| Anomaly detection | 5/5 injected anomalies detected |
| Test coverage | ≥ 80% |

---

## Documentation

| Document | Description |
|---|---|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Design decisions and rationale |
| [SAFETY.md](docs/SAFETY.md) | Responsible AI mechanisms |
| [VECTOR_DB_ANALYSIS.md](docs/VECTOR_DB_ANALYSIS.md) | pgvector vs alternatives |
| [CHUNKING_STRATEGY.md](docs/CHUNKING_STRATEGY.md) | Chunking approach |
| [SOVEREIGN-INFERENCE.md](docs/SOVEREIGN-INFERENCE.md) | Data sovereignty decision framework |
| [results/bedrock_vs_ollama_vs_mistral.md](results/bedrock_vs_ollama_vs_mistral.md) | Five-model inference benchmark across frontier, open-weight managed, and sovereign deployment tiers |
| [results/model_comparison_report.md](results/model_comparison_report.md) | **Model comparison report** — six models, two pipelines, quantisation trade-offs, deployment recommendations, honest write-up of what went wrong |

---

## Technology

Python 3.11+ · PostgreSQL 15 + pgvector · Pydantic v2 · AWS Bedrock ·
Ollama · Redis Streams · scikit-learn · Prophet · Langfuse · Streamlit
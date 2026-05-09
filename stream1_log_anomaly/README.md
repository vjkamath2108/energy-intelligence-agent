# Stream 1 — Real-Time Log Anomaly Pipeline

Streaming EKS log ingestion → statistical anomaly detection → LLM-generated
human-readable incident explanation.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│  Log Generator  │────▶│  Redis Streams   │────▶│  asyncio Consumer   │
│ generate_logs.py│XADD │  (log_events)    │XREAD │  redis_consumer.py  │
└─────────────────┘     └──────────────────┘     └──────────┬──────────┘
                                                             │
                                                             ▼
                                                  ┌─────────────────────┐
                                                  │  Isolation Forest   │
                                                  │  anomaly_scorer.py  │
                                                  └──────────┬──────────┘
                                                             │
                                              anomaly?       │
                                         ┌─────────Yes───────┘
                                         ▼
                              ┌─────────────────────┐
                              │   LLM Explainer     │
                              │  Bedrock (default)  │
                              │  Ollama  (--local)  │
                              └─────────────────────┘
                                         │
                                         ▼
                              ┌─────────────────────┐
                              │  Explanation output  │
                              │  severity / summary  │
                              │  cause / action      │
                              └─────────────────────┘
```

## Design Decisions

- **Classical ML scores, LLM explains.** Isolation Forest owns the anomaly
  decision. The LLM never scores — it only generates human-readable
  explanations when IF flags an event. This keeps latency low and the
  decision boundary auditable.

- **Redis Streams over file tail.** XREAD with blocking mode gives efficient
  event-driven ingestion with no polling overhead. File tail remains available
  via `--mode file` for environments without Redis.

- **Swappable inference backends.** Bedrock (default) and Ollama (local) share
  an identical interface and output schema. Switch by flag — no code changes.
  Ollama mode makes zero cloud calls — designed for data sovereignty constraints.

- **Pydantic v2 throughout.** All data models are strongly typed and validated
  at ingestion. `Explanation` is frozen — once generated, it cannot be mutated.

- **Buffer 20 events before fitting.** Isolation Forest needs enough events to
  characterise normal behaviour before scoring. 20 events gives a stable
  baseline while ensuring the anomaly cluster falls in the scoring window.

## Quick Start

### Prerequisites

```bash
# Redis
brew install redis && brew services start redis

# Python dependencies
pip install -r requirements.txt

# AWS credentials (for Bedrock mode)
aws configure
```

### Run the pipeline

```bash
# Bedrock mode (default) — calls AWS Bedrock for explanations
python main.py --mode redis --explainer bedrock

# Ollama mode — fully local, zero cloud calls
ollama pull llama3.2:3b
python main.py --mode redis --explainer ollama

# File tail mode (no Redis required)
python main.py --mode file --explainer bedrock
```

### Run tests

```bash
pytest stream1_log_anomaly/tests/ -v
```

### Run individual components

```bash
# Smoke test: log generator
python -m stream1_log_anomaly.src.streaming.generate_logs

# Smoke test: Redis consumer
python -m stream1_log_anomaly.src.streaming.redis_consumer

# Smoke test: anomaly scorer
python -m stream1_log_anomaly.src.detection.anomaly_scorer

# Smoke test: Bedrock explainer (hits real Bedrock)
python -m stream1_log_anomaly.src.explanation.bedrock_explainer

# Smoke test: Ollama explainer (local only)
python -m stream1_log_anomaly.src.explanation.ollama_explainer
```

## Stack

| Component | Technology | Rationale |
|---|---|---|
| Event ingestion | Redis Streams | Efficient blocking reads, no polling |
| Anomaly detection | Isolation Forest (scikit-learn) | Handles high-dimensional features, continuous scoring |
| LLM explanation | Claude Sonnet via Bedrock | Structured JSON output, grounded explanations |
| Local inference | Ollama llama3.2:3b | Zero cloud calls, sovereign data pattern |
| Data validation | Pydantic v2 | Strong typing, immutable audit models |
| Async runtime | asyncio | Non-blocking pipeline, event-driven |

## Bedrock vs Ollama — Benchmark Observations

| | Bedrock Sonnet | Ollama llama3.2:3b |
|---|---|---|
| JSON validity (first pass) | 5/5 | 2/5 |
| Needed retry | 0 | 3 |
| Cascade reasoning | ✅ Traced root cause across services | ❌ Treated each event independently |
| Severity accuracy | All HIGH (correct) | Mix of HIGH/MEDIUM |
| Data leaves network | Yes (AWS) | No |
| Cost per call | ~$0.003 | ~$0 |

Full benchmark report: `results/bedrock_vs_ollama.md` (Sprint 2)

## Project Structure

```
stream1_log_anomaly/
├── src/
│   ├── models.py                    # Pydantic v2 schemas (LogEvent, AnomalyResult)
│   ├── streaming/
│   │   ├── generate_logs.py         # Synthetic log generator
│   │   ├── tail_consumer.py         # asyncio file tail (--mode file)
│   │   └── redis_consumer.py        # Redis Streams consumer (--mode redis)
│   ├── detection/
│   │   └── anomaly_scorer.py        # Isolation Forest scorer
│   └── explanation/
│       ├── prompts.py               # Shared prompt templates
│       ├── bedrock_explainer.py     # AWS Bedrock Claude explainer
│       └── ollama_explainer.py      # Local Ollama explainer
├── tests/
│   ├── test_models.py
│   ├── test_anomaly_scorer.py
│   └── test_explainer.py
├── logs/
│   └── sample.jsonl                 # 50 synthetic log events (5 injected anomalies)
├── main.py                          # Pipeline entry point
└── requirements.txt
```
# Benchmark Report — Frontier vs Open-Weight Inference
## Stream 1: Log Anomaly Explanation Pipeline

**Date:** 2026-06-05  
**Dataset:** stream1_log_anomaly/logs/sample.jsonl (98 anomaly events from 300 synthetic logs)  
**Frontier model:** Claude Sonnet 4.5 (AWS Bedrock, us.* cross-region)  
**Open-weight models:** Mistral 7B (Bedrock), Mistral 7B (Sovereign/Ollama), llama3.2:3b (Ollama), phi4-mini (Ollama), qwen3:14b (Ollama)  
**Anomaly detector:** Isolation Forest (contamination=auto)  

---

## Test Methodology

All models ran against identical anomaly events detected by Isolation Forest
on the same 300-event synthetic log dataset (seed=42, reproducible).
Each model received the same prompt template and the same ±5 context window.

**Open-weight deployment modes benchmarked:**
- **Mistral 7B (Bedrock)** — managed API, data leaves machine to AWS
- **Mistral 7B (Sovereign/Ollama)** — fully local, zero external calls,
  addresses NIS2 data residency requirements for UK energy operators

**Option B ground truth:** Severity accuracy measured only on injected
pattern events. Isolated events included in latency/validity only.

Three injected anomaly patterns:
- **Pattern A** (events 60–69): Cascading failure — auth → inventory → payment-svc → api-gateway.
  All ERROR, 500 status, latency 1500–2500ms. Ground truth: HIGH.
- **Pattern B** (events 130–139): Single-service latency spike — payment-svc.
  Latency 2000–4000ms, mixed 200/500. Ground truth: HIGH.
- **Pattern C** (events 200–212): Error rate surge — api-gateway.
  Normal latency (100–400ms), elevated 500s. Tests error-only signal. Ground truth: HIGH.
- **Isolated events**: Outside injected windows. Excluded from accuracy.

---

## Results

| Metric | Claude Sonnet (Bedrock) | Mistral 7B (Bedrock) | Mistral 7B (Sovereign/Ollama) | llama3.2:3b (Ollama) | phi4-mini (Ollama) | qwen3:14b (Ollama) |
|---|---|---|---|---|---|---|
| Total events | 98 | 49 | 49 | 147 | 98 | 98 |
| Scoreable events (injected patterns) | 58 | 29 | 29 | 87 | 58 | 58 |
| Isolated events (excl. from accuracy) | 40 | 20 | 20 | 60 | 40 | 40 |
| JSON validity | 98/98 (100%) | 49/49 (100%) | 49/49 (100%) | 147/147 (100%) | 98/98 (100%) | 74/98 (75%) |
| Severity accuracy (injected only) | 48/58 (82%) | 29/29 (100%) | 29/29 (100%) | 14/87 (16%) | 57/58 (98%) | 39/58 (67%) |
| Pattern A accuracy (cascade) | 20/20 | 10/10 | 10/10 | 5/30 | 19/20 | 14/20 |
| Pattern B accuracy (latency spike) | 12/14 | 7/7 | 7/7 | 5/21 | 14/14 | 12/14 |
| Pattern C accuracy (error surge) | 16/24 | 12/12 | 12/12 | 4/36 | 24/24 | 13/24 |
| Latency P50 (ms) | 5194 | 960 | 4198 | 1305 | 1560 | 31812 |
| Latency P95 (ms) | 6482 | 1214 | 5054 | 4486 | 1861 | 87430 |
| Data leaves network | Yes (AWS us.* cross-region) | Yes (AWS Bedrock) | No | No | No | No |
| Cost per call | ~$0.003 | ~$0.0001 | ~$0 | ~$0 | ~$0 | ~$0 |
| Production-ready | Yes | Yes | No | No | No | No |

---

## Key Findings

### 1. Frontier vs Open-Weight Quality Gap

Accuracy measured on injected pattern events only (Option B).

- **Claude Sonnet (Bedrock)**: 48/58 (82%) overall.
- **Mistral 7B (Bedrock)**: 29/29 (100%) overall.
- **Mistral 7B (Sovereign/Ollama)**: 29/29 (100%) overall.
- **llama3.2:3b (Ollama)**: 14/87 (16%) overall.
- **phi4-mini (Ollama)**: 57/58 (98%) overall.
- **qwen3:14b (Ollama)**: 39/58 (67%) overall.

### 2. Mistral 7B — Managed vs Sovereign Deployment

Same model, two deployment modes. Key trade-off:
- **Bedrock**: lower latency, managed infrastructure, data leaves machine
- **Sovereign/Ollama**: ~4–5x higher latency, zero external dependency,
  satisfies NIS2 data residency for UK energy operators

For async anomaly pipelines the latency trade-off is acceptable.
For real-time triage dashboards it requires GPU hardware to close the gap.

### 3. Pattern C — Error Surge Without Latency

Pattern C tests severity identification from error signal alone,
without a latency spike. Only elevated 500 status codes as signal.

- **Claude Sonnet (Bedrock)**: Pattern C accuracy 16/24.
- **Mistral 7B (Bedrock)**: Pattern C accuracy 12/12.
- **Mistral 7B (Sovereign/Ollama)**: Pattern C accuracy 12/12.
- **llama3.2:3b (Ollama)**: Pattern C accuracy 4/36.
- **phi4-mini (Ollama)**: Pattern C accuracy 24/24.
- **qwen3:14b (Ollama)**: Pattern C accuracy 13/24.

### 4. Latency vs Quality Trade-off

- **Claude Sonnet (Bedrock)**: P50 5194ms, P95 6482ms, accuracy 48/58 (82%).
- **Mistral 7B (Bedrock)**: P50 960ms, P95 1214ms, accuracy 29/29 (100%).
- **Mistral 7B (Sovereign/Ollama)**: P50 4198ms, P95 5054ms, accuracy 29/29 (100%).
- **llama3.2:3b (Ollama)**: P50 1305ms, P95 4486ms, accuracy 14/87 (16%).
- **phi4-mini (Ollama)**: P50 1560ms, P95 1861ms, accuracy 57/58 (98%).
- **qwen3:14b (Ollama)**: P50 31812ms, P95 87430ms, accuracy 39/58 (67%).

### 5. JSON Reliability

- **Claude Sonnet (Bedrock)**: 98/98 (100%).
- **Mistral 7B (Bedrock)**: 49/49 (100%).
- **Mistral 7B (Sovereign/Ollama)**: 49/49 (100%).
- **llama3.2:3b (Ollama)**: 147/147 (100%).
- **phi4-mini (Ollama)**: 98/98 (100%).
- **qwen3:14b (Ollama)**: 74/98 (75%).

---

## Three-Way Deployment Decision Framework

| | Claude Sonnet (Bedrock) | Mistral 7B (Bedrock) | Mistral 7B (Sovereign) |
|---|---|---|---|
| GDPR sub-processor | Anthropic | Mistral AI | None |
| Data residency | AWS region | AWS region | Client machine |
| NIS2 compliant | Requires DPA | Requires DPA | Yes — zero egress |
| Model quality | Frontier | Open-weight 7B | Open-weight 7B |
| Latency | Low | Low | ~4–5x higher |
| Cost crossover | Baseline | ~30x cheaper | ~$0 |
| Ops burden | Low | Low | Medium (hardware) |
| Production-ready | Yes | Yes | With GPU |

**Note on inference profiles:** For UK energy clients under NIS2, use an
`eu.*` inference profile or provisioned throughput in eu-west-2 to
guarantee data residency even with managed Bedrock backends.

---

## Recommendation by Use Case

**Use Claude Sonnet (Bedrock) when:**
- Quality and reliability are primary concerns
- Volume is below ~50M tokens/day
- Client has signed a DPA with Anthropic

**Use Mistral 7B (Bedrock) when:**
- Cost is a primary constraint and frontier quality is not required
- Client is comfortable with open-weight model quality trade-offs
- Volume makes Claude cost prohibitive

**Use Mistral 7B (Sovereign/Ollama) when:**
- Client has NIS2 obligations (energy, critical infrastructure)
- Data sovereignty requirements preclude any external API endpoints
- Async pipeline — latency trade-off is acceptable
- Use GPU hardware (g5.xlarge on EKS/vLLM) for production-grade latency

---

## Files

- `src/explanation/bedrock_explainer.py` — Claude Sonnet via Bedrock
- `src/explanation/mistral_bedrock_explainer.py` — Mistral 7B via Bedrock
- `src/explanation/mistral_local_explainer.py` — Mistral 7B via Ollama (sovereign)
- `src/explanation/ollama_explainer.py` — generic Ollama backend
- `src/explanation/prompts.py`
- `src/detection/anomaly_scorer.py`
- `results/benchmark_raw.jsonl` — raw per-event results

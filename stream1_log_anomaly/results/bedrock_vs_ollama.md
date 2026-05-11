# Bedrock vs Ollama — Benchmark Report
## Stream 1: Log Anomaly Explanation Pipeline

**Date:** 11 May 2026  
**Dataset:** 5 anomaly events from `logs/sample.jsonl` (events 033–037)  
**Anomaly type:** Cascading failure — auth-service → inventory → payment-svc → api-gateway  
**Hardware:** MacBook Pro (local), Ollama running on-device  
**Bedrock region:** us-west-2  

---

## Test Methodology

Both explainers were run against the identical set of 5 anomaly events
produced by Isolation Forest on `sample.jsonl`. Each explainer received
the same prompt template (from `src/explanation/prompts.py`) and the same
±5 context window around each anomalous event.

Results were captured automatically via `src/instrumentation/anomaly_log.py`
and visualised in the Streamlit dashboard (`dashboard.py`).

No manual intervention — both runs were fully automated.

---

## Results

| Metric | Bedrock (claude-sonnet-4-5) | Ollama (llama3.2:3b) |
|---|---|---|
| Events explained | 5 | 5 |
| JSON validity (first pass) | 5/5 (100%) | 2/5 (40%) |
| Retries needed | 0 | 3 |
| Truncated responses | 0 | 1 (missing closing `}`) |
| HIGH severity (correct) | 5/5 | 1/5 |
| MEDIUM severity (incorrect) | 0/5 | 4/5 |
| Cascade reasoning | ✅ Yes | ❌ No |
| Avg explanation latency | 5658ms | 1768ms |
| Data leaves network | Yes (AWS us-west-2) | No |
| Cost per call | ~$0.003 | ~$0 |
| Production-ready | Yes | No |

---

## Qualitative Analysis

### Cascade Reasoning

This is the most significant quality gap.

**Bedrock** understood that auth-service was the cascade origin and traced
propagation through inventory → payment-svc → api-gateway across all 5 events.
By event 033, it had already identified auth-service as the root cause. By
event 036, it correctly described api-gateway experiencing "compounded latency
due to retries or waiting on multiple failing dependencies."

**Ollama** treated each event independently. Event 034 (inventory) was
explained without reference to the auth-service failure that preceded it.
Event 036 (api-gateway) was described as "resource saturation" with no
mention of the upstream cascade. This is a fundamental limitation — the
model is not using the ±5 context window effectively.

### Severity Accuracy

All 5 events are part of a simultaneous multi-service cascading failure
with latency spikes 15–25x above baseline and 500 status codes. The correct
severity is HIGH for all 5.

Bedrock rated all 5 as HIGH. Ollama rated 4 as MEDIUM — likely because it
evaluated each event in isolation rather than in the context of a system-wide
outage.

### JSON Reliability

Bedrock returned valid JSON on the first pass for all 5 calls. Ollama
required retries on 3 of 5 calls and truncated one response (missing closing
`}`). The defensive `_parse_explanation` logic handled this correctly, but
the reliability gap is real and matters in a production pipeline where
retry latency compounds.

---

## Latency Analysis

Ollama was 3.2x faster on average (1768ms vs 5658ms). This is expected:

- Bedrock latency includes network round-trip to us-west-2 + inference time
- Ollama latency is pure local inference on a 3B parameter model
- The latency gap would narrow significantly for a larger local model
  (e.g. llama3.1:8b or qwen3:14b)

However, latency alone does not tell the full story. A faster but incorrect
explanation (wrong severity, no cascade reasoning) has negative operational
value — it misleads the on-call engineer.

---

## Cost Analysis

| | Bedrock | Ollama |
|---|---|---|
| Per-call cost | ~$0.003 | ~$0 |
| 1,000 calls/day | ~$3/day | ~$0/day |
| 10,000 calls/day | ~$30/day | ~$0/day |
| 50M tokens/day crossover | — | EKS GPU ~$3/hr becomes cheaper |

At low-to-medium volume (under ~50M tokens/day), Bedrock wins on total cost
of ownership when you factor in GPU infrastructure, model updates, security
patching, and engineering time for self-hosted inference.

Above ~50M tokens/day, a self-hosted model on EKS (p3.2xlarge ~$3/hr) becomes
cheaper than Bedrock on raw token cost.

---

## Three-Way Decision Framework

| | Bedrock | EKS self-hosted (vLLM) | Ollama (local) |
|---|---|---|---|
| GDPR sub-processor | Anthropic | None | None |
| Data residency | AWS region | Client VPC | Client machine |
| Audit proof | AWS CloudTrail | VPC Flow Logs | Hard to audit |
| Model quality | Full Claude Sonnet | Open-source only | Open-source only |
| Cascade reasoning | ✅ Strong | Depends on model | ❌ Weak (3B) |
| Cost crossover | Baseline | ~50M tokens/day | ~$0 (dev only) |
| Ops burden | Low | High | Very low |
| Production-ready | Yes | Yes | No |
| Sovereign data | No | Yes | Yes (dev only) |

---

## Recommendation by Use Case

**Use Bedrock when:**
- Quality and reliability are the primary concern
- Volume is below ~50M tokens/day
- Client has signed a DPA with Anthropic
- Bedrock in the client's own AWS account with EU region satisfies GDPR

**Use EKS self-hosted (vLLM) when:**
- Client has NIS2 obligations (energy, critical infrastructure)
- Data sovereignty requirements preclude external API endpoints
- Volume exceeds ~50M tokens/day and operational maturity exists
- Regulator requires auditable proof of data residency (VPC Flow Logs)

**Use Ollama when:**
- Local development and testing only
- Proving the sovereign inference pattern before committing to EKS
- Zero-cost exploration of model quality trade-offs

---

## What I Would Do Differently in Production

1. **Run a larger local model for fairer comparison.** llama3.2:3b is the
   smallest sensible model. A fairer Bedrock comparison would be
   llama3.1:8b or qwen3:14b — better cascade reasoning, still local.

2. **Run 30 anomaly events, not 5.** 5 events is enough to observe the
   quality gap but not enough for statistical confidence on latency variance.
   Sprint 4 will run both explainers on 30 synthetic anomaly events.

3. **Add human quality ratings.** The severity accuracy comparison here is
   based on ground truth I know from constructing the dataset. In production,
   you'd need human-rated quality scores (1–5) from on-call engineers who
   reviewed the explanations.

4. **Measure token efficiency.** Bedrock and Ollama may produce explanations
   of very different lengths. Tokens per explanation is a meaningful efficiency
   metric alongside latency and cost.

---

## Files Referenced

- `src/explanation/bedrock_explainer.py` — Bedrock implementation
- `src/explanation/ollama_explama_explainer.py` — Ollama implementation
- `src/explanation/prompts.py` — Shared prompt template (identical for both)
- `src/instrumentation/anomaly_log.py` — Result capture
- `results/anomaly_log.jsonl` — Raw results
- `dashboard.py` — Streamlit visualisation
- `docs/screenshots/dashboard_bedrock.png` — Bedrock dashboard
- `docs/screenshots/dashboard_ollama.png` — Ollama dashboard

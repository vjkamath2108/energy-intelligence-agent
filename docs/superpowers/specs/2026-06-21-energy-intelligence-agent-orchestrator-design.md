# EnergyIntelligenceAgent Orchestrator — Design Spec

**Date:** 2026-06-21
**Task:** W5.1 — EnergyIntelligenceAgent orchestrator
**Output:** `src/agent/energy_agent.py`

---

## Context

The EIA pipeline has three independently-tested components:

- `HybridRetriever.search(query)` → `list[RetrievedDocument]` (BM25 + semantic + RRF)
- `ClaudeRecommender.recommend(query, docs)` / `OllamaRecommender.recommend(query, docs)` → `DispatchRecommendation` (safety + confidence + constraint checks already embedded)
- `AuditLogger.record(entry)` → appends to JSONL audit trail

The orchestrator composes these into a single `dispatch(query) → DispatchRecommendation` call. It is a library class; `main.py` (W5.2) instantiates it and calls `dispatch()`.

---

## Class Structure

```python
@dataclass
class EnergyIntelligenceAgent:
    backend: Literal["bedrock", "ollama"] = "bedrock"
    retriever: HybridRetriever = field(default_factory=HybridRetriever)
    audit_logger: AuditLogger = field(default_factory=AuditLogger)

    def __post_init__(self) -> None:
        self._recommender: ClaudeRecommender | OllamaRecommender = (
            ClaudeRecommender() if self.backend == "bedrock" else OllamaRecommender()
        )

    async def dispatch(self, query: str) -> DispatchRecommendation:
        ...
```

**Design decisions:**
- `backend` is owned by the agent, not by the caller. `main.py` passes `--recommender bedrock|ollama` as a CLI flag and the agent creates the right recommender internally.
- `retriever` and `audit_logger` are dataclass fields so tests can inject mocks without subclassing.
- Single public method: `dispatch()`. No `retrieve()` or `recommend()` exposed — the internal steps are implementation detail.

---

## `dispatch()` Flow

```
dispatch(query)
    │
    ├─ try:
    │   ├─ docs = await retriever.search(query)
    │   └─ rec  = await recommender.recommend(query, docs)
    │
    ├─ except Exception as e:
    │   └─ rec = DispatchRecommendation(
    │               query=query,
    │               action="FALLBACK: system unavailable — manual review required",
    │               confidence=0.0,
    │               supporting_documents=[],
    │               reasoning=f"{type(e).__name__}: {e}",
    │               requires_human_review=True,
    │               hallucination_flagged=False,
    │           )
    │
    └─ always:
        ├─ entry = entry_from_recommendation(rec)
        ├─ await audit_logger.record(entry)
        └─ return rec
```

**Design decisions:**
- One `try/except` covers both retrieval and generation failures. The operator always receives a typed `DispatchRecommendation`, never an unhandled exception.
- The fallback is a fully valid Pydantic object — no sentinel types or `None` returns.
- Audit runs unconditionally (finally-style pattern): the fallback case is on the compliance trail.
- `requires_human_review` on the fallback is always `True`. On the happy path it comes from the recommender's own safety + confidence + constraint logic — the orchestrator does not re-evaluate it.

---

## Graceful Degradation

| Failure mode | Caught by | Result |
|---|---|---|
| DB unreachable — retrieval throws | `except Exception` | Fallback recommendation, `requires_human_review=True` |
| Bedrock throttled beyond retries | `except Exception` | Fallback recommendation, `requires_human_review=True` |
| Invalid JSON after retry | `except Exception` | Fallback recommendation, `requires_human_review=True` |
| Low confidence / hallucination detected | Inside recommender | `requires_human_review=True`, `hallucination_flagged=True`, still a valid recommendation |

The last row is not a failure — the recommender handles it and returns a valid (flagged) recommendation. The orchestrator does not need to know about it.

---

## Testing

Four test cases in `tests/test_agent.py`:

1. **Happy path** — mock retriever returns one doc, mock recommender returns a valid `DispatchRecommendation`. Assert `dispatch()` returns it unchanged and exactly one audit entry is recorded.

2. **Retrieval failure** — mock retriever raises `RuntimeError`. Assert `dispatch()` returns `DispatchRecommendation` with `requires_human_review=True`, `confidence=0.0`, and the audit entry is still recorded.

3. **Recommender failure** — retriever succeeds, recommender raises. Same fallback assertions as above.

4. **Backend selection** — `EnergyIntelligenceAgent(backend="bedrock")` creates `ClaudeRecommender`; `backend="ollama"` creates `OllamaRecommender`. Assert via `type(agent._recommender)`.

Tests mock at the dataclass field level (inject a mock `retriever` and `audit_logger`). They do not mock `ClaudeRecommender` or `OllamaRecommender` internals — those have their own test suites.

---

## Gate

`dispatch()` returns a valid, fully-typed `DispatchRecommendation` end-to-end — confirmed by pytest passing all four test cases.

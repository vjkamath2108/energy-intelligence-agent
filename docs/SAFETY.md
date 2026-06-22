# Safety Design — Energy Intelligence Agent

This document describes the responsible AI mechanisms in the EIA pipeline:
how the system detects hallucinations, calibrates confidence, enforces
regulatory constraints, records every decision, and decides when to escalate
to a human operator.

---

## 1. Why Safety Matters Here

Energy dispatch recommendations are safety-critical and legally binding.
A recommendation that:

- **Contradicts current regulation** exposes the operator to Ofgem fines.
- **Exceeds a BM unit's physical limit** risks equipment damage or grid instability.
- **Is confidently wrong** is more dangerous than an uncertain recommendation
  that a human double-checks.

The safety layer is not a bolt-on. It is in the critical path of every
recommendation before it reaches an operator.

---

## 2. Hallucination Detection

**Location:** `src/recommendation_engine/safety.py` · `check_grounding()`

Every recommendation is checked against the retrieved documents that supported
it before it is returned. Two checks run in sequence.

### 2.1 Term Overlap (always runs)

Measures the fraction of meaningful tokens in the recommended action that
appear verbatim in the retrieved document corpus.

```
overlap = |tokens(action) ∩ tokens(corpus)| / |tokens(action)|
```

Stopwords are excluded. Only tokens of length ≥ 3 count.

| Result | Interpretation |
|--------|----------------|
| overlap ≥ 0.30 | Action vocabulary is grounded in retrieved evidence |
| overlap < 0.30 | Action introduces terms not present in retrieved docs — hallucination risk |

**Threshold:** 0.30 (tuned; lower catches more but produces more false positives on
regulatory language that paraphrases source text).

### 2.2 Semantic Similarity (runs when an embedder is available)

Embeds the full action + reasoning string and the retrieved corpus using
Bedrock Titan V2, then computes cosine similarity.

| Result | Interpretation |
|--------|----------------|
| similarity ≥ 0.50 | Output is semantically consistent with retrieved context |
| similarity < 0.50 | Semantic drift — output may be fabricating facts |

Skipped in sovereign mode (Ollama path) to avoid Bedrock calls when the
client's data must not leave their network. Term overlap alone provides
the grounding check in that mode.

### 2.3 Outcome

If either check fails:
- `hallucination_flagged = True` on the `DispatchRecommendation`
- `requires_human_review = True` (always set alongside hallucination flag)

---

## 3. Confidence Scoring and Calibration

**Location:** `src/recommendation_engine/confidence.py`

The raw confidence value from the LLM is not used directly. It is combined
with retrieval and grounding signals into a composite score, then optionally
calibrated against historical validation data.

### 3.1 Composite Score

Three signals are blended with fixed weights:

| Signal | Weight | Source |
|--------|--------|--------|
| LLM self-confidence | 60% | `confidence` field in LLM JSON response |
| Grounding quality | 25% | Term overlap ratio (+ semantic similarity if available) |
| Retrieval quality | 15% | Mean `similarity_score` of cited documents |

```
composite = 0.60 × llm_confidence
          + 0.25 × grounding_score
          + 0.15 × mean_retrieval_similarity
```

Weight rationale: the LLM has the most signal (it saw the full context);
grounding verifies the output is anchored to retrieved evidence;
retrieval quality reflects how relevant the source documents were.

### 3.2 Platt Scaling Calibration

Raw composite scores are not well-calibrated by default — a score of 0.8
does not mean an 80% chance the recommendation is correct. Platt scaling
fits a logistic transform `σ(A·score + B)` on a labelled validation set
so that predicted probabilities match observed outcomes.

**Gate:** Brier score < 0.20 on the validation set.

```
Brier score = mean((predicted_probability − actual_outcome)²)
```

A Brier score of 0.00 is perfect. A random classifier scores 0.25.
The EIA gate of 0.20 ensures meaningful calibration before the calibrator
is used in production.

When no fitted calibrator is available (cold start, sovereign mode without
validation data), the raw composite score is used directly.

### 3.3 Human Review Threshold

```
requires_human_review = (confidence < 0.60) OR hallucination_flagged OR constraint_violated
```

A confidence threshold of 0.60 was selected as the operating point:
below this the composite signal indicates too much uncertainty for an
automated dispatch recommendation without operator oversight.

---

## 4. Regulatory Constraint Checking

**Location:** `src/safety/constraint_validator.py`

Retrieved regulatory documents are parsed for operational constraints
**before** the recommendation is returned. This catches cases where the
LLM produces a factually grounded recommendation that nonetheless violates
an explicit regulatory rule present in the retrieved context.

### 4.1 Constraint Types Extracted

| Type | Example regulatory language | What is checked |
|------|-----------------------------|-----------------|
| `mw_ceiling` | "must not exceed 500 MW" / "maximum output of 300 MW" | Action MW > ceiling → violation |
| `directional_prohibition` | "upward dispatch instructions are not permitted" | Action direction matches ban → violation |
| `unit_prohibition` | "BM Unit DRAX1G must not be dispatched" | Action names banned unit → violation |

Extraction is sentence-level (regex, no LLM) to limit false positives from
multi-clause sentences. This is consistent with the architectural principle:
*classical methods do structured extraction; LLMs do natural language only.*

### 4.2 Conservative Validation

The MW ceiling check is intentionally conservative: it only fires when the
action contains an **explicit MW value** that exceeds the ceiling. An action
that says "ramp up DRAX1G as needed" does not trigger a ceiling violation
even if a ceiling is in the retrieved docs — the ambiguity is left for a human.

### 4.3 Outcome

Any constraint violation sets `requires_human_review = True`. Violations
are logged at WARNING level with the source document ID and the reason.

---

## 5. Audit Trail

**Location:** `src/audit/audit_log.py`

Every recommendation that passes the safety checks is recorded before being
returned to the operator. The audit trail is:

- **Immutable:** `AuditLogEntry` is `frozen=True` (Pydantic v2). No mutation,
  update, or delete operations are exposed.
- **Append-only:** The JSONL file is opened exclusively in `"a"` mode.
  A new `AuditLogger` instance opened on the same path appends; it never truncates.
- **Concurrency-safe:** An `asyncio.Lock` serialises writes within a process.
- **Compliant format:** One JSON object per line (JSONL), UTF-8, human-readable.

### 5.1 What Each Entry Records

```json
{
  "entry_id": "uuid4",
  "recommendation_id": "uuid4",
  "query": "What action on the North constraint?",
  "retrieved_doc_ids": ["reg-001", "dec-042"],
  "confidence": 0.83,
  "hallucination_detection_result": false,
  "action_taken": "Issue upward BM dispatch to DRAX1G for 200 MW",
  "timestamp": "2026-06-21T14:32:01.123456+00:00"
}
```

The `hallucination_detection_result` field is `true` when hallucination was
detected — it records the *detection outcome*, not the *absence* of hallucination.

### 5.2 Corrections

The audit trail is append-only by design. If a logged recommendation is later
found to be incorrect, a **new** `AuditLogEntry` is created that references the
original `recommendation_id` with a note in `action_taken`. Existing entries
are never modified or deleted.

---

## 6. Decision Tree: What Triggers Human Review

```
                    ┌─────────────────────┐
                    │  Recommendation     │
                    │  generated by LLM   │
                    └──────────┬──────────┘
                               │
              ┌────────────────▼────────────────┐
              │  Term overlap ≥ 0.30?            │
              │  (Semantic similarity ≥ 0.50?)   │
              └────────┬──────────────┬──────────┘
                       │ YES          │ NO
                       ▼             ▼
              ┌─────────────┐   ┌─────────────────────────────┐
              │  Continue   │   │  hallucination_flagged=True  │
              └──────┬──────┘   │  requires_human_review=True  │
                     │          │  → ESCALATE TO OPERATOR      │
                     │          └─────────────────────────────┘
              ┌──────▼──────┐
              │  Composite  │
              │  confidence │
              │  ≥ 0.60?    │
              └──────┬──────┘
                  YES│          NO
                     │          ▼
                     │   ┌──────────────────────────────┐
                     │   │  requires_human_review=True  │
                     │   │  (low confidence)            │
                     │   │  → ESCALATE TO OPERATOR      │
                     │   └──────────────────────────────┘
              ┌──────▼──────┐
              │  Regulatory │
              │  constraints│
              │  pass?      │
              └──────┬──────┘
                  YES│          NO
                     │          ▼
                     │   ┌──────────────────────────────┐
                     │   │  requires_human_review=True  │
                     │   │  (constraint violation)      │
                     │   │  → ESCALATE TO OPERATOR      │
                     │   └──────────────────────────────┘
              ┌──────▼──────┐
              │  Audit log  │
              │  entry      │
              │  written    │
              └──────┬──────┘
                     │
              ┌──────▼──────┐
              │  Return     │
              │  recommendation
              │  to operator│
              └─────────────┘
```

**Summary of escalation triggers:**

| Trigger | Flag set | Condition |
|---------|----------|-----------|
| Term overlap below threshold | `hallucination_flagged` | overlap < 0.30 |
| Semantic similarity below threshold | `hallucination_flagged` | similarity < 0.50 |
| Composite confidence low | `requires_human_review` | confidence < 0.60 |
| Regulatory constraint violated | `requires_human_review` | any violation found |
| No retrieved documents | Both flags | corpus is empty |

---

## 7. Example Traces

### 7.1 Clean Recommendation (passes all checks)

**Query:** "What action should we take on the North constraint at SP28?"

**Retrieved docs:** `decisions/constraint-management-2024-north.pdf`,
`regulations/bsc-section-v.pdf`

**LLM output:**
```json
{
  "action": "Issue upward BM dispatch instruction to DRAX1G for 200 MW in SP28",
  "confidence": 0.87,
  "reasoning": "North region constraint confirmed — 400 MW import shortfall per constraint-management-2024-north.pdf. BSC Section V clause 4.3 authorises emergency dispatch above gate closure.",
  "cited_doc_ids": ["doc-001", "doc-002"]
}
```

**Safety checks:**
```
Term overlap:           0.74  ✓ (≥ 0.30)
Semantic similarity:    0.81  ✓ (≥ 0.50)
Composite confidence:   0.83  ✓ (≥ 0.60)
  breakdown: 0.60×0.87 + 0.25×0.78 + 0.15×0.85 = 0.52 + 0.20 + 0.13
Constraint check:       PASS  ✓ (no ceiling, no ban on DRAX1G, upward permitted)
```

**Result:** `requires_human_review=False`, `hallucination_flagged=False`

**Audit entry written:** ✓

---

### 7.2 Flagged Recommendation — Hallucination Detected

**Query:** "Is DRAX1G available for upward dispatch?"

**Retrieved docs:** `regulations/ner-licence-condition-c17.pdf` (DRAX1G maintenance ban)

**LLM output:**
```json
{
  "action": "Issue upward BM dispatch instruction to DRAX1G for 350 MW",
  "confidence": 0.79,
  "reasoning": "DRAX1G is the cheapest available unit in the North constraint zone.",
  "cited_doc_ids": ["doc-003"]
}
```

**Safety checks:**
```
Term overlap:           0.41  ✓ (≥ 0.30)
Semantic similarity:    0.63  ✓ (≥ 0.50)
Composite confidence:   0.79  ✓ (≥ 0.60)
Constraint check:       FAIL  ✗
  → unit_prohibition: "BM Unit DRAX1G must not be dispatched"
     (source: ner-licence-condition-c17.pdf)
```

**Result:** `requires_human_review=True`, `hallucination_flagged=False`

**Why:** The LLM produced a factually grounded recommendation (correct term
overlap and semantic similarity) but the retrieved regulation explicitly bans
DRAX1G from dispatch. The constraint validator caught what the grounding checks
could not.

**Audit entry written:** ✓ (all decisions are audited, including escalated ones)

---

### 7.3 Flagged Recommendation — Low Confidence

**Query:** "Which unit should handle the South constraint?"

**Retrieved docs:** `regulations/grid-code-section-4.pdf` (no specific dispatch guidance)

**LLM output:**
```json
{
  "action": "Consider dispatching available units in the South constraint zone",
  "confidence": 0.31,
  "reasoning": "Retrieved documents do not specify a preferred unit for this scenario.",
  "cited_doc_ids": ["doc-005"]
}
```

**Safety checks:**
```
Term overlap:           0.52  ✓ (≥ 0.30)
Semantic similarity:    0.58  ✓ (≥ 0.50)
Composite confidence:   0.41  ✗ (< 0.60)
  breakdown: 0.60×0.31 + 0.25×0.55 + 0.15×0.72 = 0.19 + 0.14 + 0.11
Constraint check:       PASS  ✓
```

**Result:** `requires_human_review=True`, `hallucination_flagged=False`

**Why:** The LLM correctly identified that it did not have sufficient evidence.
Low self-reported confidence combined with low retrieval quality pulled the
composite score below 0.60. The system escalates rather than guessing.

**Audit entry written:** ✓

---

## 8. What the System Does Not Do

| Claim | Reality |
|-------|---------|
| Detects all hallucinations | Term overlap catches vocabulary drift; semantic similarity catches meaning drift. Neither is perfect. Both together raise the bar significantly. |
| Guarantees regulatory compliance | The constraint validator catches constraints it can *extract* via regex from retrieved text. A constraint in a document that was not retrieved, or expressed in unusual language, may be missed. |
| Eliminates human judgment | The human review flag is the point — it identifies the cases where human judgment is required. It is not a claim that automated cases are error-free. |
| Self-corrects | Audit entries are immutable. Corrections require new entries. The system does not amend or retry on its own. |

# Chunking Strategy — Energy Intelligence Agent

## 1. Overview

Chunking strategy determines the granularity of text units stored in the vector
database and retrieved during RAG generation. For energy domain documents — which
are structured around numbered Articles, Clauses and Sections that each express a
single obligation — the unit of retrieval should align with the unit of meaning.
A chunk that straddles two clauses forces the LLM to reason across an arbitrary
boundary, increasing the risk of omitting numeric thresholds or misattributing
conditions to the wrong obligation.

The three strategies below were evaluated against 62 synthetic energy documents
across three categories:

- `regulations/` — grid codes, DNO standards, licence conditions
- `tariffs/` — TNUoS, BSUoS, DUoS charge schedules
- `decisions/` — historical dispatch and balancing action records

---

## 2. Strategies Evaluated

### 2.1 FixedSizeChunker (512 tokens, 50 overlap)

Splits text into fixed token windows with a 50-token overlap between consecutive
chunks. The overlap is intended to preserve context at boundaries, but the split
point is determined solely by token count — it has no awareness of sentence or
clause structure.

**Measured metrics (62 documents):**
- Total chunks: 62
- Avg chunk size: 2,192 chars
- Max: 3,002 chars | Min: 1,608 chars
- Overlap: 9.8%
- Throughput: 38,515 docs/sec

**Why it fails for regulatory text:** Each of the 62 documents fits within the
512-token window, so the chunker produces exactly one chunk per document. This
means a query for a specific clause retrieves the entire document. For longer
documents that do overflow the window, the cut falls mid-sentence based on token
count alone — a threshold and its antecedent condition end up in different chunks.

---

### 2.2 SlidingWindowChunker (256 window, 128 step)

Produces overlapping windows with a 128-token stride, so consecutive chunks share
50% of their tokens. The intent is to ensure no sentence falls entirely outside a
retrieval window.

**Measured metrics (62 documents):**
- Total chunks: 144
- Avg chunk size: 1,418 chars
- Max: 1,810 chars | Min: 864 chars
- Overlap: 50.0%
- Throughput: 39,840 docs/sec

**Trade-off:** The 50% overlap doubles storage and retrieval cost. The retriever
returns near-duplicate chunks for the same passage, which wastes the
`supporting_documents` list and inflates LLM context with repeated text. The
hallucination detector also sees the same content twice with minor prefix/suffix
differences, corrupting the term-overlap signal.

---

### 2.3 SemanticChunker (max 800 tokens)

Splits text in three passes, in priority order:

1. **Regulatory section boundaries** — `Section N`, `Article N`, `Clause N`,
   `Appendix N`, `Schedule N`, `Part N`, or a markdown heading (`#`, `##`)
2. **Paragraph breaks** — blank lines within a section, with small paragraphs
   merged until each chunk reaches ~80 tokens to avoid sub-100-char fragments
3. **Sentence boundaries** — only applied when a merged segment still exceeds
   the 800-token cap

This means the default unit of retrieval is a complete regulatory section, not an
arbitrary window.

**Measured metrics (62 documents):**
- Total chunks: 246
- Avg chunk size: 552 chars
- Max: 715 chars | Min: 312 chars
- Overlap: 0.0%
- Throughput: 15,504 docs/sec

**Why it wins for energy domain:** Regulatory text is written to be read at
clause granularity. Each numbered clause is a self-contained obligation with its
own conditions and numeric thresholds. Splitting at clause boundaries produces
chunks that are both complete retrieval units and verifiable citation targets.

---

## 3. Comparison Table

| Strategy              | Total Chunks | Avg (chars) | Max (chars) | Min (chars) | Overlap | Docs/sec |
|-----------------------|:------------:|:-----------:|:-----------:|:-----------:|:-------:|:--------:|
| FixedSize (512/50)    |      62      |    2,192    |    3,002    |    1,608    |   9.8%  |  38,515  |
| Sliding (256/128)     |     144      |    1,418    |    1,810    |      864    |  50.0%  |  39,840  |
| **Semantic (max=800)**|   **246**    |    **552**  |    **715**  |    **312**  | **0.0%**| 15,504   |

Measured on 62 synthetic energy documents, 2026-05-22.

---

## 4. Concrete Example

**Input text:**

```
Section 2 — Voltage and Reactive Power Requirements

2.1 All transmission-connected generators must maintain operation within
the voltage range 11 kV to 400 kV as measured at the grid supply point (GSP).

2.2 The reactive power capability requirement is defined as the ability to
absorb or generate reactive power in accordance with the P–Q capability
diagram specified in Schedule 1.

2.3 Where a generating unit cannot comply with the reactive capability
requirements, the participant shall notify the System Operator no later
than 15 Business Days before the scheduled outage or modification date.
```

---

**FixedSizeChunker** — document fits within 512 tokens, so this entire section
plus surrounding content is returned as one undifferentiated chunk. For a longer
document that overflows, the cut falls mid-clause:

```
Chunk N (tokens 0–511):
  "...2.1 All transmission-connected generators must maintain operation within
  the voltage range 11 kV to 400 kV as measured at the grid supply point (GSP).
  2.2 The reactive power capability requirement is defined as the ability to
  absorb or generate reactive power in accordance with the P–Q capability"
  ← cuts mid-clause 2.2, threshold in Schedule 1 is in the next chunk

Chunk N+1 (tokens 461–end, 50-token overlap):
  "  absorb or generate reactive power in accordance with the P–Q capability
  diagram specified in Schedule 1.
  2.3 Where a generating unit cannot comply..."
  ← clause 2.2 is now split across two chunks; neither is a clean citation unit
```

---

**SemanticChunker** — splits at the `Section 2` boundary, then merges the
three sub-clauses into one coherent chunk (combined token count stays under 800):

```
Chunk N:
  "Section 2 — Voltage and Reactive Power Requirements

  2.1 All transmission-connected generators must maintain operation within
  the voltage range 11 kV to 400 kV as measured at the grid supply point (GSP).

  2.2 The reactive power capability requirement is defined as the ability to
  absorb or generate reactive power in accordance with the P–Q capability
  diagram specified in Schedule 1.

  2.3 Where a generating unit cannot comply with the reactive capability
  requirements, the participant shall notify the System Operator no later
  than 15 Business Days before the scheduled outage or modification date."
```

A query for *"reactive power capability requirements"* retrieves this chunk —
a complete, self-contained section with a verifiable citation to clause 2.2.

---

## 5. Decision

**SemanticChunker selected.**

- Regulatory obligations are clause-scoped: each numbered clause expresses a
  single condition and its threshold. Splitting at clause boundaries preserves
  that unit, so retrieved chunks are directly quotable as evidence.
- The 246-chunk count at 552 chars average produces a retrieval index of
  manageable size (projects to ~4,000 chunks for 1,000 production documents)
  with no redundant overlap storage cost.
- Hallucination detection relies on term overlap between the LLM output and
  retrieved chunks. Complete clauses have stable, self-consistent vocabulary;
  mid-sentence fragments leak terms across chunks and corrupt the overlap signal.

This decision would change if documents were unstructured field notes or
free-text incident reports with no numbered clause structure — in that case
SlidingWindowChunker's recall-oriented overlap would be the better trade-off.

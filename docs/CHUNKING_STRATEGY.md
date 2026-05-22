# Chunking Strategy — Energy Intelligence Agent

## Measured Metrics (100 synthetic documents, run 2026-05-20)

| Strategy | Total Chunks | Avg Size (chars) | Max (chars) | Min (chars) | Overlap % | Docs/sec |
|---|---|---|---|---|---|---|
| FixedSize (512 tok / 50 overlap) | 62 | 2,192 | 3,002 | 1,608 | 9.8% | ~44,000 |
| SlidingWindow (256 tok / 128 step) | 144 | 1,418 | 1,810 | 864 | 50.0% | ~49,000 |
| **SemanticChunker (max 800 tok)** | **1,320** | **102** | **343** | **9** | **0%** | **~9,900** |

The tiny minimum (9 chars) is a document title heading (`# BSC SECTION V`) — a
legitimate chunk that becomes a section anchor, not a bug.

---

## Decision: SemanticChunker

**SemanticChunker is the production strategy.** Reasons:

### 1. Regulatory text is boundary-structured, not length-structured

Energy regulations are written as numbered Articles, Sections and Clauses that
each express a single, complete obligation:

> *"3.1 Primary frequency response shall be delivered within 10 seconds of the
> frequency deviation event and maintained for a minimum of 20 seconds."*

This is a self-contained retrieval unit. A fixed-size splitter cuts across it
mid-sentence based on token budget alone, producing fragments that quote half
an obligation without its numeric threshold — exactly the hallucination surface
the system is designed to avoid.

### 2. Citation accuracy

When the LLM cites a retrieved chunk as evidence for a dispatch recommendation,
the operator sees `source + chunk_index`. A chunk that starts at the beginning
of *Section 3 — Frequency Response Obligations* is a verifiable reference.
A chunk that starts mid-sentence in the middle of Section 3 is not.

### 3. Hallucination detection

The term-overlap hallucination check (`>=30%` overlap) works better when chunks
contain complete sentences. Half-sentences leak vocabulary across chunks and
corrupt the overlap signal.

---

## Trade-offs

### FixedSizeChunker — why it fails for regulatory text

| Pro | Con |
|---|---|
| Fastest (~44k docs/sec) | Splits mid-clause and mid-sentence |
| Predictable chunk count | Destroys regulatory structure |
| Zero boundary logic needed | Citation points to arbitrary text positions |

For energy regulations the biggest problem is **numeric thresholds and their
antecedents**. The threshold in `3.1` (`10 seconds`) has no meaning without
the sentence that precedes it. A fixed-size splitter that cuts at token 512
may put the threshold in chunk N and its condition in chunk N-1, and the
retriever may return only one of them.

### SlidingWindowChunker — why it fails despite high overlap

50% overlap sounds like it protects context, but it **doubles storage and
retrieval cost** while still cutting mid-sentence. Each chunk now appears in
two overlapping windows, so the retriever returns near-duplicate chunks that
waste the `supporting_documents` list and inflate the LLM context. The
hallucination detector also sees the same text twice with minor prefix/suffix
differences, which muddies the overlap score.

### SemanticChunker — trade-offs accepted

The SemanticChunker is ~5x slower than fixed-size and produces ~21x more
chunks. For an offline ingestion pipeline that runs once per regulation update
(monthly) this is acceptable. The retrieval benefit justifies the ingestion cost.

The 0% mechanical overlap does not mean zero semantic overlap — section
headings are included at the start of each Section chunk, so context is
preserved at boundaries without the storage penalty of sliding windows.

---

## Concrete Example: Same Regulatory Paragraph, Two Strategies

**Source paragraph** (from `regulations/eso-balancing-code-bc1.pdf`):

```
Section 3 - Frequency Response Obligations

3.1 Primary frequency response shall be delivered within 10 seconds of
the frequency deviation event and maintained for a minimum of 20 seconds.

3.2 Secondary frequency response shall be delivered within 30 seconds
and maintained until the System Operator issues a stand-down instruction.

3.3 Participants providing FFR services must ensure their control systems
are capable of automated response without manual intervention.
```

---

**FixedSizeChunker (512 tokens, 50 overlap)** — the entire document fits in
one chunk (444 tokens), so the splitter never cuts. For slightly longer
documents it produces two chunks split at an arbitrary token boundary:

```
Chunk 0 (tokens 0-511):
  "...3.2 Secondary frequency response shall be delivered within 30 seconds
  and maintained until the System Operator issues a stand-down instruction.
  3.3 Participants providing FFR services must ensure their control systems"
  <- cuts mid-section

Chunk 1 (tokens 461-end, 50-token overlap):
  "...3.3 Participants providing FFR services must ensure their control systems
  are capable of automated response without manual intervention. Clause 4..."
  <- duplicates 50 tokens from above, still mid-context
```

A query for *"how long must primary frequency response be maintained?"*
retrieves Chunk 0 — which contains the answer (`20 seconds`) — but also
retrieves Chunk 1 which repeats the same text with a 50-token prefix. The
LLM receives near-duplicate evidence.

---

**SemanticChunker (max 800 tokens)** — splits at the `Section 3` header and
each numbered sub-clause boundary:

```
Chunk N   (12 tokens): "Section 3 - Frequency Response Obligations"
Chunk N+1 (28 tokens): "3.1 Primary frequency response shall be delivered
                         within 10 seconds of the frequency deviation event
                         and maintained for a minimum of 20 seconds."
Chunk N+2 (23 tokens): "3.2 Secondary frequency response shall be delivered
                         within 30 seconds and maintained until the System
                         Operator issues a stand-down instruction."
Chunk N+3 (22 tokens): "3.3 Participants providing FFR services must ensure
                         their control systems are capable of automated response
                         without manual intervention."
```

A query for *"how long must primary frequency response be maintained?"* retrieves
Chunk N+1 only — a complete, self-contained obligation with a verifiable
citation to `eso-balancing-code-bc1.pdf`, chunk index N+1.

---

## Future: IVFFlat Index

Once embeddings are populated in W2.1, rebuild the vector index with:

```sql
CREATE INDEX idx_chunks_embedding
    ON document_chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);   -- rule of thumb: sqrt(total_chunks)
```

The semantic chunk count (~1,300 per 100 docs) scales to ~13,000 for 1,000
production documents, so `lists = 115` is the correct starting point.

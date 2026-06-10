---
name: run-evals
description: Runs the EIA retrieval evaluation suite (Precision@5, NDCG@5, Hit Rate, latency). Gates on hybrid Precision@5 ≥ 0.70. Use when the user wants to check retrieval quality, verify the hybrid retriever gate, or compare BM25 vs semantic vs hybrid.
disable-model-invocation: true
---

Run the retrieval evaluation suite using the shell command below. Do NOT generate or invent results — use `!` to execute the real eval and report the actual output.

## Steps

1. Run the eval:

```
! cd /Users/vijaykamath/vijay/projects/energy-intelligence-agent && AWS_PROFILE=containers python -m eval.retrieval_evals
```

2. Parse the output table and report:
   - Hybrid Precision@5 score
   - Whether the gate passed (≥ 0.70) or failed
   - BM25 vs semantic vs hybrid comparison (which retriever led on what query type)
   - Latency p95 — flag if > 500ms

3. If the gate **passes**: confirm it and note the score.

4. If the gate **fails**: identify which query categories are dragging the score (BM25-favoured / semantic-favoured / mixed) and suggest the most likely fix:
   - Low BM25 scores → check tsvector index, re-ingest
   - Low semantic scores → check embeddings populated, HNSW index built
   - Low hybrid scores → check RRF fusion, fetch_k_multiplier

## Rules

- Never fabricate scores or a results table — only report what the command actually printed
- If the command fails (DB unreachable, no embeddings), report the error clearly and suggest the fix:
  - `asyncpg.exceptions.ConnectionRefusedError` → DB not running or wrong DB_HOST
  - `NullValueNotAllowed on embedding` → embeddings not populated, run `python -m src.retriever.embeddings`
  - `AWS credentials` error → check AWS_PROFILE env var
- This skill requires a live PostgreSQL DB and AWS Bedrock (office laptop only)
- After reporting results, ask the user if they want to mark the Week 2 Checkpoint complete in Todoist

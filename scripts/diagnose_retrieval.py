"""Retrieval diagnostics — run this when eval/retrieval_evals.py fails the gate.

Checks:
  1. DB chunk counts + tsvector/embedding coverage
  2. Source file inventory
  3. Direct BM25 probe on known-failing queries
  4. Direct semantic probe on same queries

Usage:
    AWS_PROFILE=containers python scripts/diagnose_retrieval.py
"""

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.db import close_pool, get_connection
from src.retriever.bm25 import BM25Retriever

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

_PROBE_QUERIES = [
    ("BM25-favoured",  "TNUoS tariff 2024 generation charges"),
    ("BM25-favoured",  "BSUoS balancing services use of system charges"),
    ("BM25-favoured",  "DUoS red band amber green unit charges"),
    ("semantic",       "how much does the system operator spend balancing the grid"),
    ("mixed",          "upward balancing action constraint zone offer accepted MWh"),
]


async def check_db() -> bool:
    print("\n── 1. DB coverage ─────────────────────────────────────────")
    conn = await get_connection()

    total   = await conn.fetchval("SELECT COUNT(*) FROM document_chunks")
    no_fts  = await conn.fetchval("SELECT COUNT(*) FROM document_chunks WHERE fts IS NULL")
    no_emb  = await conn.fetchval("SELECT COUNT(*) FROM document_chunks WHERE embedding IS NULL")

    print(f"  Total chunks   : {total}")
    print(f"  Missing fts    : {no_fts}  {'⚠ BM25 will return 0' if no_fts else '✓'}")
    print(f"  Missing embed  : {no_emb}  {'⚠ semantic will return 0' if no_emb else '✓'}")

    print("\n── 2. Source inventory (first 25) ─────────────────────────")
    rows = await conn.fetch(
        "SELECT source, COUNT(*) as chunks FROM document_chunks "
        "GROUP BY source ORDER BY source LIMIT 25"
    )
    if rows:
        for r in rows:
            print(f"  {r['chunks']:>3} chunks  {r['source']}")
    else:
        print("  ⚠ No rows — table is empty. Run: python -m src.data_pipeline.ingest")

    healthy = total > 0 and no_fts == 0
    return healthy


async def probe_bm25() -> None:
    print("\n── 3. BM25 probe ───────────────────────────────────────────")
    bm25 = BM25Retriever()
    for tag, query in _PROBE_QUERIES:
        results = await bm25.search(query, k=3)
        hits = len(results)
        status = "✓" if hits > 0 else "✗ 0 hits"
        print(f"  [{tag:<16}] {status}  {query[:55]!r}")
        for d in results:
            print(f"      → {d.source}  score={d.similarity_score:.4f}")
            print(f"        {d.content[:90].replace(chr(10), ' ')}...")


async def fix_fts() -> None:
    """Apply BM25 schema (fts column + GIN index) if missing."""
    print("\n── Applying BM25 schema (fts column + GIN index)… ─────────")
    bm25 = BM25Retriever()
    await bm25.apply_schema()
    conn = await get_connection()
    no_fts = await conn.fetchval("SELECT COUNT(*) FROM document_chunks WHERE fts IS NULL")
    print(f"  Missing fts after apply_schema: {no_fts}")


async def main() -> None:
    try:
        healthy = await check_db()
        if not healthy:
            print("\n⚠ fts column has NULLs — applying schema fix automatically…")
            await fix_fts()
        await probe_bm25()
    finally:
        await close_pool()

    print("\n── Next steps ──────────────────────────────────────────────")
    print("  If BM25 still returns 0 after schema fix:")
    print("    → synthetic docs may not contain expected terminology")
    print("    → check synthetic_docs.py DUoS/tariff content vs eval queries")
    print("    → re-ingest: python -m src.data_pipeline.ingest")
    print("  If embeddings are missing:")
    print("    → python -m src.retriever.embeddings")
    print("  After any fix, re-run: python -m eval.retrieval_evals")


if __name__ == "__main__":
    asyncio.run(main())

"""Document ingestion pipeline with chunking strategy A/B comparison.

Workflow:
  1. Generate ~100 synthetic energy documents.
  2. Run FixedSizeChunker, SlidingWindowChunker, and SemanticChunker on all docs.
  3. Print a formatted comparison table of chunk metrics.
  4. Store SemanticChunker output into PostgreSQL document_chunks table.

Standalone:
  python src/data_pipeline/ingest.py           # full run (requires DB)
  python src/data_pipeline/ingest.py --dry-run # comparison table only, no DB write
"""

import argparse
import asyncio
import json
import logging
import time
from dataclasses import dataclass

from src.data_pipeline.chunkers import (
    BaseChunker,
    FixedSizeChunker,
    SemanticChunker,
    SlidingWindowChunker,
)
from src.data_pipeline.synthetic_docs import SyntheticDocument, generate_documents
from src.db import close_pool, get_connection
from src.models import RetrievedDocument

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass
class ChunkMetrics:
    """Measured performance and size metrics for one chunking strategy."""

    strategy: str
    total_chunks: int
    avg_chunk_size: float       # characters
    max_chunk_size: int         # characters
    min_chunk_size: int         # characters
    overlap_pct: float          # estimated: (overlap_tokens / chunk_size) * 100
    docs_per_second: float


def _measure(
    strategy_name: str,
    chunker: BaseChunker,
    docs: list[SyntheticDocument],
    overlap_tokens: int,
    chunk_size_tokens: int,
) -> tuple[ChunkMetrics, list[RetrievedDocument]]:
    """Run chunker over all docs, measure timing and size distribution.

    Args:
        strategy_name:    Human-readable label for the comparison table.
        chunker:          Chunker instance to benchmark.
        docs:             List of synthetic documents.
        overlap_tokens:   Token overlap used by this strategy (for overlap_pct).
        chunk_size_tokens: Nominal token window (for overlap_pct denominator).

    Returns:
        Tuple of (ChunkMetrics, all produced chunks).
    """
    all_chunks: list[RetrievedDocument] = []
    t0 = time.perf_counter()

    for doc in docs:
        all_chunks.extend(chunker.chunk(doc["content"], doc["source"]))

    elapsed = time.perf_counter() - t0

    sizes = [len(c.content) for c in all_chunks]
    overlap_pct = (overlap_tokens / chunk_size_tokens * 100) if chunk_size_tokens else 0.0

    return (
        ChunkMetrics(
            strategy=strategy_name,
            total_chunks=len(all_chunks),
            avg_chunk_size=sum(sizes) / len(sizes) if sizes else 0.0,
            max_chunk_size=max(sizes) if sizes else 0,
            min_chunk_size=min(sizes) if sizes else 0,
            overlap_pct=overlap_pct,
            docs_per_second=len(docs) / elapsed if elapsed > 0 else 0.0,
        ),
        all_chunks,
    )


def print_comparison_table(metrics: list[ChunkMetrics]) -> None:
    """Print a formatted ASCII comparison table of chunking metrics.

    Args:
        metrics: List of ChunkMetrics, one per strategy.
    """
    col_w = [22, 13, 16, 14, 14, 12, 18]
    headers = [
        "Strategy", "Total Chunks", "Avg Size (chars)",
        "Max (chars)", "Min (chars)", "Overlap %", "Docs/second",
    ]
    sep = "+" + "+".join("-" * w for w in col_w) + "+"
    header_row = "|" + "|".join(
        h.center(w) for h, w in zip(headers, col_w)
    ) + "|"

    print("\n" + sep)
    print(header_row)
    print(sep)

    for m in metrics:
        row = "|" + "|".join([
            m.strategy.center(col_w[0]),
            str(m.total_chunks).center(col_w[1]),
            f"{m.avg_chunk_size:,.0f}".center(col_w[2]),
            str(m.max_chunk_size).center(col_w[3]),
            str(m.min_chunk_size).center(col_w[4]),
            f"{m.overlap_pct:.1f}%".center(col_w[5]),
            f"{m.docs_per_second:.1f}".center(col_w[6]),
        ]) + "|"
        print(row)

    print(sep + "\n")


# ---------------------------------------------------------------------------
# Database operations
# ---------------------------------------------------------------------------

_INSERT_SQL = """
INSERT INTO document_chunks
    (doc_id, content, source, chunk_index, similarity_score, metadata)
VALUES ($1, $2, $3, $4, $5, $6)
ON CONFLICT (doc_id) DO UPDATE
    SET content          = EXCLUDED.content,
        source           = EXCLUDED.source,
        chunk_index      = EXCLUDED.chunk_index,
        similarity_score = EXCLUDED.similarity_score,
        metadata         = EXCLUDED.metadata
"""


async def _ensure_schema(conn: object) -> None:
    """Create document_chunks table and indexes if they do not exist.

    Args:
        conn: An asyncpg connection with pgvector already registered.
    """
    import pathlib
    schema_path = pathlib.Path(__file__).parent / "schema.sql"
    ddl = schema_path.read_text()
    # Execute non-empty statements only, skip comment-only lines
    for statement in ddl.split(";"):
        stmt = statement.strip()
        if stmt and not stmt.startswith("--"):
            await conn.execute(stmt)  # type: ignore[attr-defined]


async def store_chunks(chunks: list[RetrievedDocument]) -> None:
    """Insert or upsert chunks into the document_chunks table.

    Uses executemany for batched inserts. Runs the schema DDL first
    so the table is created if absent.

    Args:
        chunks: List of RetrievedDocument instances to persist.
    """
    logger.info("Storing %d chunks into document_chunks …", len(chunks))
    async with get_connection() as conn:
        await _ensure_schema(conn)

        rows = [
            (
                c.doc_id,
                c.content,
                c.source,
                c.chunk_index,
                c.similarity_score,
                json.dumps(c.metadata),
            )
            for c in chunks
        ]
        await conn.executemany(_INSERT_SQL, rows)

    logger.info("Stored %d chunks successfully.", len(chunks))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


async def run_pipeline(dry_run: bool = False) -> None:
    """Full ingestion pipeline: generate → compare → store.

    Args:
        dry_run: When True, skip the database write step.
    """
    # 1. Generate documents
    logger.info("Generating synthetic documents …")
    docs = generate_documents()
    logger.info("Generated %d documents.", len(docs))

    # 2. Run all three chunkers and collect metrics
    fixed_chunker = FixedSizeChunker(chunk_size=512, overlap=50)
    sliding_chunker = SlidingWindowChunker(window_size=256, step=128)
    semantic_chunker = SemanticChunker(max_chunk_size=800)

    fixed_metrics, fixed_chunks = _measure(
        "FixedSize(512/50)", fixed_chunker, docs,
        overlap_tokens=50, chunk_size_tokens=512,
    )
    sliding_metrics, sliding_chunks = _measure(
        "Sliding(256/128)", sliding_chunker, docs,
        overlap_tokens=128, chunk_size_tokens=256,
    )
    semantic_metrics, semantic_chunks = _measure(
        "Semantic(max=800)", semantic_chunker, docs,
        overlap_tokens=0, chunk_size_tokens=800,
    )

    logger.info(
        "Chunks produced — Fixed: %d  Sliding: %d  Semantic: %d",
        fixed_metrics.total_chunks,
        sliding_metrics.total_chunks,
        semantic_metrics.total_chunks,
    )

    # 3. Print comparison table
    print_comparison_table([fixed_metrics, sliding_metrics, semantic_metrics])

    # 4. Store SemanticChunker output
    if dry_run:
        logger.info("--dry-run: skipping database write.")
        return

    await store_chunks(semantic_chunks)
    await close_pool()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Energy document ingestion pipeline")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print comparison table without writing to the database.",
    )
    args = parser.parse_args()
    asyncio.run(run_pipeline(dry_run=args.dry_run))

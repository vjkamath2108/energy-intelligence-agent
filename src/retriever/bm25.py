"""BM25 full-text retriever using PostgreSQL tsvector / ts_rank_cd.

PostgreSQL's tsvector + GIN index + ts_rank_cd implements BM25-style keyword
scoring natively.  No external library is required.

Public API:
  BM25Retriever.search()         — multi-word keyword search
  BM25Retriever.search_phrase()  — exact phrase search
  BM25Retriever.apply_schema()   — idempotent DDL: fts column + GIN index
"""

import asyncio
import logging
import pathlib

from src.db import close_pool, get_connection
from src.models import RetrievedDocument

logger = logging.getLogger(__name__)

# Path to the schema file that contains the BM25 DDL additions.
_SCHEMA_PATH = pathlib.Path(__file__).parent.parent / "data_pipeline" / "schema.sql"

# Only the BM25-specific statements need to run in apply_schema(); the base
# table DDL is handled at deploy time.
_BM25_DDL = """\
ALTER TABLE document_chunks
    ADD COLUMN IF NOT EXISTS fts tsvector
    GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;

CREATE INDEX IF NOT EXISTS idx_chunks_fts
    ON document_chunks USING gin(fts);"""

_SELECT_COLS = "doc_id, content, source, chunk_index, metadata"


class BM25Retriever:
    """Keyword retriever backed by PostgreSQL tsvector and ts_rank_cd.

    Uses plainto_tsquery for regular searches (safe with arbitrary user input)
    and phraseto_tsquery for exact-phrase matching.  Both respect the
    configured *language* dictionary so stop-words and stemming are consistent
    with the fts generated column.

    Args:
        language: PostgreSQL text-search configuration name (default "english").
    """

    def __init__(self, language: str = "english") -> None:
        self._language = language

    async def search(
        self,
        query: str,
        k: int = 5,
        threshold: float = 0.0,
    ) -> list[RetrievedDocument]:
        """Keyword search using plainto_tsquery.

        plainto_tsquery tokenises *query* and ANDs the stems together, so
        "BSUoS balancing charges" matches documents containing all three stems.
        Special characters are ignored — no injection risk.

        Args:
            query: Natural-language or keyword query string.
            k: Maximum number of results to return.
            threshold: Minimum ts_rank_cd score; rows below this are discarded.

        Returns:
            List of RetrievedDocument ordered by descending rank score.
        """
        if not query.strip():
            return []

        async with get_connection() as conn:
            rows = await conn.fetch(
                f"""
                SELECT
                    {_SELECT_COLS},
                    ts_rank_cd(fts, plainto_tsquery($1, $2)) AS rank
                FROM document_chunks
                WHERE fts @@ plainto_tsquery($1, $2)
                  AND ts_rank_cd(fts, plainto_tsquery($1, $2)) >= $3
                ORDER BY rank DESC
                LIMIT $4
                """,
                self._language,
                query,
                threshold,
                k,
            )

        return [_row_to_doc(row) for row in rows]

    async def search_phrase(
        self,
        phrase: str,
        k: int = 5,
    ) -> list[RetrievedDocument]:
        """Exact-phrase search using phraseto_tsquery.

        Matches documents where the words in *phrase* appear in sequence,
        making it suitable for energy domain multi-word terms such as
        "balancing mechanism" or "frequency response".

        Args:
            phrase: Exact phrase to search for.
            k: Maximum number of results to return.

        Returns:
            List of RetrievedDocument ordered by descending rank score.
        """
        if not phrase.strip():
            return []

        async with get_connection() as conn:
            rows = await conn.fetch(
                f"""
                SELECT
                    {_SELECT_COLS},
                    ts_rank_cd(fts, phraseto_tsquery($1, $2)) AS rank
                FROM document_chunks
                WHERE fts @@ phraseto_tsquery($1, $2)
                ORDER BY rank DESC
                LIMIT $3
                """,
                self._language,
                phrase,
                k,
            )

        return [_row_to_doc(row) for row in rows]

    async def apply_schema(self) -> None:
        """Apply the BM25 DDL additions idempotently.

        Adds the fts generated column and GIN index to document_chunks if they
        do not already exist.  Safe to call on every startup.

        Raises:
            asyncpg.PostgresError: If the DDL fails for a reason other than
                the object already existing.
        """
        async with get_connection() as conn:
            await conn.execute(_BM25_DDL)
        logger.info("BM25 schema applied — fts column and GIN index ready")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _row_to_doc(row: object) -> RetrievedDocument:
    """Map an asyncpg Record to a RetrievedDocument.

    Args:
        row: asyncpg Record with columns matching _SELECT_COLS + rank.

    Returns:
        RetrievedDocument with similarity_score set from ts_rank_cd rank.
    """
    import json

    metadata_raw = row["metadata"]  # type: ignore[index]
    if not metadata_raw:
        metadata: dict[str, str] = {}
    elif isinstance(metadata_raw, str):
        metadata = json.loads(metadata_raw)
    else:
        metadata = dict(metadata_raw)

    return RetrievedDocument(
        doc_id=row["doc_id"],  # type: ignore[index]
        content=row["content"],  # type: ignore[index]
        source=row["source"],  # type: ignore[index]
        chunk_index=row["chunk_index"],  # type: ignore[index]
        similarity_score=float(row["rank"]),  # type: ignore[index]
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

_TEST_QUERIES = [
    "BSUoS balancing services charges",
    "frequency response obligations",
    "TNUoS tariff 2024",
    "dispatch instruction balancing mechanism",
]


async def _smoke_test() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )
    retriever = BM25Retriever()
    try:
        await retriever.apply_schema()

        for query in _TEST_QUERIES:
            results = await retriever.search(query, k=3)
            print(f"\nQuery: {query!r}")
            if not results:
                print("  (no results)")
            for doc in results:
                print(
                    f"  [bm25={doc.similarity_score:.4f}] {doc.source}"
                    f"  chunk={doc.chunk_index}"
                )
        # Side-by-side note: BM25 ranks on exact term frequency; compare with
        # semantic scores from W2.1 which capture paraphrase similarity.
    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(_smoke_test())

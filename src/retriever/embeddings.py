"""Bedrock Titan Embeddings V2 pipeline for populating pgvector embeddings.

Three entry points:
  - embed_all_chunks()   — populate NULL embedding rows in document_chunks
  - build_hnsw_index()   — create cosine HNSW index on the embedding column
  - search_similar()     — embed a query and return ranked RetrievedDocuments
"""

import asyncio
import json
import logging
import time
from typing import Any, cast

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from src.db import close_pool, get_connection
from src.models import RetrievedDocument

logger = logging.getLogger(__name__)

_TITAN_MODEL_ID = "amazon.titan-embed-text-v2:0"
# Titan V2 supports 256, 512, or 1024 — not 1536 (that's OpenAI's dimension)
_EMBEDDING_DIM = 1024


def _parse_metadata(value: Any) -> dict[str, str]:
    """Coerce a JSONB column value to a plain dict.

    asyncpg may return JSONB as a raw JSON string rather than a decoded dict
    depending on codec registration.  Handles str, dict, and None.

    Args:
        value: Raw value from asyncpg row.

    Returns:
        Decoded dict, or empty dict when value is None/empty.
    """
    if not value:
        return {}
    if isinstance(value, str):
        return json.loads(value)  # type: ignore[no-any-return]
    return dict(value)


class TitanEmbedder:
    """Wrapper around Bedrock Titan Embeddings V2.

    Args:
        region: AWS region hosting the Bedrock endpoint.
    """

    def __init__(self, region: str = "eu-west-2") -> None:
        # max_pool_connections matches batch_size so boto3 doesn't discard connections
        self._client = boto3.client(
            "bedrock-runtime",
            region_name=region,
            config=Config(max_pool_connections=20),
        )

    async def embed(self, text: str) -> list[float]:
        """Embed a single string using Titan V2.

        Args:
            text: Input text to embed.

        Returns:
            Embedding vector of length 1024.

        Raises:
            RuntimeError: On Bedrock API failure after retries.
        """
        loop = asyncio.get_running_loop()

        def _invoke() -> list[float]:
            body = json.dumps(
                {"inputText": text, "dimensions": _EMBEDDING_DIM, "normalize": True}
            )
            try:
                response = self._client.invoke_model(
                    modelId=_TITAN_MODEL_ID,
                    body=body,
                    contentType="application/json",
                    accept="application/json",
                )
                result: dict[str, Any] = json.loads(response["body"].read())
                return cast(list[float], result["embedding"])
            except (BotoCoreError, ClientError) as exc:
                raise RuntimeError(f"Bedrock invoke_model failed: {exc}") from exc

        return await loop.run_in_executor(None, _invoke)

    async def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 20,
    ) -> list[list[float]]:
        """Embed a list of strings in batches, with rate-limit safety sleep.

        Args:
            texts: Strings to embed.
            batch_size: Number of concurrent Bedrock calls per batch.

        Returns:
            List of embeddings in the same order as *texts*.
        """
        results: list[list[float]] = []
        total_batches = (len(texts) + batch_size - 1) // batch_size

        for batch_num, start in enumerate(range(0, len(texts), batch_size), start=1):
            batch = texts[start : start + batch_size]
            embeddings = await asyncio.gather(*[self.embed(t) for t in batch])
            results.extend(embeddings)
            logger.info("Embedded batch %d/%d", batch_num, total_batches)
            if batch_num < total_batches:
                await asyncio.sleep(0.5)

        return results


# ---------------------------------------------------------------------------
# Pipeline functions
# ---------------------------------------------------------------------------


async def _ensure_embedding_column_dim(conn: Any) -> None:
    """Alter the embedding column to vector(1024) if it is currently vector(1536).

    Safe to call when all rows are NULL (instant rewrite).  Skips the ALTER
    when the column is already the right dimension.

    Args:
        conn: An active asyncpg connection.
    """
    current_dim: int | None = await conn.fetchval(
        """
        SELECT (a.atttypmod - 4)
        FROM   pg_attribute a
        JOIN   pg_class c ON c.oid = a.attrelid
        WHERE  c.relname = 'document_chunks'
          AND  a.attname = 'embedding'
          AND  a.attnum  > 0
        """
    )
    if current_dim == _EMBEDDING_DIM:
        return
    logger.info(
        "Altering embedding column from vector(%s) to vector(%d)",
        current_dim,
        _EMBEDDING_DIM,
    )
    await conn.execute(
        f"ALTER TABLE document_chunks "
        f"ALTER COLUMN embedding TYPE vector({_EMBEDDING_DIM})"
    )


async def embed_all_chunks() -> None:
    """Fetch NULL-embedding rows from document_chunks, embed, and store them.

    Ensures the embedding column matches _EMBEDDING_DIM before writing.
    Uses asyncpg executemany for the batch UPDATE so only one round-trip is
    needed per batch rather than one per row.
    """
    embedder = TitanEmbedder()

    async with get_connection() as conn:
        await _ensure_embedding_column_dim(conn)
        rows = await conn.fetch(
            "SELECT doc_id, content FROM document_chunks WHERE embedding IS NULL"
        )

    if not rows:
        logger.info("No chunks with NULL embeddings found — nothing to do")
        return

    logger.info("Found %d chunks to embed", len(rows))

    doc_ids = [row["doc_id"] for row in rows]
    contents = [row["content"] for row in rows]

    embeddings = await embedder.embed_batch(contents)

    updates = [(emb, doc_id) for emb, doc_id in zip(embeddings, doc_ids)]

    async with get_connection() as conn:
        await conn.executemany(
            "UPDATE document_chunks SET embedding = $1 WHERE doc_id = $2",
            updates,
        )

    logger.info("Updated %d embeddings successfully", len(updates))


async def build_hnsw_index() -> None:
    """Drop and rebuild the cosine HNSW index on document_chunks.embedding.

    m=16 gives good recall/speed balance for the current ~246-chunk corpus.
    ef_construction=64 is the standard starting point for small indexes.
    """
    start = time.perf_counter()

    async with get_connection() as conn:
        await conn.execute("DROP INDEX IF EXISTS idx_chunks_embedding")
        await conn.execute(
            """
            CREATE INDEX idx_chunks_embedding
                ON document_chunks
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64)
            """
        )

    elapsed = time.perf_counter() - start
    logger.info("HNSW index built in %.2fs", elapsed)


# ---------------------------------------------------------------------------
# Vector search
# ---------------------------------------------------------------------------


async def search_similar(
    query: str,
    embedder: TitanEmbedder,
    k: int = 5,
    threshold: float = 0.7,
) -> list[RetrievedDocument]:
    """Embed *query* and return the top-k most similar document chunks.

    Uses cosine similarity via pgvector's <=> operator.
    Only chunks with similarity >= *threshold* are returned.

    Args:
        query: Natural-language query string.
        embedder: TitanEmbedder instance to use for query embedding.
        k: Maximum number of results to return.
        threshold: Minimum cosine similarity (0–1) for a chunk to qualify.

    Returns:
        List of RetrievedDocument sorted by descending similarity.
    """
    query_embedding = await embedder.embed(query)

    async with get_connection() as conn:
        rows = await conn.fetch(
            """
            SELECT
                doc_id,
                content,
                source,
                chunk_index,
                1 - (embedding <=> $1::vector) AS similarity_score,
                metadata
            FROM document_chunks
            WHERE 1 - (embedding <=> $1::vector) >= $2
            ORDER BY embedding <=> $1::vector
            LIMIT $3
            """,
            query_embedding,
            threshold,
            k,
        )

    return [
        RetrievedDocument(
            doc_id=row["doc_id"],
            content=row["content"],
            source=row["source"],
            chunk_index=row["chunk_index"],
            similarity_score=float(row["similarity_score"]),
            metadata=_parse_metadata(row["metadata"]),
        )
        for row in rows
    ]


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

_TEST_QUERIES = [
    "primary frequency response obligations",
    "TNUoS tariff charges 2024",
    "dispatch instruction balancing mechanism",
]


async def _smoke_test() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )
    try:
        await embed_all_chunks()
        await build_hnsw_index()

        embedder = TitanEmbedder()
        for query in _TEST_QUERIES:
            results = await search_similar(query, embedder, k=3, threshold=0.0)
            print(f"\nQuery: {query!r}")
            if not results:
                print("  (no results)")
            for doc in results:
                print(
                    f"  [{doc.similarity_score:.4f}] {doc.source}"
                    f" chunk={doc.chunk_index}"
                )
    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(_smoke_test())

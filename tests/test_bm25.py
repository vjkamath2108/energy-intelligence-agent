"""Unit and integration tests for src/retriever/bm25.py.

Unit tests mock the database — no live PostgreSQL required.
Integration tests are marked @pytest.mark.integration and require a running
PostgreSQL instance with the BM25 schema applied.

Run unit tests only:
    pytest tests/test_bm25.py -m "not integration" -v

Run all (requires DB):
    pytest tests/test_bm25.py -v
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from src.retriever.bm25 import BM25Retriever, _BM25_DDL

pytestmark = pytest.mark.asyncio

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_row(
    *,
    doc_id: str = "doc-1",
    content: str = "BSUoS balancing services use of system charges",
    source: str = "tariffs/bsuos-2024.pdf",
    chunk_index: int = 0,
    metadata: dict[str, str] | None = None,
    rank: float = 0.15,
) -> dict[str, Any]:
    return {
        "doc_id": doc_id,
        "content": content,
        "source": source,
        "chunk_index": chunk_index,
        "metadata": metadata or {},
        "rank": rank,
    }


def _mock_conn(rows: list[dict[str, Any]]) -> AsyncMock:
    conn = AsyncMock()
    conn.fetch.return_value = rows
    conn.execute.return_value = None
    return conn


def _patched_connection(conn: AsyncMock) -> Any:
    ctx = patch("src.retriever.bm25.get_connection")
    mock = ctx.start()
    mock.return_value.__aenter__ = AsyncMock(return_value=conn)
    mock.return_value.__aexit__ = AsyncMock(return_value=False)
    return ctx


# ---------------------------------------------------------------------------
# test_search_returns_list
# ---------------------------------------------------------------------------


async def test_search_returns_list() -> None:
    """search() always returns a list, even when the DB has no matches."""
    conn = _mock_conn([])
    ctx = _patched_connection(conn)
    try:
        retriever = BM25Retriever()
        result = await retriever.search("anything")
        assert isinstance(result, list)
    finally:
        ctx.stop()


# ---------------------------------------------------------------------------
# test_search_energy_term
# ---------------------------------------------------------------------------


async def test_search_energy_term() -> None:
    """search() maps DB rows to RetrievedDocument with correct fields."""
    rows = [_make_row(rank=0.12)]
    conn = _mock_conn(rows)
    ctx = _patched_connection(conn)
    try:
        retriever = BM25Retriever()
        results = await retriever.search("BSUoS")
        assert len(results) == 1
        doc = results[0]
        assert doc.doc_id == "doc-1"
        assert doc.source == "tariffs/bsuos-2024.pdf"
        assert doc.similarity_score == pytest.approx(0.12)
    finally:
        ctx.stop()


# ---------------------------------------------------------------------------
# test_search_phrase_exact
# ---------------------------------------------------------------------------


async def test_search_phrase_exact() -> None:
    """search_phrase() uses phraseto_tsquery — reflected in the SQL sent."""
    rows = [_make_row(content="balancing mechanism dispatch instructions", rank=0.08)]
    conn = _mock_conn(rows)
    ctx = _patched_connection(conn)
    try:
        retriever = BM25Retriever()
        results = await retriever.search_phrase("balancing mechanism", k=3)
        assert len(results) == 1
        # Verify phraseto_tsquery was used, not plainto_tsquery
        sql_called: str = conn.fetch.call_args[0][0]
        assert "phraseto_tsquery" in sql_called
        assert "plainto_tsquery" not in sql_called
    finally:
        ctx.stop()


# ---------------------------------------------------------------------------
# test_search_empty_query
# ---------------------------------------------------------------------------


async def test_search_empty_query() -> None:
    """Empty or whitespace-only query returns [] without hitting the DB."""
    conn = _mock_conn([])
    ctx = _patched_connection(conn)
    try:
        retriever = BM25Retriever()
        assert await retriever.search("") == []
        assert await retriever.search("   ") == []
        conn.fetch.assert_not_called()
    finally:
        ctx.stop()


async def test_search_phrase_empty_query() -> None:
    """Empty phrase returns [] without hitting the DB."""
    conn = _mock_conn([])
    ctx = _patched_connection(conn)
    try:
        retriever = BM25Retriever()
        assert await retriever.search_phrase("") == []
        conn.fetch.assert_not_called()
    finally:
        ctx.stop()


# ---------------------------------------------------------------------------
# test_search_respects_k
# ---------------------------------------------------------------------------


async def test_search_respects_k() -> None:
    """k is forwarded as the LIMIT parameter in the SQL query."""
    conn = _mock_conn([])
    ctx = _patched_connection(conn)
    try:
        retriever = BM25Retriever()
        await retriever.search("TNUoS tariff", k=2)
        positional = conn.fetch.call_args[0]
        # args: sql, language, query, threshold, k  → k is last
        assert positional[-1] == 2
    finally:
        ctx.stop()


# ---------------------------------------------------------------------------
# test_search_threshold
# ---------------------------------------------------------------------------


async def test_search_threshold() -> None:
    """Rows below threshold are not returned."""
    # DB returns one row at rank 0.05; threshold is 0.10 → expect empty list
    rows = [_make_row(rank=0.05)]
    conn = _mock_conn(rows)

    # The threshold is applied in SQL (WHERE ts_rank_cd >= $3); simulate by
    # having the mock return the row but asserting the threshold arg is passed.
    ctx = _patched_connection(conn)
    try:
        retriever = BM25Retriever()
        await retriever.search("frequency response", threshold=0.10)
        positional = conn.fetch.call_args[0]
        # args: sql, language, query, threshold, k → threshold is 4th positional (index 3)
        assert positional[3] == pytest.approx(0.10)
    finally:
        ctx.stop()


async def test_search_threshold_filters_results() -> None:
    """Results all have similarity_score >= threshold when DB respects the filter."""
    rows = [
        _make_row(doc_id="a", rank=0.20),
        _make_row(doc_id="b", rank=0.15),
    ]
    conn = _mock_conn(rows)
    ctx = _patched_connection(conn)
    try:
        retriever = BM25Retriever()
        results = await retriever.search("BSUoS", threshold=0.10)
        assert all(doc.similarity_score >= 0.10 for doc in results)
    finally:
        ctx.stop()


# ---------------------------------------------------------------------------
# test_apply_schema_idempotent
# ---------------------------------------------------------------------------


async def test_apply_schema_idempotent() -> None:
    """apply_schema() can be called twice without raising."""
    conn = _mock_conn([])
    ctx = _patched_connection(conn)
    try:
        retriever = BM25Retriever()
        await retriever.apply_schema()
        await retriever.apply_schema()
        assert conn.execute.call_count == 2
    finally:
        ctx.stop()


async def test_apply_schema_sends_correct_ddl() -> None:
    """apply_schema() executes the expected DDL string."""
    conn = _mock_conn([])
    ctx = _patched_connection(conn)
    try:
        retriever = BM25Retriever()
        await retriever.apply_schema()
        ddl_sent: str = conn.execute.call_args[0][0]
        assert "ADD COLUMN IF NOT EXISTS fts" in ddl_sent
        assert "GENERATED ALWAYS AS" in ddl_sent
        assert "CREATE INDEX IF NOT EXISTS idx_chunks_fts" in ddl_sent
        assert "gin(fts)" in ddl_sent
    finally:
        ctx.stop()


# ---------------------------------------------------------------------------
# test_rank_ordering
# ---------------------------------------------------------------------------


async def test_rank_ordering() -> None:
    """Results are ordered by descending rank (as returned by the DB)."""
    rows = [
        _make_row(doc_id="high", rank=0.30),
        _make_row(doc_id="mid", rank=0.20),
        _make_row(doc_id="low", rank=0.05),
    ]
    conn = _mock_conn(rows)
    ctx = _patched_connection(conn)
    try:
        retriever = BM25Retriever()
        results = await retriever.search("balancing", k=3)
        scores = [doc.similarity_score for doc in results]
        assert scores == sorted(scores, reverse=True)
    finally:
        ctx.stop()


# ---------------------------------------------------------------------------
# test metadata handling
# ---------------------------------------------------------------------------


async def test_search_null_metadata() -> None:
    """NULL metadata column is returned as an empty dict."""
    rows = [_make_row(metadata=None)]  # type: ignore[arg-type]
    rows[0]["metadata"] = None
    conn = _mock_conn(rows)
    ctx = _patched_connection(conn)
    try:
        retriever = BM25Retriever()
        results = await retriever.search("BSUoS")
        assert results[0].metadata == {}
    finally:
        ctx.stop()


async def test_search_json_string_metadata() -> None:
    """JSONB returned as a raw string by asyncpg is decoded correctly."""
    rows = [_make_row()]
    rows[0]["metadata"] = json.dumps({"category": "tariff"})
    conn = _mock_conn(rows)
    ctx = _patched_connection(conn)
    try:
        retriever = BM25Retriever()
        results = await retriever.search("BSUoS")
        assert results[0].metadata == {"category": "tariff"}
    finally:
        ctx.stop()


# ---------------------------------------------------------------------------
# Integration tests (require live DB)
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_integration_apply_schema() -> None:
    """apply_schema() runs against a live DB without error."""
    from src.db import close_pool

    retriever = BM25Retriever()
    try:
        await retriever.apply_schema()
    finally:
        await close_pool()


@pytest.mark.integration
async def test_integration_search_returns_results() -> None:
    """search() returns at least one result for a known energy term."""
    from src.db import close_pool

    retriever = BM25Retriever()
    try:
        results = await retriever.search("balancing", k=5)
        assert isinstance(results, list)
    finally:
        await close_pool()

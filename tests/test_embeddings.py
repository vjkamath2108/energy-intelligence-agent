"""Unit tests for src/retriever/embeddings.py.

All Bedrock and database calls are mocked — no live AWS or PostgreSQL
connection is required to run this suite.

Run with:
    pytest tests/test_embeddings.py -v
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.retriever.embeddings import (
    TitanEmbedder,
    _EMBEDDING_DIM,
    _parse_metadata,
    build_hnsw_index,
    embed_all_chunks,
    search_similar,
)

pytestmark = pytest.mark.asyncio

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_EMBEDDING = [0.1] * _EMBEDDING_DIM  # 1024 floats


def _make_bedrock_response(embedding: list[float]) -> dict[str, Any]:
    """Build a fake boto3 invoke_model response."""

    class _Body:
        def read(self) -> bytes:
            return json.dumps({"embedding": embedding}).encode()

    return {"body": _Body()}


# ---------------------------------------------------------------------------
# TitanEmbedder.embed
# ---------------------------------------------------------------------------


async def test_embed_returns_correct_length() -> None:
    """embed() returns a list of exactly 1024 floats."""
    embedder = TitanEmbedder()
    embedder._client = MagicMock()
    embedder._client.invoke_model.return_value = _make_bedrock_response(_FAKE_EMBEDDING)

    result = await embedder.embed("test input")

    assert isinstance(result, list)
    assert len(result) == _EMBEDDING_DIM
    assert all(isinstance(v, float) for v in result)


async def test_embed_passes_correct_payload() -> None:
    """embed() calls invoke_model with the expected body fields."""
    embedder = TitanEmbedder()
    embedder._client = MagicMock()
    embedder._client.invoke_model.return_value = _make_bedrock_response(_FAKE_EMBEDDING)

    await embedder.embed("energy regulation text")

    call_kwargs = embedder._client.invoke_model.call_args.kwargs
    body = json.loads(call_kwargs["body"])

    assert body["inputText"] == "energy regulation text"
    assert body["dimensions"] == _EMBEDDING_DIM
    assert body["normalize"] is True


async def test_embed_raises_on_bedrock_error() -> None:
    """embed() wraps boto3 ClientError in RuntimeError."""
    from botocore.exceptions import ClientError

    embedder = TitanEmbedder()
    embedder._client = MagicMock()
    embedder._client.invoke_model.side_effect = ClientError(
        {"Error": {"Code": "ThrottlingException", "Message": "rate exceeded"}},
        "InvokeModel",
    )

    with pytest.raises(RuntimeError, match="Bedrock invoke_model failed"):
        await embedder.embed("some text")


# ---------------------------------------------------------------------------
# TitanEmbedder.embed_batch
# ---------------------------------------------------------------------------


async def test_embed_batch_preserves_order() -> None:
    """embed_batch() returns embeddings in the same order as the input texts."""
    embedder = TitanEmbedder()
    texts = [f"doc {i}" for i in range(5)]
    # Each call returns a unique embedding identified by the index
    side_effects = [[float(i)] * _EMBEDDING_DIM for i in range(5)]

    call_count = 0

    async def fake_embed(text: str) -> list[float]:
        nonlocal call_count
        idx = int(text.split()[-1])
        call_count += 1
        return side_effects[idx]

    embedder.embed = fake_embed  # type: ignore[method-assign]

    results = await embedder.embed_batch(texts, batch_size=3)

    assert len(results) == 5
    for i, emb in enumerate(results):
        assert emb[0] == float(i), f"Order mismatch at index {i}"


async def test_embed_batch_respects_batch_size(monkeypatch: pytest.MonkeyPatch) -> None:
    """embed_batch() sleeps between batches but not after the last one."""
    embedder = TitanEmbedder()

    async def fake_embed(text: str) -> list[float]:
        return _FAKE_EMBEDDING

    embedder.embed = fake_embed  # type: ignore[method-assign]

    sleep_calls: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    monkeypatch.setattr("src.retriever.embeddings.asyncio.sleep", fake_sleep)

    texts = ["a", "b", "c", "d", "e"]
    await embedder.embed_batch(texts, batch_size=2)

    # 5 texts / batch_size 2 = 3 batches → 2 sleeps (not after the last batch)
    assert len(sleep_calls) == 2
    assert all(d == 0.5 for d in sleep_calls)


# ---------------------------------------------------------------------------
# embed_all_chunks
# ---------------------------------------------------------------------------


async def test_embed_all_chunks_skips_when_none_null() -> None:
    """embed_all_chunks() exits early and logs when no NULL rows are found."""
    mock_conn = AsyncMock()
    mock_conn.fetch.return_value = []

    with patch("src.retriever.embeddings.get_connection") as mock_ctx:
        mock_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_ctx.return_value.__aexit__ = AsyncMock(return_value=False)
        await embed_all_chunks()

    mock_conn.executemany.assert_not_called()


async def test_embed_all_chunks_updates_rows() -> None:
    """embed_all_chunks() calls executemany once with all (embedding, doc_id) pairs."""
    rows = [
        {"doc_id": "doc-1", "content": "content one"},
        {"doc_id": "doc-2", "content": "content two"},
    ]

    mock_conn = AsyncMock()
    mock_conn.fetch.return_value = rows
    mock_conn.executemany.return_value = None

    fake_embeddings = [_FAKE_EMBEDDING, _FAKE_EMBEDDING]

    with (
        patch("src.retriever.embeddings.get_connection") as mock_ctx,
        patch.object(TitanEmbedder, "embed_batch", new=AsyncMock(return_value=fake_embeddings)),
    ):
        mock_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_ctx.return_value.__aexit__ = AsyncMock(return_value=False)
        await embed_all_chunks()

    mock_conn.executemany.assert_called_once()
    call_args = mock_conn.executemany.call_args
    sql: str = call_args[0][0]
    updates: list[tuple[Any, str]] = call_args[0][1]

    assert "UPDATE document_chunks" in sql
    assert len(updates) == 2
    doc_ids_in_update = {u[1] for u in updates}
    assert doc_ids_in_update == {"doc-1", "doc-2"}


# ---------------------------------------------------------------------------
# _parse_metadata
# ---------------------------------------------------------------------------


def test_parse_metadata_none_returns_empty_dict() -> None:
    """None metadata maps to an empty dict."""
    assert _parse_metadata(None) == {}


def test_parse_metadata_dict_passthrough() -> None:
    """A dict value is returned as-is (asyncpg decoded it already)."""
    assert _parse_metadata({"category": "regulation"}) == {"category": "regulation"}


def test_parse_metadata_json_string_decoded() -> None:
    """A raw JSON string (as asyncpg may return for JSONB) is decoded."""
    raw = json.dumps({"category": "tariff", "year": "2024"})
    assert _parse_metadata(raw) == {"category": "tariff", "year": "2024"}


def test_parse_metadata_empty_string_returns_empty_dict() -> None:
    """An empty string is treated as missing metadata."""
    assert _parse_metadata("") == {}


# ---------------------------------------------------------------------------
# build_hnsw_index
# ---------------------------------------------------------------------------


async def test_build_hnsw_index_executes_correct_statements() -> None:
    """build_hnsw_index() drops the old index then creates a new HNSW one."""
    executed: list[str] = []
    mock_conn = AsyncMock()
    mock_conn.execute.side_effect = lambda sql: executed.append(sql.strip()) or None  # type: ignore[return-value]

    with patch("src.retriever.embeddings.get_connection") as mock_ctx:
        mock_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_ctx.return_value.__aexit__ = AsyncMock(return_value=False)
        await build_hnsw_index()

    assert any("DROP INDEX IF EXISTS idx_chunks_embedding" in s for s in executed)
    assert any("hnsw" in s.lower() for s in executed)
    assert any("vector_cosine_ops" in s for s in executed)


# ---------------------------------------------------------------------------
# search_similar
# ---------------------------------------------------------------------------


async def test_search_similar_returns_retrieved_documents() -> None:
    """search_similar() maps database rows to RetrievedDocument instances."""
    db_rows = [
        {
            "doc_id": "chunk-001",
            "content": "Frequency response obligations for TSOs",
            "source": "regulations/grid-code.pdf",
            "chunk_index": 3,
            "similarity_score": 0.92,
            "metadata": {"category": "regulation"},
        }
    ]

    mock_conn = AsyncMock()
    mock_conn.fetch.return_value = db_rows

    embedder = TitanEmbedder()

    with (
        patch("src.retriever.embeddings.get_connection") as mock_ctx,
        patch.object(TitanEmbedder, "embed", new=AsyncMock(return_value=_FAKE_EMBEDDING)),
    ):
        mock_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_ctx.return_value.__aexit__ = AsyncMock(return_value=False)
        results = await search_similar("frequency response", embedder, k=5, threshold=0.7)

    assert len(results) == 1
    doc = results[0]
    assert doc.doc_id == "chunk-001"
    assert doc.similarity_score == pytest.approx(0.92)
    assert doc.source == "regulations/grid-code.pdf"
    assert doc.chunk_index == 3
    assert doc.metadata == {"category": "regulation"}


async def test_search_similar_passes_threshold_and_limit() -> None:
    """search_similar() forwards threshold and k to the SQL query."""
    mock_conn = AsyncMock()
    mock_conn.fetch.return_value = []

    embedder = TitanEmbedder()

    with (
        patch("src.retriever.embeddings.get_connection") as mock_ctx,
        patch.object(TitanEmbedder, "embed", new=AsyncMock(return_value=_FAKE_EMBEDDING)),
    ):
        mock_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_ctx.return_value.__aexit__ = AsyncMock(return_value=False)
        await search_similar("query", embedder, k=3, threshold=0.85)

    call_args = mock_conn.fetch.call_args
    positional_args = call_args[0]
    # SQL is first arg; query_embedding=$1, threshold=$2, k=$3
    assert positional_args[2] == 0.85
    assert positional_args[3] == 3


async def test_search_similar_empty_metadata() -> None:
    """search_similar() handles NULL metadata without raising."""
    db_rows = [
        {
            "doc_id": "chunk-002",
            "content": "balancing mechanism instructions",
            "source": "decisions/dispatch-log.pdf",
            "chunk_index": 0,
            "similarity_score": 0.75,
            "metadata": None,
        }
    ]

    mock_conn = AsyncMock()
    mock_conn.fetch.return_value = db_rows

    embedder = TitanEmbedder()

    with (
        patch("src.retriever.embeddings.get_connection") as mock_ctx,
        patch.object(TitanEmbedder, "embed", new=AsyncMock(return_value=_FAKE_EMBEDDING)),
    ):
        mock_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_ctx.return_value.__aexit__ = AsyncMock(return_value=False)
        results = await search_similar("dispatch", embedder)

    assert results[0].metadata == {}

"""Tests for src/retriever/hybrid_retriever.py.

Unit tests use no live DB or Bedrock — all external calls are mocked.
Integration tests are marked @pytest.mark.integration.

Run unit tests only:
    pytest tests/test_hybrid_retriever.py -m "not integration" -v

Run all (requires DB + Bedrock):
    pytest tests/test_hybrid_retriever.py -v
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.models import RetrievedDocument
from src.retriever.hybrid_retriever import HybridRetriever, _rrf_fuse

# asyncio_mode = auto in pytest.ini — no pytestmark needed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _doc(
    doc_id: str,
    source: str = "regulations/test.pdf",
    content: str | None = None,
    score: float = 0.5,
) -> RetrievedDocument:
    return RetrievedDocument(
        doc_id=doc_id,
        content=content if content is not None else f"content-{doc_id}",
        source=source,
        chunk_index=0,
        similarity_score=score,
        metadata={},
    )


def _mock_bm25(docs: list[RetrievedDocument]) -> MagicMock:
    m = MagicMock()
    m.search = AsyncMock(return_value=docs)
    return m


# ---------------------------------------------------------------------------
# _rrf_fuse — pure function, no I/O
# ---------------------------------------------------------------------------


def test_rrf_fuse_empty_inputs() -> None:
    assert _rrf_fuse([], [], k=5) == []


def test_rrf_fuse_bm25_only_rank1_score() -> None:
    """Rank-1 BM25-only doc scores exactly 1/(60+1)."""
    result = _rrf_fuse([_doc("a"), _doc("b")], [], k=5)
    assert result[0].doc_id == "a"
    assert result[0].similarity_score == pytest.approx(1 / 61, rel=1e-4)


def test_rrf_fuse_semantic_only_rank1_score() -> None:
    """Rank-1 semantic-only doc scores exactly 1/(60+1)."""
    result = _rrf_fuse([], [_doc("x"), _doc("y")], k=5)
    assert result[0].doc_id == "x"
    assert result[0].similarity_score == pytest.approx(1 / 61, rel=1e-4)


def test_rrf_fuse_rank2_score() -> None:
    """Rank-2 doc (BM25 only) scores 1/(60+2) = 1/62."""
    result = _rrf_fuse([_doc("a"), _doc("b")], [], k=5)
    assert result[1].similarity_score == pytest.approx(1 / 62, rel=1e-4)


def test_rrf_fuse_cross_retriever_accumulation() -> None:
    """Doc appearing rank-1 in both lists scores 2/(60+1) — higher than either alone."""
    shared = _doc("shared")
    bm25_docs = [shared, _doc("bm_only")]
    sem_docs = [shared, _doc("sem_only")]
    result = _rrf_fuse(bm25_docs, sem_docs, k=5)

    by_id = {r.doc_id: r.similarity_score for r in result}
    assert by_id["shared"] == pytest.approx(2 / 61, rel=1e-4)
    assert by_id["shared"] > by_id["bm_only"]
    assert by_id["shared"] > by_id["sem_only"]


def test_rrf_fuse_dedup_single_result_per_doc_id() -> None:
    """Same doc_id in both lists produces exactly one result."""
    shared = _doc("dup")
    result = _rrf_fuse([shared], [shared], k=5)
    assert len(result) == 1
    assert result[0].doc_id == "dup"


def test_rrf_fuse_longer_content_wins_on_dedup() -> None:
    """When same doc_id appears in both lists, the version with longer content is kept."""
    short = _doc("dup", content="short")
    long = _doc("dup", content="much longer content string for this test")
    result = _rrf_fuse([short], [long], k=5)
    assert result[0].content == long.content


def test_rrf_fuse_k_limits_output() -> None:
    """Output length is capped at k even when more candidates exist."""
    docs = [_doc(f"d{i}") for i in range(10)]
    assert len(_rrf_fuse(docs, [], k=3)) == 3


def test_rrf_fuse_ordering_descending() -> None:
    """Results are sorted by descending RRF score."""
    bm25_docs = [_doc("a"), _doc("b"), _doc("c")]
    sem_docs = [_doc("c"), _doc("b")]  # c and b get bonus from appearing in both
    result = _rrf_fuse(bm25_docs, sem_docs, k=5)
    scores = [r.similarity_score for r in result]
    assert scores == sorted(scores, reverse=True)


def test_rrf_fuse_single_doc_each_retriever() -> None:
    """Two unique docs — no accumulation — each scored from one retriever."""
    result = _rrf_fuse([_doc("bm")], [_doc("sem")], k=5)
    # Both are rank-1 in their respective list → same score
    assert len(result) == 2
    assert result[0].similarity_score == pytest.approx(result[1].similarity_score, rel=1e-4)


def test_rrf_fuse_k_zero_returns_empty() -> None:
    assert _rrf_fuse([_doc("a")], [_doc("b")], k=0) == []


# ---------------------------------------------------------------------------
# HybridRetriever.search — mocked unit tests
# ---------------------------------------------------------------------------


async def test_search_empty_query_returns_empty() -> None:
    retriever = HybridRetriever(bm25=_mock_bm25([]), embedder=MagicMock())
    assert await retriever.search("") == []
    assert await retriever.search("   ") == []


async def test_search_calls_both_retrievers() -> None:
    """Both BM25 and semantic search are called for a non-empty query."""
    bm25_docs = [_doc("b1", "tariffs/bsuos.pdf")]
    sem_docs = [_doc("s1", "regulations/grid-code.pdf")]

    mock_bm25 = _mock_bm25(bm25_docs)
    with patch(
        "src.retriever.hybrid_retriever.search_similar",
        new=AsyncMock(return_value=sem_docs),
    ):
        retriever = HybridRetriever(bm25=mock_bm25, embedder=MagicMock())
        results = await retriever.search("BSUoS charges", k=5)

    mock_bm25.search.assert_awaited_once()
    # Both unique docs appear in fused output
    assert len(results) == 2


async def test_search_fetch_k_uses_multiplier() -> None:
    """Each sub-retriever is called with k * fetch_k_multiplier, not k."""
    mock_bm25 = _mock_bm25([])
    with patch(
        "src.retriever.hybrid_retriever.search_similar",
        new=AsyncMock(return_value=[]),
    ):
        retriever = HybridRetriever(bm25=mock_bm25, embedder=MagicMock(), fetch_k_multiplier=3)
        await retriever.search("frequency response", k=5)

    mock_bm25.search.assert_awaited_once_with("frequency response", k=15)


async def test_search_passes_threshold_zero_to_semantic() -> None:
    """Hybrid passes threshold=0.0 to search_similar — BM25 handles topic filtering."""
    mock_bm25 = _mock_bm25([])
    with patch(
        "src.retriever.hybrid_retriever.search_similar",
        new=AsyncMock(return_value=[]),
    ) as mock_sem:
        retriever = HybridRetriever(bm25=mock_bm25, embedder=MagicMock())
        await retriever.search("TNUoS tariff", k=5)

    assert mock_sem.call_args.kwargs.get("threshold") == pytest.approx(0.0)


async def test_search_output_limited_to_k() -> None:
    """Fused output is capped at k even when sub-retrievers return more."""
    bm25_docs = [_doc(f"b{i}") for i in range(10)]
    sem_docs = [_doc(f"s{i}") for i in range(10)]
    mock_bm25 = _mock_bm25(bm25_docs)
    with patch(
        "src.retriever.hybrid_retriever.search_similar",
        new=AsyncMock(return_value=sem_docs),
    ):
        retriever = HybridRetriever(bm25=mock_bm25, embedder=MagicMock())
        results = await retriever.search("dispatch", k=5)

    assert len(results) <= 5


async def test_search_rrf_scores_replace_raw_scores() -> None:
    """similarity_score on returned docs is the RRF score, not the raw BM25/cosine score."""
    bm25_docs = [_doc("a", score=0.99)]  # high raw BM25 score
    sem_docs = []
    mock_bm25 = _mock_bm25(bm25_docs)
    with patch(
        "src.retriever.hybrid_retriever.search_similar",
        new=AsyncMock(return_value=sem_docs),
    ):
        retriever = HybridRetriever(bm25=mock_bm25, embedder=MagicMock())
        results = await retriever.search("tariff", k=5)

    # RRF score for rank-1 from one retriever = 1/61 ≈ 0.016 — not 0.99
    assert results[0].similarity_score == pytest.approx(1 / 61, rel=1e-4)


async def test_apply_schema_delegates_to_bm25() -> None:
    """apply_schema() calls BM25Retriever.apply_schema() exactly once."""
    mock_bm25 = _mock_bm25([])
    mock_bm25.apply_schema = AsyncMock()
    retriever = HybridRetriever(bm25=mock_bm25, embedder=MagicMock())
    await retriever.apply_schema()
    mock_bm25.apply_schema.assert_awaited_once()


# ---------------------------------------------------------------------------
# Integration tests (require live DB + Bedrock)
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_integration_hybrid_search_returns_results() -> None:
    """Hybrid search returns a list for a known energy domain query."""
    from src.db import close_pool
    from src.retriever.embeddings import TitanEmbedder

    retriever = HybridRetriever(embedder=TitanEmbedder())
    try:
        results = await retriever.search("BSUoS balancing charges", k=5)
        assert isinstance(results, list)
        assert all(0.0 <= r.similarity_score <= 1.0 for r in results)
    finally:
        await close_pool()


@pytest.mark.integration
async def test_integration_apply_schema() -> None:
    """apply_schema() runs without error against a live DB."""
    from src.db import close_pool
    from src.retriever.embeddings import TitanEmbedder

    retriever = HybridRetriever(embedder=TitanEmbedder())
    try:
        await retriever.apply_schema()
    finally:
        await close_pool()

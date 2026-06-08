"""Tests for src/recommendation_engine/safety.py — hallucination detection.

Tests the term overlap, semantic similarity, and composite grounding checks.
All tests are unit tests — no external API calls.

Run:
    pytest tests/test_explanation.py -v
"""

from __future__ import annotations

import math
import pytest
from unittest.mock import AsyncMock, MagicMock

from src.models import RetrievedDocument
from src.recommendation_engine.safety import (
    SEMANTIC_THRESHOLD,
    TERM_OVERLAP_THRESHOLD,
    SafetyResult,
    _cosine,
    _tokenise,
    check_grounding,
    term_overlap,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _doc(doc_id: str, content: str = "dispatch instruction upward balancing constraint") -> RetrievedDocument:
    return RetrievedDocument(
        doc_id=doc_id,
        content=content,
        source="decisions/test.pdf",
        chunk_index=0,
        similarity_score=0.8,
        metadata={},
    )


class _FixedEmbedder:
    """Returns the same vector for every embed call."""

    def __init__(self, vector: list[float]) -> None:
        self._vector = vector

    async def embed(self, text: str) -> list[float]:
        return self._vector


class _SequentialEmbedder:
    """Returns vectors in sequence — first call → vectors[0], second → vectors[1]."""

    def __init__(self, *vectors: list[float]) -> None:
        self._vectors = list(vectors)
        self._idx = 0

    async def embed(self, text: str) -> list[float]:
        vec = self._vectors[self._idx % len(self._vectors)]
        self._idx += 1
        return vec


# ---------------------------------------------------------------------------
# _tokenise
# ---------------------------------------------------------------------------


def test_tokenise_removes_stopwords() -> None:
    tokens = _tokenise("the grid is experiencing frequency deviation")
    assert "the" not in tokens
    assert "is" not in tokens
    assert "grid" in tokens
    assert "frequency" in tokens
    assert "deviation" in tokens


def test_tokenise_requires_minimum_length() -> None:
    """Tokens shorter than 3 chars are excluded."""
    tokens = _tokenise("BSC dispatch MW Hz")
    assert "bsc" in tokens
    # "mw" is 2 chars — excluded
    assert "mw" not in tokens
    # "hz" is 2 chars — excluded
    assert "hz" not in tokens


def test_tokenise_lowercases_input() -> None:
    assert "drax1g" in _tokenise("DRAX1G power station")


# ---------------------------------------------------------------------------
# term_overlap
# ---------------------------------------------------------------------------


def test_term_overlap_full_match_returns_one() -> None:
    """All claim tokens present in corpus → 1.0."""
    score = term_overlap("dispatch constraint north region", "dispatch constraint north region penalty")
    assert score == pytest.approx(1.0)


def test_term_overlap_no_match_returns_zero() -> None:
    """No claim tokens in corpus → 0.0."""
    score = term_overlap("dispatch constraint", "unrelated completely different text here")
    assert score == pytest.approx(0.0)


def test_term_overlap_partial_match() -> None:
    """Exactly half of meaningful tokens matched → 0.5."""
    # claim meaningful tokens: {dispatch, constraint} (2 tokens)
    # corpus contains: dispatch but not constraint
    score = term_overlap("dispatch constraint", "dispatch unrelated")
    assert score == pytest.approx(0.5)


def test_term_overlap_empty_claim_returns_one() -> None:
    """Claim with no meaningful tokens is vacuously grounded."""
    score = term_overlap("the a an is in", "some corpus content")
    assert score == pytest.approx(1.0)


def test_term_overlap_threshold_boundary() -> None:
    """Verify TERM_OVERLAP_THRESHOLD = 0.30 matches documented gate."""
    assert TERM_OVERLAP_THRESHOLD == pytest.approx(0.30)


# ---------------------------------------------------------------------------
# _cosine
# ---------------------------------------------------------------------------


def test_cosine_identical_unit_vectors() -> None:
    a = [1.0, 0.0, 0.0]
    assert _cosine(a, a) == pytest.approx(1.0)


def test_cosine_orthogonal_vectors() -> None:
    a = [1.0, 0.0, 0.0]
    b = [0.0, 1.0, 0.0]
    assert _cosine(a, b) == pytest.approx(0.0)


def test_cosine_antiparallel_clipped_to_zero() -> None:
    """Cosine is clipped to 0 — negative values make no sense for similarity."""
    a = [1.0, 0.0]
    b = [-1.0, 0.0]
    result = _cosine(a, b)
    assert result == pytest.approx(0.0)


def test_cosine_zero_vector_returns_zero() -> None:
    assert _cosine([0.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)
    assert _cosine([1.0, 0.0], [0.0, 0.0]) == pytest.approx(0.0)


def test_cosine_known_value() -> None:
    """cos(45°) between [1,1,0] and [1,0,0] = 1/√2 ≈ 0.707."""
    a = [1.0, 1.0, 0.0]
    b = [1.0, 0.0, 0.0]
    expected = 1.0 / math.sqrt(2)
    assert _cosine(a, b) == pytest.approx(expected, rel=1e-4)


# ---------------------------------------------------------------------------
# check_grounding — term overlap only (no embedder)
# ---------------------------------------------------------------------------


async def test_check_grounding_no_docs_returns_not_grounded() -> None:
    """No retrieved documents → always not grounded."""
    result = await check_grounding("dispatch DRAX1G 200 MW", "constraint confirmed", [])
    assert result.is_grounded is False
    assert "no retrieved documents" in result.flagged_reasons[0]


async def test_check_grounding_high_overlap_is_grounded() -> None:
    """Action terms all present in doc content → grounded."""
    docs = [_doc("d1", content="dispatch instruction upward balancing constraint north region")]
    result = await check_grounding(
        "dispatch upward constraint north region",
        "balancing instruction confirmed",
        docs,
    )
    assert result.is_grounded is True
    assert result.term_overlap_ratio >= TERM_OVERLAP_THRESHOLD
    assert result.semantic_similarity is None


async def test_check_grounding_low_overlap_not_grounded() -> None:
    """Action terms absent from docs → not grounded, term overlap check fails."""
    docs = [_doc("d1", content="completely unrelated text about something else entirely")]
    result = await check_grounding(
        "dispatch DRAX1G constraint penalty settlement",
        "BSUoS charges confirmed",
        docs,
    )
    assert result.is_grounded is False
    assert result.term_overlap_ratio < TERM_OVERLAP_THRESHOLD
    assert any("term overlap" in r for r in result.flagged_reasons)


async def test_check_grounding_flags_reasons_tuple() -> None:
    """flagged_reasons is a tuple of strings describing failed checks."""
    docs = [_doc("d1", content="unrelated")]
    result = await check_grounding("completely different words here action", "", docs)
    assert isinstance(result.flagged_reasons, tuple)
    assert len(result.flagged_reasons) >= 1


# ---------------------------------------------------------------------------
# check_grounding — with mock embedder (semantic similarity)
# ---------------------------------------------------------------------------


async def test_check_grounding_with_identical_embedder_passes() -> None:
    """Same embedding for claim and corpus → cosine=1.0 → semantic check passes."""
    docs = [_doc("d1", content="unrelated content")]  # low term overlap intentionally
    embedder = _FixedEmbedder([1.0, 0.0, 0.0])

    result = await check_grounding(
        "dispatch constraint",
        "reasoning",
        docs,
        embedder=embedder,
    )
    assert result.semantic_similarity == pytest.approx(1.0)
    # Term overlap may fail (low overlap with "unrelated content"), but
    # we verify the semantic_similarity field is set correctly
    assert result.semantic_similarity is not None


async def test_check_grounding_with_orthogonal_embedder_flags_semantic() -> None:
    """Orthogonal embeddings → cosine=0.0 → semantic similarity check fails."""
    docs = [_doc("d1", content="dispatch constraint north region balancing")]
    # High term overlap but orthogonal embeddings
    embedder = _SequentialEmbedder([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])

    result = await check_grounding(
        "dispatch constraint north region",
        "balancing confirmed",
        docs,
        embedder=embedder,
    )
    assert result.semantic_similarity == pytest.approx(0.0)
    assert any("semantic similarity" in r for r in result.flagged_reasons)
    assert result.is_grounded is False


async def test_check_grounding_embedder_called_twice() -> None:
    """Embedder is called once for the claim and once for the corpus."""
    docs = [_doc("d1")]
    mock_embed = AsyncMock(return_value=[1.0, 0.0, 0.0])
    embedder = MagicMock()
    embedder.embed = mock_embed

    await check_grounding("dispatch constraint", "reasoning", docs, embedder=embedder)
    assert mock_embed.call_count == 2


# ---------------------------------------------------------------------------
# SafetyResult — frozen immutability
# ---------------------------------------------------------------------------


def test_safety_result_is_frozen() -> None:
    """SafetyResult is frozen — attribute reassignment raises AttributeError."""
    result = SafetyResult(
        is_grounded=True,
        term_overlap_ratio=0.8,
        semantic_similarity=None,
        flagged_reasons=(),
    )
    with pytest.raises((AttributeError, TypeError)):
        result.is_grounded = False  # type: ignore[misc]


def test_safety_result_flagged_reasons_is_tuple() -> None:
    result = SafetyResult(
        is_grounded=False,
        term_overlap_ratio=0.2,
        semantic_similarity=0.3,
        flagged_reasons=("term overlap 0.20 < threshold 0.30",),
    )
    assert isinstance(result.flagged_reasons, tuple)
    assert "term overlap" in result.flagged_reasons[0]

"""Tests for ClaudeRecommender and OllamaRecommender.

All tests are unit tests — no live AWS or Ollama calls.
Bedrock calls are mocked via unittest.mock.
Ollama calls are mocked via patch.object on _call_ollama.

Run:
    pytest tests/test_recommendation_engine.py -v
"""

from __future__ import annotations

import io
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from botocore.exceptions import ClientError  # type: ignore[import-untyped]

from src.models import DispatchRecommendation, RetrievedDocument
from src.recommendation_engine.safety import SafetyResult

# Import submodules so patch() can resolve "src.recommendation_engine.claude_client.*"
import src.recommendation_engine.claude_client  # noqa: F401
import src.recommendation_engine.ollama_client  # noqa: F401
from src.recommendation_engine.claude_client import ClaudeRecommender
from src.recommendation_engine.ollama_client import OllamaRecommender


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _doc(doc_id: str, source: str = "decisions/dispatch-north.pdf") -> RetrievedDocument:
    return RetrievedDocument(
        doc_id=doc_id,
        content=f"dispatch instruction upward balancing constraint {doc_id}",
        source=source,
        chunk_index=0,
        similarity_score=0.8,
        metadata={},
    )


def _bedrock_response(text: str) -> dict:
    """Wrap text in the Bedrock invoke_model response structure."""
    body = json.dumps({"content": [{"text": text}]}).encode()
    return {"body": io.BytesIO(body)}


def _valid_llm_json(doc_id: str = "doc-1", confidence: float = 0.85) -> str:
    return json.dumps({
        "action": "Issue upward dispatch instruction to DRAX1G for 200 MW",
        "confidence": confidence,
        "reasoning": "North region constraint confirmed by retrieved dispatch document",
        "cited_doc_ids": [doc_id],
    })


def _grounded() -> SafetyResult:
    return SafetyResult(
        is_grounded=True,
        term_overlap_ratio=0.8,
        semantic_similarity=None,
        flagged_reasons=(),
    )


def _ungrounded() -> SafetyResult:
    return SafetyResult(
        is_grounded=False,
        term_overlap_ratio=0.1,
        semantic_similarity=None,
        flagged_reasons=("term overlap 0.10 < threshold 0.30",),
    )


# ---------------------------------------------------------------------------
# ClaudeRecommender — happy path
# ---------------------------------------------------------------------------


async def test_claude_client_returns_dispatch_recommendation() -> None:
    """Happy path: valid Bedrock response produces a DispatchRecommendation."""
    context_docs = [_doc("doc-1")]

    with patch("src.recommendation_engine.claude_client.boto3") as mock_boto3:
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.invoke_model.return_value = _bedrock_response(_valid_llm_json("doc-1"))

        with patch(
            "src.recommendation_engine.claude_client.check_grounding",
            new=AsyncMock(return_value=_grounded()),
        ):
            recommender = ClaudeRecommender()
            result = await recommender.recommend("North constraint", context_docs)

    assert isinstance(result, DispatchRecommendation)
    assert result.action == "Issue upward dispatch instruction to DRAX1G for 200 MW"
    assert result.confidence == pytest.approx(0.83)
    assert result.query == "North constraint"
    assert result.requires_human_review is False
    assert result.hallucination_flagged is False


async def test_claude_client_supporting_docs_from_cited_ids() -> None:
    """supporting_documents contains the cited doc, not all context docs."""
    context_docs = [_doc("cited"), _doc("uncited")]

    with patch("src.recommendation_engine.claude_client.boto3") as mock_boto3:
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.invoke_model.return_value = _bedrock_response(_valid_llm_json("cited"))

        with patch(
            "src.recommendation_engine.claude_client.check_grounding",
            new=AsyncMock(return_value=_grounded()),
        ):
            recommender = ClaudeRecommender()
            result = await recommender.recommend("query", context_docs)

    assert len(result.supporting_documents) == 1
    assert result.supporting_documents[0].doc_id == "cited"


# ---------------------------------------------------------------------------
# ClaudeRecommender — JSON retry
# ---------------------------------------------------------------------------


async def test_claude_client_retries_on_invalid_json() -> None:
    """First call returns garbage; second call returns valid JSON — result is valid."""
    context_docs = [_doc("doc-1")]
    garbage_response = _bedrock_response("I cannot provide a JSON response right now.")
    valid_response = _bedrock_response(_valid_llm_json("doc-1"))

    with patch("src.recommendation_engine.claude_client.boto3") as mock_boto3:
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.invoke_model.side_effect = [
            {"body": garbage_response["body"]},
            {"body": valid_response["body"]},
        ]

        with patch(
            "src.recommendation_engine.claude_client.check_grounding",
            new=AsyncMock(return_value=_grounded()),
        ):
            recommender = ClaudeRecommender()
            result = await recommender.recommend("query", context_docs)

    assert isinstance(result, DispatchRecommendation)
    assert mock_client.invoke_model.call_count == 2


# ---------------------------------------------------------------------------
# ClaudeRecommender — throttling retry
# ---------------------------------------------------------------------------


async def test_claude_client_retries_on_throttling() -> None:
    """ThrottlingException on first attempt, success on second."""
    context_docs = [_doc("doc-1")]

    throttle_error = ClientError(
        {"Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}},
        "InvokeModel",
    )

    with patch("src.recommendation_engine.claude_client.boto3") as mock_boto3:
        with patch("src.recommendation_engine.claude_client.time") as mock_time:
            mock_client = MagicMock()
            mock_boto3.client.return_value = mock_client
            mock_client.invoke_model.side_effect = [
                throttle_error,
                _bedrock_response(_valid_llm_json("doc-1")),
            ]

            with patch(
                "src.recommendation_engine.claude_client.check_grounding",
                new=AsyncMock(return_value=_grounded()),
            ):
                recommender = ClaudeRecommender()
                result = await recommender.recommend("query", context_docs)

    assert isinstance(result, DispatchRecommendation)
    mock_time.sleep.assert_called_once()


# ---------------------------------------------------------------------------
# ClaudeRecommender — safety flags
# ---------------------------------------------------------------------------


async def test_claude_client_low_confidence_triggers_human_review() -> None:
    """Confidence below 0.60 sets requires_human_review=True."""
    context_docs = [_doc("doc-1")]

    with patch("src.recommendation_engine.claude_client.boto3") as mock_boto3:
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.invoke_model.return_value = _bedrock_response(
            _valid_llm_json("doc-1", confidence=0.45)
        )

        with patch(
            "src.recommendation_engine.claude_client.check_grounding",
            new=AsyncMock(return_value=_grounded()),
        ):
            recommender = ClaudeRecommender()
            result = await recommender.recommend("query", context_docs)

    assert result.requires_human_review is True
    assert result.hallucination_flagged is False


async def test_claude_client_hallucination_sets_both_flags() -> None:
    """Ungrounded response sets hallucination_flagged and requires_human_review."""
    context_docs = [_doc("doc-1")]

    with patch("src.recommendation_engine.claude_client.boto3") as mock_boto3:
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.invoke_model.return_value = _bedrock_response(
            _valid_llm_json("doc-1", confidence=0.8)
        )

        with patch(
            "src.recommendation_engine.claude_client.check_grounding",
            new=AsyncMock(return_value=_ungrounded()),
        ):
            recommender = ClaudeRecommender()
            result = await recommender.recommend("query", context_docs)

    assert result.hallucination_flagged is True
    assert result.requires_human_review is True


async def test_claude_client_recommendation_is_frozen() -> None:
    """DispatchRecommendation is frozen — attribute assignment raises AttributeError."""
    context_docs = [_doc("doc-1")]

    with patch("src.recommendation_engine.claude_client.boto3") as mock_boto3:
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.invoke_model.return_value = _bedrock_response(_valid_llm_json("doc-1"))

        with patch(
            "src.recommendation_engine.claude_client.check_grounding",
            new=AsyncMock(return_value=_grounded()),
        ):
            recommender = ClaudeRecommender()
            result = await recommender.recommend("query", context_docs)

    from pydantic import ValidationError
    with pytest.raises((AttributeError, TypeError, ValidationError)):
        result.action = "mutated"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# OllamaRecommender — happy path + schema parity
# ---------------------------------------------------------------------------


async def test_ollama_client_returns_dispatch_recommendation() -> None:
    """OllamaRecommender returns the same DispatchRecommendation schema."""
    context_docs = [_doc("doc-1")]

    with patch("src.recommendation_engine.ollama_client.httpx.AsyncClient"):
        with patch(
            "src.recommendation_engine.ollama_client.check_grounding",
            new=AsyncMock(return_value=_grounded()),
        ):
            recommender = OllamaRecommender()

            with patch.object(
                recommender,
                "_call_ollama",
                new=AsyncMock(return_value=_valid_llm_json("doc-1")),
            ):
                result = await recommender.recommend("North constraint", context_docs)

    assert isinstance(result, DispatchRecommendation)
    assert result.action == "Issue upward dispatch instruction to DRAX1G for 200 MW"
    assert result.confidence == pytest.approx(0.83)
    assert result.requires_human_review is False
    assert result.hallucination_flagged is False


async def test_ollama_client_makes_no_aws_calls() -> None:
    """OllamaRecommender must not call boto3 — sovereign mode has no AWS access."""
    context_docs = [_doc("doc-1")]

    with patch("src.recommendation_engine.ollama_client.httpx.AsyncClient"):
        with patch(
            "src.recommendation_engine.ollama_client.check_grounding",
            new=AsyncMock(return_value=_grounded()),
        ):
            recommender = OllamaRecommender()

            with patch.object(
                recommender,
                "_call_ollama",
                new=AsyncMock(return_value=_valid_llm_json("doc-1")),
            ):
                with patch("boto3.client") as mock_boto3:
                    await recommender.recommend("query", context_docs)
                    mock_boto3.assert_not_called()


async def test_ollama_client_retries_on_invalid_json() -> None:
    """OllamaRecommender retries once on invalid JSON, same as ClaudeRecommender."""
    context_docs = [_doc("doc-1")]

    with patch("src.recommendation_engine.ollama_client.httpx.AsyncClient"):
        with patch(
            "src.recommendation_engine.ollama_client.check_grounding",
            new=AsyncMock(return_value=_grounded()),
        ):
            recommender = OllamaRecommender()

            mock_call = AsyncMock(
                side_effect=["not valid json at all", _valid_llm_json("doc-1")]
            )
            with patch.object(recommender, "_call_ollama", new=mock_call):
                result = await recommender.recommend("query", context_docs)

    assert isinstance(result, DispatchRecommendation)
    assert mock_call.call_count == 2


# ---------------------------------------------------------------------------
# OllamaRecommender — health check
# ---------------------------------------------------------------------------


async def test_ollama_health_check_raises_when_not_running() -> None:
    """check_health raises RuntimeError when Ollama server is not reachable."""
    import httpx

    with patch("src.recommendation_engine.ollama_client.httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
        mock_cls.return_value = mock_client
        recommender = OllamaRecommender()

        with pytest.raises(RuntimeError, match="Cannot connect to Ollama"):
            await recommender.check_health()


async def test_ollama_health_check_raises_when_model_missing() -> None:
    """check_health raises RuntimeError when configured model is not pulled."""
    with patch("src.recommendation_engine.ollama_client.httpx.AsyncClient") as mock_cls:
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"models": [{"name": "other-model:latest"}]}

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_cls.return_value = mock_client
        recommender = OllamaRecommender(model="llama3.2:3b")

        with pytest.raises(RuntimeError, match="not available in Ollama"):
            await recommender.check_health()

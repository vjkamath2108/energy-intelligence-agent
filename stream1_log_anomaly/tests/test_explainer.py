"""
Tests for BedrockExplainer and OllamaExplainer.

No real API calls — both explainers are tested via mocking.
This validates the interface contract and JSON parsing logic,
not the LLM output quality.
"""

import json
import pytest
from unittest.mock import AsyncMock, patch
from stream1_log_anomaly.src.models import LogEvent
from stream1_log_anomaly.src.models import Explanation
from stream1_log_anomaly.src.explanation.bedrock_explainer import BedrockExplainer
from stream1_log_anomaly.src.explanation.ollama_explainer import OllamaExplainer


VALID_EXPLANATION_JSON = json.dumps({
    "severity": "HIGH",
    "summary": "Auth service latency spike causing cascade failures.",
    "likely_cause": "Database connection pool exhaustion.",
    "recommended_action": "Restart auth-service and check DB connections.",
})


def make_event(latency_ms: int = 1624, level: str = "ERROR") -> LogEvent:
    return LogEvent(
        ts="2024-01-01T15:50:56",
        level=level,
        service="auth-service",
        latency_ms=latency_ms,
        status_code=500,
    )


# ── Bedrock ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_bedrock_explain_returns_explanation() -> None:
    explainer = BedrockExplainer()
    with patch(
        "stream1_log_anomaly.src.explanation.bedrock_explainer._call_bedrock",
        new=AsyncMock(return_value=VALID_EXPLANATION_JSON),
    ):
        result = await explainer.explain(make_event(), [make_event()])

    assert isinstance(result, Explanation)
    assert result.severity == "HIGH"
    assert result.recommended_action != ""


@pytest.mark.asyncio
async def test_bedrock_retries_on_invalid_json() -> None:
    explainer = BedrockExplainer()
    with patch(
        "stream1_log_anomaly.src.explanation.bedrock_explainer._call_bedrock",
        new=AsyncMock(side_effect=[
            "not valid json",           # first call fails
            VALID_EXPLANATION_JSON,     # retry succeeds
        ]),
    ):
        result = await explainer.explain(make_event(), [make_event()])

    assert isinstance(result, Explanation)


# ── Ollama ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_ollama_explain_returns_explanation() -> None:
    explainer = OllamaExplainer()
    with patch(
        "stream1_log_anomaly.src.explanation.ollama_explainer.OllamaExplainer._call_ollama",
        new=AsyncMock(return_value=VALID_EXPLANATION_JSON),
    ):
        result = await explainer.explain(make_event(), [make_event()])

    assert isinstance(result, Explanation)
    assert result.severity == "HIGH"


@pytest.mark.asyncio
async def test_ollama_retries_on_invalid_json() -> None:
    explainer = OllamaExplainer()
    with patch(
        "stream1_log_anomaly.src.explanation.ollama_explainer.OllamaExplainer._call_ollama",
        new=AsyncMock(side_effect=[
            "not valid json",
            VALID_EXPLANATION_JSON,
        ]),
    ):
        result = await explainer.explain(make_event(), [make_event()])

    assert isinstance(result, Explanation)


# ── Interface parity ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_bedrock_and_ollama_return_same_schema() -> None:
    """Both explainers must return identical Explanation schema."""
    bedrock = BedrockExplainer()
    ollama = OllamaExplainer()

    with patch(
        "stream1_log_anomaly.src.explanation.bedrock_explainer._call_bedrock",
        new=AsyncMock(return_value=VALID_EXPLANATION_JSON),
    ):
        bedrock_result = await bedrock.explain(make_event(), [])

    with patch(
        "stream1_log_anomaly.src.explanation.ollama_explainer.OllamaExplainer._call_ollama",
        new=AsyncMock(return_value=VALID_EXPLANATION_JSON),
    ):
        ollama_result = await ollama.explain(make_event(), [])

    assert type(bedrock_result) == type(ollama_result)
    assert bedrock_result.severity == ollama_result.severity
"""Unit tests for src/models.py.

All tests are pure Pydantic — no database, no network, no async.
Run with: .venv/bin/pytest tests/test_models.py -v
"""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from src.models import (
    AnomalyResult,
    AuditLogEntry,
    DispatchRecommendation,
    Explanation,
    LogEvent,
    RetrievedDocument,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_doc(**overrides: object) -> RetrievedDocument:
    defaults: dict = {
        "doc_id": "doc-1",
        "content": "Grid voltage must not exceed 415 V.",
        "source": "regulation/dno-p28.pdf",
        "chunk_index": 0,
        "similarity_score": 0.92,
    }
    return RetrievedDocument(**(defaults | overrides))


def _make_recommendation(**overrides: object) -> DispatchRecommendation:
    defaults: dict = {
        "query": "Should I curtail unit 3?",
        "action": "Curtail unit 3 to 50% output.",
        "confidence": 0.87,
        "supporting_documents": [_make_doc()],
        "reasoning": "Voltage constraint from P28 exceeded.",
    }
    return DispatchRecommendation(**(defaults | overrides))


def _make_log_event(**overrides: object) -> LogEvent:
    defaults: dict = {
        "timestamp": datetime.now(timezone.utc),
        "level": "ERROR",
        "service": "dispatch-api",
        "message": "Connection timeout after 5000 ms",
        "latency_ms": 5000.0,
        "status_code": 504,
    }
    return LogEvent(**(defaults | overrides))


# ---------------------------------------------------------------------------
# RetrievedDocument
# ---------------------------------------------------------------------------


def test_retrieved_document_defaults() -> None:
    """metadata defaults to an empty dict when not supplied."""
    doc = _make_doc()
    assert doc.metadata == {}


def test_retrieved_document_similarity_bounds() -> None:
    """similarity_score must be within [0.0, 1.0]."""
    with pytest.raises(ValidationError):
        _make_doc(similarity_score=1.1)
    with pytest.raises(ValidationError):
        _make_doc(similarity_score=-0.1)


# ---------------------------------------------------------------------------
# DispatchRecommendation
# ---------------------------------------------------------------------------


def test_dispatch_recommendation_auto_fields() -> None:
    """recommendation_id and created_at are auto-populated on construction."""
    rec = _make_recommendation()
    assert rec.recommendation_id, "recommendation_id should not be empty"
    assert rec.created_at.tzinfo is not None, "created_at must be timezone-aware"


def test_dispatch_recommendation_unique_ids() -> None:
    """Each instance gets a distinct recommendation_id."""
    a = _make_recommendation()
    b = _make_recommendation()
    assert a.recommendation_id != b.recommendation_id


def test_dispatch_recommendation_immutable() -> None:
    """frozen=True — attribute assignment must raise ValidationError."""
    rec = _make_recommendation()
    with pytest.raises((ValidationError, TypeError)):
        rec.action = "Do something else"  # type: ignore[misc]


def test_dispatch_recommendation_defaults() -> None:
    """requires_human_review and hallucination_flagged default to False."""
    rec = _make_recommendation()
    assert rec.requires_human_review is False
    assert rec.hallucination_flagged is False


# ---------------------------------------------------------------------------
# AuditLogEntry
# ---------------------------------------------------------------------------


def test_audit_log_entry_immutable() -> None:
    """frozen=True — attribute assignment must raise ValidationError."""
    entry = AuditLogEntry(
        recommendation_id="rec-abc",
        query="Should I curtail unit 3?",
        retrieved_doc_ids=["doc-1", "doc-2"],
        confidence=0.87,
        hallucination_detection_result=False,
        action_taken="Curtail unit 3 to 50% output.",
    )
    with pytest.raises((ValidationError, TypeError)):
        entry.action_taken = "Something else"  # type: ignore[misc]


def test_audit_log_entry_auto_fields() -> None:
    """entry_id and timestamp are auto-populated on construction."""
    entry = AuditLogEntry(
        recommendation_id="rec-abc",
        query="test query",
        retrieved_doc_ids=[],
        confidence=0.5,
        hallucination_detection_result=False,
        action_taken="no-op",
    )
    assert entry.entry_id, "entry_id should not be empty"
    assert entry.timestamp.tzinfo is not None, "timestamp must be timezone-aware"


# ---------------------------------------------------------------------------
# LogEvent
# ---------------------------------------------------------------------------


def test_log_event_valid_levels() -> None:
    """All three valid Literal values for level are accepted."""
    for level in ("INFO", "WARN", "ERROR"):
        event = _make_log_event(level=level)
        assert event.level == level


def test_log_event_invalid_level() -> None:
    """An unrecognised level value raises ValidationError."""
    with pytest.raises(ValidationError):
        _make_log_event(level="DEBUG")


def test_log_event_optional_fields() -> None:
    """latency_ms and status_code default to None."""
    event = _make_log_event(latency_ms=None, status_code=None)
    assert event.latency_ms is None
    assert event.status_code is None


# ---------------------------------------------------------------------------
# AnomalyResult
# ---------------------------------------------------------------------------


def test_anomaly_result_score_convention() -> None:
    """Negative anomaly_score with is_anomaly=True matches Isolation Forest convention."""
    result = AnomalyResult(
        event_id="evt-1",
        anomaly_score=-0.23,
        is_anomaly=True,
        window_size=100,
    )
    assert result.anomaly_score < 0
    assert result.is_anomaly is True


def test_anomaly_result_normal_point() -> None:
    """A score near 0 with is_anomaly=False represents a normal point."""
    result = AnomalyResult(
        event_id="evt-2",
        anomaly_score=0.05,
        is_anomaly=False,
        window_size=100,
    )
    assert result.is_anomaly is False


def test_anomaly_result_auto_detected_at() -> None:
    """detected_at is auto-populated with a timezone-aware UTC datetime."""
    result = AnomalyResult(
        event_id="evt-3",
        anomaly_score=-0.1,
        is_anomaly=True,
        window_size=50,
    )
    assert result.detected_at.tzinfo is not None


# ---------------------------------------------------------------------------
# Explanation
# ---------------------------------------------------------------------------


def test_explanation_severity_levels() -> None:
    """All four severity Literal values are accepted."""
    for severity in ("LOW", "MEDIUM", "HIGH", "CRITICAL"):
        exp = Explanation(
            event_id="evt-1",
            severity=severity,  # type: ignore[arg-type]
            summary="Latency spike detected.",
            likely_cause="Downstream DB contention.",
            recommended_action="Check RDS CloudWatch metrics.",
            explainer_backend="bedrock",
        )
        assert exp.severity == severity


def test_explanation_invalid_severity() -> None:
    """An unrecognised severity raises ValidationError."""
    with pytest.raises(ValidationError):
        Explanation(
            event_id="evt-1",
            severity="FATAL",  # type: ignore[arg-type]
            summary="x",
            likely_cause="x",
            recommended_action="x",
            explainer_backend="bedrock",
        )


def test_explanation_backends() -> None:
    """Both valid explainer_backend values are accepted."""
    for backend in ("bedrock", "ollama"):
        exp = Explanation(
            event_id="evt-1",
            severity="LOW",
            summary="x",
            likely_cause="x",
            recommended_action="x",
            explainer_backend=backend,  # type: ignore[arg-type]
        )
        assert exp.explainer_backend == backend


def test_explanation_invalid_backend() -> None:
    """An unrecognised explainer_backend raises ValidationError."""
    with pytest.raises(ValidationError):
        Explanation(
            event_id="evt-1",
            severity="LOW",
            summary="x",
            likely_cause="x",
            recommended_action="x",
            explainer_backend="vllm",  # type: ignore[arg-type]
        )

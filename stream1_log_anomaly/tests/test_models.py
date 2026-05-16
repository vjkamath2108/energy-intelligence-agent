"""Smoke tests for Stream 1 core models and anomaly scorer."""
import pytest
from stream1_log_anomaly.src.models import LogEvent, Explanation
from stream1_log_anomaly.src.detection.anomaly_scorer import AnomalyScorer


def _make_event(**kwargs) -> LogEvent:
    defaults = {
        "ts": "2026-05-01T08:00:00",
        "service": "auth-service",
        "level": "INFO",
        "latency_ms": 100,
        "status_code": 200,
    }
    return LogEvent(**{**defaults, **kwargs})


def test_log_event_valid():
    event = _make_event()
    assert event.service == "auth-service"
    assert event.latency_ms == 100


def test_log_event_invalid_missing_field():
    with pytest.raises(Exception):
        LogEvent(ts="2026-05-01T08:00:00", service="auth-service")


def test_explanation_valid():
    e = Explanation(
        severity="HIGH",
        summary="Test summary",
        likely_cause="Test cause",
        recommended_action="Test action",
    )
    assert e.severity == "HIGH"


def test_anomaly_scorer_buffers_before_fitting():
    scorer = AnomalyScorer()
    event = _make_event()
    # First 20 events should return None (buffering)
    for _ in range(19):
        result = scorer.score(event)
        assert result is None


def test_anomaly_scorer_scores_after_buffer():
    scorer = AnomalyScorer()
    normal = _make_event()
    for _ in range(20):
        scorer.score(normal)
    # After buffer, should return a result
    result = scorer.score(normal)
    assert result is not None
    assert isinstance(result.is_anomaly, bool)
    assert isinstance(result.anomaly_score, float)


def test_anomaly_scorer_detects_spike():
    scorer = AnomalyScorer()
    import random
    rng = random.Random(42)
    # Train with varied normal events so IF can learn the baseline
    for _ in range(20):
        scorer.score(_make_event(
            latency_ms=rng.randint(80, 120),
            status_code=200,
            level="INFO"
        ))
    # Inject an extreme spike — well outside training distribution
    spike = _make_event(latency_ms=9999, status_code=500, level="ERROR")
    result = scorer.score(spike)
    assert result is not None
    assert result.anomaly_score < -0.5  # strongly anomalous score

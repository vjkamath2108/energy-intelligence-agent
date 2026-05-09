"""Tests for Isolation Forest anomaly scorer."""

import numpy as np
import pytest
from stream1_log_anomaly.src.models import LogEvent
from stream1_log_anomaly.src.detection.anomaly_scorer import AnomalyScorer


def make_event(
    latency_ms: int = 100,
    level: str = "INFO",
    status_code: int = 200,
) -> LogEvent:
    return LogEvent(
        ts="2024-01-01T15:50:00",
        level=level,
        service="test-service",
        latency_ms=latency_ms,
        status_code=status_code,
    )


def normal_events(n: int = 20) -> list[LogEvent]:
    """Generate varied but normal events to give IF a realistic distribution."""
    import random
    random.seed(42)
    return [
        make_event(
            latency_ms=random.randint(50, 300),
            level=random.choice(["INFO", "INFO", "INFO", "WARN"]),
            status_code=random.choice([200, 200, 200, 200, 500]),
        )
        for _ in range(n)
    ]


def test_scorer_returns_none_while_buffering() -> None:
    scorer = AnomalyScorer()
    for event in normal_events(19):
        result = scorer.score(event)
        assert result is None


def test_scorer_fits_after_buffer_full() -> None:
    scorer = AnomalyScorer()
    for event in normal_events(20):
        scorer.score(event)
    # model should be fitted now
    assert scorer._model is not None


def test_scorer_returns_result_after_buffer() -> None:
    scorer = AnomalyScorer()
    for event in normal_events(20):
        scorer.score(event)
    result = scorer.score(make_event())
    assert result is not None
    assert isinstance(result.anomaly_score, float)
    assert isinstance(result.is_anomaly, (bool, np.bool_))


def test_scorer_detects_injected_anomaly() -> None:
    scorer = AnomalyScorer()
    for event in normal_events(20):
        scorer.score(event)
    for event in normal_events(10):
        scorer.score(event)

    anomaly = make_event(latency_ms=2500, level="ERROR", status_code=500)
    result = scorer.score(anomaly)

    assert result is not None
    # Score should be clearly more negative than normal events
    assert result.anomaly_score < -0.5
    assert result.is_anomaly == True


def test_scorer_normal_event_not_flagged() -> None:
    scorer = AnomalyScorer()
    for event in normal_events(20):
        scorer.score(event)

    result = scorer.score(make_event(latency_ms=95, level="INFO", status_code=200))
    assert result is not None
    assert result.is_anomaly == False
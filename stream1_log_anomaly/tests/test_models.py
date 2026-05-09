"""Tests for LogEvent Pydantic model validation."""

import pytest
from stream1_log_anomaly.src.models import LogEvent


def valid_event_dict() -> dict:
    return {
        "ts": "2024-01-01T15:50:56",
        "level": "INFO",
        "service": "auth-service",
        "latency_ms": 90,
        "status_code": 200,
    }


def test_valid_log_event() -> None:
    event = LogEvent(**valid_event_dict())
    assert event.level == "INFO"
    assert event.service == "auth-service"
    assert event.latency_ms == 90
    assert event.status_code == 200


def test_log_event_error_level() -> None:
    data = valid_event_dict()
    data["level"] = "ERROR"
    data["latency_ms"] = 1624
    data["status_code"] = 500
    event = LogEvent(**data)
    assert event.level == "ERROR"
    assert event.latency_ms == 1624


def test_log_event_missing_field_raises() -> None:
    data = valid_event_dict()
    del data["latency_ms"]
    with pytest.raises(Exception):
        LogEvent(**data)


def test_log_event_invalid_type_raises() -> None:
    data = valid_event_dict()
    data["latency_ms"] = "not_a_number"
    with pytest.raises(Exception):
"""
Stream 1 data models.

All pipeline data shapes defined here. Import from this module only —
never define models inline in individual components.
"""

from pydantic import BaseModel, ConfigDict, Field


class LogEvent(BaseModel):
    """A single parsed log event from the tail consumer or Redis stream."""

    model_config = ConfigDict(frozen=True)

    ts: str
    level: str  # INFO | WARN | ERROR
    service: str
    latency_ms: int
    status_code: int = Field(ge=100, le=599)


class AnomalyResult(BaseModel):
    """Output of AnomalyScorer for a single log event."""

    model_config = ConfigDict(frozen=True)

    event: LogEvent
    anomaly_score: float
    is_anomaly: bool
    window_id: int
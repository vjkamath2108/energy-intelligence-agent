"""Pydantic v2 data models for the Energy Intelligence Agent.

Three groups:
  - RAG pipeline:  RetrievedDocument, DispatchRecommendation, AuditLogEntry
  - Stream 1:      LogEvent, AnomalyResult, Explanation
"""

import uuid
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# RAG pipeline models
# ---------------------------------------------------------------------------


class RetrievedDocument(BaseModel):
    """A single document chunk returned by the retriever.

    Carries both the text content and provenance metadata so the
    recommendation engine can cite sources and the audit trail can
    record exactly which chunks influenced a decision.
    """

    doc_id: str = Field(description="Unique identifier for this document chunk.")
    content: str = Field(description="Raw text of the retrieved chunk.")
    source: str = Field(
        description="Origin path or identifier, e.g. 'regulation/dno-p28.pdf'."
    )
    chunk_index: int = Field(
        description="Zero-based position of this chunk within its source document."
    )
    similarity_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Retrieval score in [0, 1]; higher means more relevant.",
    )
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Arbitrary key-value pairs attached by the ingestion pipeline.",
    )


class DispatchRecommendation(BaseModel):
    """A safety-critical dispatch recommendation produced by the RAG pipeline.

    Frozen after creation — downstream consumers must not mutate this object.
    Any change to a recommendation must produce a new instance with a new id.
    """

    model_config = ConfigDict(frozen=True)

    recommendation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Stable UUID4 identifier for this recommendation.",
    )
    query: str = Field(
        description="The original natural-language query from the operator."
    )
    action: str = Field(
        description="The recommended dispatch action in plain language."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Model confidence in [0, 1]; calibrated against a validation set.",
    )
    supporting_documents: list[RetrievedDocument] = Field(
        description="Ordered list of retrieved chunks that support this recommendation."
    )
    reasoning: str = Field(
        description="Chain-of-thought explanation produced by the LLM."
    )
    requires_human_review: bool = Field(
        default=False,
        description="True when confidence is below threshold or anomalies were detected.",
    )
    hallucination_flagged: bool = Field(
        default=False,
        description="True when the hallucination detector rejected the response.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp at which this recommendation was created.",
    )


class AuditLogEntry(BaseModel):
    """Immutable record of a single recommendation decision for audit purposes.

    Append-only — frozen after creation. Never mutate or delete entries;
    corrections are expressed as new entries referencing the original id.
    """

    model_config = ConfigDict(frozen=True)

    entry_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Stable UUID4 identifier for this audit entry.",
    )
    recommendation_id: str = Field(
        description="Foreign key linking back to the DispatchRecommendation."
    )
    query: str = Field(description="The original query that triggered this decision.")
    retrieved_doc_ids: list[str] = Field(
        description="Ordered list of doc_ids that were retrieved and used."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score recorded at decision time.",
    )
    hallucination_detection_result: bool = Field(
        description="True if hallucination was detected; False if the response was clean."
    )
    action_taken: str = Field(
        description="The dispatch action that was surfaced to the operator."
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp at which this audit entry was recorded.",
    )


# ---------------------------------------------------------------------------
# Stream 1 — real-time log anomaly detection
# ---------------------------------------------------------------------------


class LogEvent(BaseModel):
    """A single structured log line from the EKS workload.

    Produced by the log generator and consumed by the tail / Redis consumer
    before being scored by the Isolation Forest anomaly detector.
    """

    event_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier assigned at ingestion time.",
    )
    timestamp: datetime = Field(
        description="Wall-clock time at which the log line was emitted."
    )
    level: Literal["INFO", "WARN", "ERROR"] = Field(description="Log severity level.")
    service: str = Field(
        description="Name of the service or pod that emitted the log line."
    )
    message: str = Field(description="Raw log message text.")
    latency_ms: float | None = Field(
        default=None,
        description="Request latency in milliseconds, if applicable.",
    )
    status_code: int | None = Field(
        default=None,
        description="HTTP status code, if applicable.",
    )


class AnomalyResult(BaseModel):
    """Output of the Isolation Forest scorer for a single log event.

    Isolation Forest returns negative scores for anomalies and scores close
    to zero (or positive) for normal points — is_anomaly reflects this
    convention rather than inverting it.
    """

    event_id: str = Field(
        description="References the LogEvent.event_id that was scored."
    )
    anomaly_score: float = Field(
        description=(
            "Raw Isolation Forest decision score. "
            "Negative values indicate anomalies; values near 0 indicate normal points."
        )
    )
    is_anomaly: bool = Field(
        description="True when anomaly_score is below the configured threshold."
    )
    detected_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp at which the anomaly score was computed.",
    )
    window_size: int = Field(
        description="Number of events in the sliding window used for scoring."
    )


class Explanation(BaseModel):
    """LLM-generated natural-language explanation for a detected anomaly.

    Produced by the Bedrock or Ollama explainer backend and attached to the
    AnomalyResult before being surfaced to the operator.
    """

    event_id: str = Field(
        description="References the LogEvent.event_id that was explained."
    )
    severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = Field(
        description="Operator-facing severity classification assigned by the LLM."
    )
    summary: str = Field(
        description="One-sentence plain-language summary of the anomaly."
    )
    likely_cause: str = Field(
        description="LLM assessment of the most probable root cause."
    )
    recommended_action: str = Field(
        description="Suggested remediation step for the on-call operator."
    )
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp at which the explanation was generated.",
    )
    explainer_backend: Literal["bedrock", "ollama"] = Field(
        description="Inference backend that produced this explanation."
    )

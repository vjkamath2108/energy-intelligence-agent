"""
Append-only anomaly log writer.

Writes each explanation to a JSONL file for the Streamlit dashboard.
Every line is a self-contained JSON record — easy to tail, easy to parse.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from stream1_log_anomaly.src.models import LogEvent, Explanation

logger = logging.getLogger(__name__)

LOG_PATH = Path("results/anomaly_log.jsonl")


def append_explanation(
    event: LogEvent,
    explanation: Explanation,
    backend: str,
    latency_ms: float,
) -> None:
    """
    Append a single explanation record to the anomaly log.

    Args:
        event: The anomalous LogEvent that triggered the explanation.
        explanation: The structured Explanation from the LLM.
        backend: "bedrock" or "ollama".
        latency_ms: Time taken to generate the explanation in ms.
    """
    record = {
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "ts": event.ts,
        "service": event.service,
        "level": event.level,
        "event_latency_ms": event.latency_ms,
        "status_code": event.status_code,
        "backend": backend,
        "explanation_latency_ms": latency_ms,
        "severity": explanation.severity,
        "summary": explanation.summary,
        "likely_cause": explanation.likely_cause,
        "recommended_action": explanation.recommended_action,
    }

    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a") as f:
            f.write(json.dumps(record) + "\n")
        logger.debug("Appended explanation to %s", LOG_PATH)
    except OSError as e:
        logger.warning("Failed to write anomaly log: %s", e)
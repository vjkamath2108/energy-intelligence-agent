"""
Isolation Forest anomaly scorer.

Buffers incoming log events until a minimum window is reached, then fits
an Isolation Forest model and scores each subsequent event.

Design decisions:
- Buffer 20 events before first fit: enough to characterise normal behaviour
  while ensuring the anomaly cluster (events 33-37) falls in the scoring window.
- contamination=0.1: 5 anomalies in 50 events = 10%. In production, start at
  0.05 and tune from observation — you won't know the true contamination rate.
- Features: latency_ms (continuous), is_error (binary), is_500 (binary).
  Three signals fire simultaneously at the anomaly cluster — IF separates
  this cleanly without needing the event count or timestamp.
- Classical ML scores, LLM explains. IF never calls Bedrock. Ever.
"""

import logging
from dataclasses import dataclass, field

from sklearn.ensemble import IsolationForest  # type: ignore[import]
import numpy as np
from stream1_log_anomaly.src.models import LogEvent


logger = logging.getLogger(__name__)

BUFFER_SIZE = 20
CONTAMINATION = 0.1


@dataclass
class AnomalyResult:
    """
    Output of the anomaly scorer for a single log event.

    Attributes:
        event: The original log event dict.
        anomaly_score: Raw Isolation Forest score. More negative = more anomalous.
        is_anomaly: True if scorer flagged this event as anomalous.
        window_id: Which scoring window produced this result (0-indexed).
    """
    event: dict
    anomaly_score: float
    is_anomaly: bool
    window_id: int


@dataclass
class AnomalyScorer:
    """
    Buffers log events, fits Isolation Forest, scores new events.

    Usage:
        scorer = AnomalyScorer()
        for event in events:
            result = scorer.score(event)
            if result is not None and result.is_anomaly:
                # trigger explainer
    """
    buffer_size: int = BUFFER_SIZE
    contamination: float = CONTAMINATION
    _buffer: list[dict] = field(default_factory=list, repr=False)
    _model: IsolationForest | None = field(default=None, repr=False)
    _window_id: int = field(default=0, repr=False)

    #
    def _extract_features(self, event: LogEvent) -> list[float]:
        """
        Extract numeric feature vector from a log event.

        Features:
            latency_ms: Raw latency in milliseconds.
            is_error: 1.0 if level is ERROR, else 0.0.
            is_500: 1.0 if status_code >= 500, else 0.0.
        """
        return [
            float(event.latency_ms),
            1.0 if event.level == "ERROR" else 0.0,
            1.0 if int(event.status_code) >= 500 else 0.0,
        ]

    def _fit(self) -> None:
        """Fit Isolation Forest on the current buffer."""
        features = [self._extract_features(e) for e in self._buffer]
        X = np.array(features)
        self._model = IsolationForest(
            contamination=self.contamination,
            random_state=42,
        )
        self._model.fit(X)
        logger.info(
            "Isolation Forest fitted on %d events (window %d)",
            len(self._buffer),
            self._window_id,
        )

    def score(self, event: LogEvent) -> AnomalyResult | None:
        """
        Score a single log event.

        Returns None while buffering. Returns AnomalyResult once the model
        is fitted and scoring begins.

        Args:
            event: Parsed log event dict with keys: ts, level, service,
                   latency_ms, status_code.

        Returns:
            AnomalyResult if scoring, None if still buffering.
        """
        if len(self._buffer) < self.buffer_size:
            self._buffer.append(event)
            logger.debug(
                "Buffering event %d/%d", len(self._buffer), self.buffer_size
            )
            if len(self._buffer) == self.buffer_size:
                self._fit()
            return None

        assert self._model is not None
        features = np.array([self._extract_features(event)])

        # score_samples returns negative values — more negative = more anomalous
        raw_score = float(self._model.score_samples(features)[0])

        # predict returns -1 for anomaly, 1 for normal
        is_anomaly = self._model.predict(features)[0] == -1

        result = AnomalyResult(
            event=event,
            anomaly_score=raw_score,
            is_anomaly=is_anomaly,
            window_id=self._window_id,
        )

        if is_anomaly:
            logger.warning(
                "ANOMALY detected | service=%s level=%s latency=%sms score=%.4f",
                event.service,
                event.level,
                event.latency_ms,
                raw_score,
            )

        return result
if __name__ == "__main__":
    import json
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    events = [
        json.loads(line)
        for line in Path("stream1_log_anomaly/logs/sample.jsonl").read_text().splitlines()
        if line.strip()
    ]

    scorer = AnomalyScorer()
    anomalies_detected = 0

    for i, event in enumerate(events):
        result = scorer.score(event)
        if result is None:
            continue
        status = "🚨 ANOMALY" if result.is_anomaly else "  normal "
        print(
            f"[{i:03d}] {status} | score={result.anomaly_score:+.4f} | "
            f"{event.get('level')} {event.get('service')} "
            f"latency={event.get('latency_ms')}ms"
        )
        if result.is_anomaly:
            anomalies_detected += 1

    print(f"\nDetected {anomalies_detected} anomalies in {len(events)} events.")
    print(f"Expected anomaly cluster: events 033-037")
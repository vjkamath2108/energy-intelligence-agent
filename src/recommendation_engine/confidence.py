"""Confidence scoring and calibration for EIA dispatch recommendations.

Two responsibilities:
  1. Composite scoring: combine LLM self-reported confidence with retrieval
     signals (grounding quality, mean citation similarity) into a single score.
  2. Calibration: fit Platt scaling on a labelled validation set so that
     predicted probabilities match observed outcomes. EIA gate: Brier score < 0.20.

Composite weights (tuned on Stream 1 benchmark outputs):
  LLM self-confidence  60 %  — primary signal; LLM has context we do not
  Grounding quality    25 %  — term overlap (+ semantic similarity when available)
  Retrieval quality    15 %  — mean similarity_score of cited documents

Platt scaling: fits σ(A·score + B) via logistic regression with negligible
regularisation. Falls back to raw score when the calibrator has not been fitted
(safe for cold-start and unit-test contexts).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]

from src.models import RetrievedDocument
from src.recommendation_engine.safety import HUMAN_REVIEW_THRESHOLD, SafetyResult

logger = logging.getLogger(__name__)

_LLM_WEIGHT: float = 0.60
_GROUNDING_WEIGHT: float = 0.25
_RETRIEVAL_WEIGHT: float = 0.15


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfidenceSignals:
    """Raw signals collected before computing a composite confidence score.

    Args:
        llm_confidence: Self-reported confidence from the LLM in [0, 1].
        safety_result: Output of the hallucination grounding check.
        supporting_documents: Document chunks cited in the recommendation.
    """

    llm_confidence: float
    safety_result: SafetyResult
    supporting_documents: list[RetrievedDocument]


@dataclass(frozen=True)
class CalibrationRecord:
    """A single (predicted_score, actual_outcome) pair for calibration fitting.

    Args:
        predicted_score: Raw composite confidence score in [0, 1].
        outcome: True when the recommendation was accepted without human override.
    """

    predicted_score: float
    outcome: bool


# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------


def compute_composite_score(signals: ConfidenceSignals) -> float:
    """Combine LLM confidence, grounding quality, and retrieval similarity.

    Args:
        signals: Raw signals from the recommendation pipeline.

    Returns:
        Float in [0, 1] — higher means more confident.
    """
    grounding_score = signals.safety_result.term_overlap_ratio
    if signals.safety_result.semantic_similarity is not None:
        grounding_score = (grounding_score + signals.safety_result.semantic_similarity) / 2.0

    if signals.supporting_documents:
        retrieval_score = sum(
            d.similarity_score for d in signals.supporting_documents
        ) / len(signals.supporting_documents)
    else:
        retrieval_score = 0.0

    composite = (
        _LLM_WEIGHT * signals.llm_confidence
        + _GROUNDING_WEIGHT * grounding_score
        + _RETRIEVAL_WEIGHT * retrieval_score
    )
    return float(np.clip(composite, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def compute_brier_score(records: Sequence[CalibrationRecord]) -> float:
    """Mean squared error between predicted probability and actual outcome.

    Perfect calibration = 0.0. Random classifier ≈ 0.25. EIA gate: < 0.20.

    Args:
        records: Validation pairs of (predicted_score, actual_outcome).

    Returns:
        Brier score in [0, 1].

    Raises:
        ValueError: If records is empty.
    """
    if not records:
        raise ValueError("Cannot compute Brier score on empty validation set")

    scores: NDArray[np.float64] = np.array([r.predicted_score for r in records])
    outcomes: NDArray[np.float64] = np.array([float(r.outcome) for r in records])
    return float(np.mean((scores - outcomes) ** 2))


class PlattCalibrator:
    """Platt scaling: logistic sigmoid mapping raw scores to calibrated probabilities.

    Fits σ(A·score + B) via logistic regression with negligible regularisation
    (C=1e10). Falls back to the raw score before fitting — safe for cold-start.

    Usage:
        calibrator = PlattCalibrator()
        calibrator.fit(validation_records)   # needs both True and False outcomes
        p = calibrator.calibrate(0.73)       # calibrated probability
        b = compute_brier_score(records)     # verify b < 0.20
    """

    def __init__(self) -> None:
        self._model: LogisticRegression | None = None

    def fit(self, records: Sequence[CalibrationRecord]) -> None:
        """Fit Platt scaling from (score, outcome) validation pairs.

        Args:
            records: Validation set — at least 10 records recommended.

        Raises:
            ValueError: If records is empty or contains only one class.
        """
        if not records:
            raise ValueError("Cannot fit calibrator on empty record set")

        X: NDArray[np.float64] = np.array([[r.predicted_score] for r in records])
        y: NDArray[np.int64] = np.array([int(r.outcome) for r in records])

        if len(np.unique(y)) < 2:
            raise ValueError(
                "Calibration set must contain both positive and negative outcomes"
            )

        self._model = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
        self._model.fit(X, y)

        brier = compute_brier_score(records)
        logger.info(
            "PlattCalibrator fitted on %d records — Brier score %.4f (gate: <0.20)",
            len(records),
            brier,
        )
        if brier >= 0.20:
            logger.warning(
                "Brier score %.4f ≥ 0.20 — calibration does not meet quality gate",
                brier,
            )

    def calibrate(self, raw_score: float) -> float:
        """Apply the fitted Platt transform to a raw composite score.

        Falls back to the raw score (clamped to [0, 1]) when not yet fitted.

        Args:
            raw_score: Raw composite confidence in [0, 1].

        Returns:
            Calibrated probability in [0, 1].
        """
        if self._model is None:
            logger.debug("PlattCalibrator not fitted — returning raw score unchanged")
            return float(np.clip(raw_score, 0.0, 1.0))

        X: NDArray[np.float64] = np.array([[raw_score]])
        prob: float = float(self._model.predict_proba(X)[0][1])
        return float(np.clip(prob, 0.0, 1.0))

    @property
    def is_fitted(self) -> bool:
        """True when the calibrator has been fitted on validation data."""
        return self._model is not None


# ---------------------------------------------------------------------------
# Main scorer
# ---------------------------------------------------------------------------


class ConfidenceScorer:
    """Compute calibrated confidence and human-review flag for a recommendation.

    Computes a composite score from pipeline signals, optionally applies Platt
    calibration, then evaluates the HUMAN_REVIEW_THRESHOLD gate.

    Args:
        calibrator: Fitted PlattCalibrator. When None or unfitted, raw composite
                    score is used directly.

    Usage:
        scorer = ConfidenceScorer(calibrator=calibrator)
        confidence, requires_review = scorer.score(signals)
    """

    def __init__(self, calibrator: PlattCalibrator | None = None) -> None:
        self._calibrator = calibrator

    def score(self, signals: ConfidenceSignals) -> tuple[float, bool]:
        """Compute calibrated confidence and human review flag.

        Args:
            signals: Raw scoring signals from the recommendation pipeline.

        Returns:
            (confidence, requires_human_review) where confidence is in [0, 1]
            and requires_human_review is True when confidence < 0.60.
        """
        raw = compute_composite_score(signals)

        if self._calibrator is not None and self._calibrator.is_fitted:
            calibrated = self._calibrator.calibrate(raw)
        else:
            calibrated = raw

        requires_review = calibrated < HUMAN_REVIEW_THRESHOLD
        if requires_review:
            logger.info(
                "Confidence %.2f below threshold %.2f — flagging for human review",
                calibrated,
                HUMAN_REVIEW_THRESHOLD,
            )

        return calibrated, requires_review

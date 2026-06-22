"""Tests for src/recommendation_engine/confidence.py

Coverage:
- compute_composite_score: weight formula, semantic_similarity path, no-docs edge case
- compute_brier_score: correct MSE, empty-set error
- PlattCalibrator: fit / calibrate / is_fitted, unfitted fallback, edge-case errors
- ConfidenceScorer: uncalibrated path, calibrated path, human-review gate
"""

from __future__ import annotations

import pytest

from src.models import RetrievedDocument
from src.recommendation_engine.confidence import (
    CalibrationRecord,
    ConfidenceScorer,
    ConfidenceSignals,
    PlattCalibrator,
    compute_brier_score,
    compute_composite_score,
)
from src.recommendation_engine.safety import HUMAN_REVIEW_THRESHOLD, SafetyResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _doc(similarity: float = 0.8) -> RetrievedDocument:
    return RetrievedDocument(
        doc_id="doc-1",
        content="dispatch constraint upward balancing",
        source="regulation/dno-p28.pdf",
        chunk_index=0,
        similarity_score=similarity,
        metadata={},
    )


def _safety(
    is_grounded: bool = True,
    term_overlap: float = 0.8,
    semantic_similarity: float | None = None,
) -> SafetyResult:
    reasons: tuple[str, ...] = () if is_grounded else ("term overlap low",)
    return SafetyResult(
        is_grounded=is_grounded,
        term_overlap_ratio=term_overlap,
        semantic_similarity=semantic_similarity,
        flagged_reasons=reasons,
    )


def _signals(
    llm_confidence: float = 0.9,
    term_overlap: float = 0.8,
    semantic_similarity: float | None = None,
    doc_similarity: float = 0.8,
    n_docs: int = 1,
) -> ConfidenceSignals:
    return ConfidenceSignals(
        llm_confidence=llm_confidence,
        safety_result=_safety(term_overlap=term_overlap, semantic_similarity=semantic_similarity),
        supporting_documents=[_doc(doc_similarity)] * n_docs,
    )


def _well_calibrated_records(n: int = 50) -> list[CalibrationRecord]:
    """Alternating high/low scores with matching outcomes — Brier score ≈ 0.01."""
    records: list[CalibrationRecord] = []
    for i in range(n):
        if i % 2 == 0:
            records.append(CalibrationRecord(predicted_score=0.90, outcome=True))
        else:
            records.append(CalibrationRecord(predicted_score=0.10, outcome=False))
    return records


# ---------------------------------------------------------------------------
# compute_composite_score
# ---------------------------------------------------------------------------


def test_composite_score_expected_value() -> None:
    """60/25/15 weights applied correctly: 0.60*0.9 + 0.25*0.8 + 0.15*0.8 = 0.86."""
    score = compute_composite_score(_signals(llm_confidence=0.9, term_overlap=0.8, doc_similarity=0.8))
    assert score == pytest.approx(0.86)


def test_composite_score_with_semantic_similarity() -> None:
    """Grounding score averages term_overlap and semantic_similarity when both present."""
    # grounding = (0.6 + 0.8) / 2 = 0.7 → 0.60*0.9 + 0.25*0.7 + 0.15*0.8 = 0.54+0.175+0.12 = 0.835
    score = compute_composite_score(
        _signals(llm_confidence=0.9, term_overlap=0.6, semantic_similarity=0.8, doc_similarity=0.8)
    )
    assert score == pytest.approx(0.835)


def test_composite_score_no_supporting_docs() -> None:
    """Retrieval term is zero when no supporting docs are provided."""
    signals = ConfidenceSignals(
        llm_confidence=0.9,
        safety_result=_safety(term_overlap=0.8),
        supporting_documents=[],
    )
    # 0.60*0.9 + 0.25*0.8 + 0.15*0.0 = 0.74
    score = compute_composite_score(signals)
    assert score == pytest.approx(0.74)


def test_composite_score_clamps_to_zero_one() -> None:
    """Output is always in [0, 1] even with extreme input values."""
    high = _signals(llm_confidence=1.0, term_overlap=1.0, doc_similarity=1.0)
    low = ConfidenceSignals(
        llm_confidence=0.0,
        safety_result=_safety(term_overlap=0.0),
        supporting_documents=[_doc(0.0)],
    )
    assert 0.0 <= compute_composite_score(high) <= 1.0
    assert 0.0 <= compute_composite_score(low) <= 1.0


# ---------------------------------------------------------------------------
# compute_brier_score
# ---------------------------------------------------------------------------


def test_brier_score_perfect_calibration() -> None:
    """Predicted equals outcome → Brier score = 0."""
    records = [
        CalibrationRecord(predicted_score=1.0, outcome=True),
        CalibrationRecord(predicted_score=0.0, outcome=False),
    ]
    assert compute_brier_score(records) == pytest.approx(0.0)


def test_brier_score_worst_case() -> None:
    """Inverted predictions → Brier score = 1."""
    records = [
        CalibrationRecord(predicted_score=0.0, outcome=True),
        CalibrationRecord(predicted_score=1.0, outcome=False),
    ]
    assert compute_brier_score(records) == pytest.approx(1.0)


def test_brier_score_well_calibrated_below_gate() -> None:
    """Well-calibrated dataset produces Brier score below the 0.20 gate."""
    records = _well_calibrated_records(50)
    assert compute_brier_score(records) < 0.20


def test_brier_score_empty_raises() -> None:
    """Empty record list raises ValueError."""
    with pytest.raises(ValueError, match="empty validation set"):
        compute_brier_score([])


# ---------------------------------------------------------------------------
# PlattCalibrator
# ---------------------------------------------------------------------------


def test_platt_calibrator_is_fitted_after_fit() -> None:
    calibrator = PlattCalibrator()
    assert not calibrator.is_fitted

    calibrator.fit(_well_calibrated_records())
    assert calibrator.is_fitted


def test_platt_calibrator_unfitted_returns_raw_score() -> None:
    """Unfitted calibrator passes raw score through unchanged."""
    calibrator = PlattCalibrator()
    assert calibrator.calibrate(0.73) == pytest.approx(0.73)
    assert calibrator.calibrate(0.0) == pytest.approx(0.0)
    assert calibrator.calibrate(1.0) == pytest.approx(1.0)


def test_platt_calibrator_fitted_output_in_range() -> None:
    """Calibrated output is always in [0, 1] after fitting."""
    calibrator = PlattCalibrator()
    calibrator.fit(_well_calibrated_records())

    for raw in [0.0, 0.1, 0.5, 0.9, 1.0]:
        result = calibrator.calibrate(raw)
        assert 0.0 <= result <= 1.0, f"calibrate({raw}) = {result} out of range"


def test_platt_calibrator_high_score_yields_high_probability() -> None:
    """After fitting on well-calibrated data, high raw score → high calibrated probability."""
    calibrator = PlattCalibrator()
    calibrator.fit(_well_calibrated_records())
    assert calibrator.calibrate(0.9) > 0.5


def test_platt_calibrator_low_score_yields_low_probability() -> None:
    """After fitting on well-calibrated data, low raw score → low calibrated probability."""
    calibrator = PlattCalibrator()
    calibrator.fit(_well_calibrated_records())
    assert calibrator.calibrate(0.1) < 0.5


def test_platt_calibrator_empty_records_raises() -> None:
    with pytest.raises(ValueError, match="empty record set"):
        PlattCalibrator().fit([])


def test_platt_calibrator_single_class_raises() -> None:
    """All outcomes the same → cannot fit a sigmoid; must raise ValueError."""
    records = [CalibrationRecord(predicted_score=0.8, outcome=True)] * 10
    with pytest.raises(ValueError, match="both positive and negative outcomes"):
        PlattCalibrator().fit(records)


def test_platt_calibrator_brier_below_gate_after_fit() -> None:
    """Brier score on the same well-calibrated set remains below 0.20 after fitting."""
    records = _well_calibrated_records(60)
    calibrator = PlattCalibrator()
    calibrator.fit(records)

    calibrated_records = [
        CalibrationRecord(predicted_score=calibrator.calibrate(r.predicted_score), outcome=r.outcome)
        for r in records
    ]
    assert compute_brier_score(calibrated_records) < 0.20


# ---------------------------------------------------------------------------
# ConfidenceScorer
# ---------------------------------------------------------------------------


def test_scorer_uncalibrated_uses_composite_score() -> None:
    """Without a calibrator, scorer returns the raw composite score."""
    scorer = ConfidenceScorer()
    signals = _signals(llm_confidence=0.9, term_overlap=0.8, doc_similarity=0.8)
    confidence, _ = scorer.score(signals)
    assert confidence == pytest.approx(0.86)


def test_scorer_above_threshold_no_human_review() -> None:
    """Composite score above HUMAN_REVIEW_THRESHOLD → requires_human_review=False."""
    scorer = ConfidenceScorer()
    signals = _signals(llm_confidence=0.9, term_overlap=0.8, doc_similarity=0.8)
    _, requires_review = scorer.score(signals)
    assert not requires_review


def test_scorer_below_threshold_triggers_human_review() -> None:
    """Composite score below 0.60 → requires_human_review=True."""
    scorer = ConfidenceScorer()
    # 0.60*0.3 + 0.25*0.2 + 0.15*0.2 = 0.18+0.05+0.03 = 0.26 < 0.60
    signals = _signals(llm_confidence=0.3, term_overlap=0.2, doc_similarity=0.2)
    confidence, requires_review = scorer.score(signals)
    assert confidence < HUMAN_REVIEW_THRESHOLD
    assert requires_review


def test_scorer_with_fitted_calibrator() -> None:
    """With a fitted calibrator, scorer uses calibrated probabilities."""
    calibrator = PlattCalibrator()
    calibrator.fit(_well_calibrated_records())

    scorer = ConfidenceScorer(calibrator=calibrator)
    signals = _signals(llm_confidence=0.9, term_overlap=0.8, doc_similarity=0.8)
    confidence, _ = scorer.score(signals)

    # Calibrated output must be in [0, 1]
    assert 0.0 <= confidence <= 1.0


def test_scorer_with_unfitted_calibrator_falls_back_to_raw() -> None:
    """Unfitted calibrator → scorer falls back to raw composite score."""
    scorer = ConfidenceScorer(calibrator=PlattCalibrator())
    signals = _signals(llm_confidence=0.9, term_overlap=0.8, doc_similarity=0.8)
    confidence, _ = scorer.score(signals)
    assert confidence == pytest.approx(0.86)


def test_human_review_threshold_is_sixty_percent() -> None:
    """HUMAN_REVIEW_THRESHOLD exported from safety module is 0.60."""
    assert HUMAN_REVIEW_THRESHOLD == pytest.approx(0.60)

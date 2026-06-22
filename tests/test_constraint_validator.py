"""Tests for src/safety/constraint_validator.py

Gate: constraint validator catches regulatory violations in test cases.

Coverage:
- extract_constraints: MW ceiling, directional prohibition, unit prohibition
  all extracted from realistic UK energy regulatory text.
- validate_action: violations detected for each constraint type;
  compliant actions pass cleanly.
- check_constraints: convenience wrapper works end-to-end.
- Edge cases: no docs, no constraints, ambiguous action (no MW value), empty action.
- Data structure immutability: frozen dataclasses.
"""

from __future__ import annotations

import pytest

from src.models import RetrievedDocument
from src.safety.constraint_validator import (
    ExtractedConstraint,
    ValidationResult,
    check_constraints,
    extract_constraints,
    validate_action,
)


# ---------------------------------------------------------------------------
# Test documents with embedded regulatory constraints
# ---------------------------------------------------------------------------

# Realistic UK Grid Code / BSC language for each constraint type.

_MW_CEILING_DOC_CONTENT = (
    "Section 4 — Dispatch Limits. "
    "The nominated de-rated capacity for this BM unit must not exceed 500 MW "
    "under normal system conditions. "
    "Where system conditions require it, the System Operator may further restrict "
    "output to a limit of 300 MW. "
    "Participants must comply with any instruction received within gate closure."
)

_DIRECTIONAL_UP_DOC_CONTENT = (
    "Clause 7 — Operational Constraints During N-1 Contingency. "
    "Following the loss of the North–South interconnector, upward dispatch "
    "instructions are not permitted for BM units in the affected constraint zone. "
    "The System Operator will issue a stand-down notification once the constraint is cleared."
)

_DIRECTIONAL_DOWN_DOC_CONTENT = (
    "Article 3 — Voltage Management Protocol. "
    "Downward dispatch is prohibited during the voltage support window "
    "as defined in Schedule B. "
    "All generators must maintain output within the declared capability range."
)

_UNIT_PROHIBITION_DOC_CONTENT = (
    "Schedule A — Unit Availability. "
    "BM Unit DRAX1G must not be dispatched during the declared maintenance window "
    "running from 06:00 to 18:00 on weekdays. "
    "The unit operator must notify the System Operator 24 hours in advance of any "
    "deviation from the declared unavailability period."
)

_UNIT_PROHIBITION_ALT_CONTENT = (
    "Licence Condition C17 — Asset Constraints. "
    "RATS1G cannot be dispatched while the 400 kV busbar at Ratcliffe is under outage. "
    "Dispatch instructions to RATS1G during this period are automatically rejected "
    "by the control system."
)

_CLEAN_DOC_CONTENT = (
    "Section 1 — General Provisions. "
    "All participants must submit half-hourly metered data by D+1. "
    "Balancing Mechanism instructions are binding on all registered BM units. "
    "Gate closure applies to all offers and bids submitted after the deadline."
)


def _doc(
    content: str,
    doc_id: str = "doc-1",
    source: str = "regulations/grid-code.pdf",
) -> RetrievedDocument:
    return RetrievedDocument(
        doc_id=doc_id,
        content=content,
        source=source,
        chunk_index=0,
        similarity_score=0.9,
        metadata={},
    )


# ---------------------------------------------------------------------------
# extract_constraints — MW ceiling
# ---------------------------------------------------------------------------


def test_extract_mw_ceiling_from_must_not_exceed() -> None:
    """'must not exceed 500 MW' → mw_ceiling constraint with limit=500.0."""
    docs = [_doc(_MW_CEILING_DOC_CONTENT)]
    constraints = extract_constraints(docs)
    ceilings = [c for c in constraints if c.constraint_type == "mw_ceiling"]
    assert any(c.mw_limit == 500.0 for c in ceilings)


def test_extract_mw_ceiling_from_limit_to() -> None:
    """'limit of 300 MW' → mw_ceiling constraint with limit=300.0."""
    docs = [_doc(_MW_CEILING_DOC_CONTENT)]
    constraints = extract_constraints(docs)
    ceilings = [c for c in constraints if c.constraint_type == "mw_ceiling"]
    assert any(c.mw_limit == 300.0 for c in ceilings)


def test_extract_mw_ceiling_records_source_doc_id() -> None:
    """Extracted ceiling carries the source doc_id."""
    docs = [_doc(_MW_CEILING_DOC_CONTENT, doc_id="reg-001")]
    constraints = extract_constraints(docs)
    ceilings = [c for c in constraints if c.constraint_type == "mw_ceiling"]
    assert all(c.source_doc_id == "reg-001" for c in ceilings)


def test_extract_mw_ceiling_maximum_pattern() -> None:
    """'maximum output of 200 MW' pattern is extracted."""
    text = "The maximum active power output of 200 MW applies at all times."
    docs = [_doc(text)]
    constraints = extract_constraints(docs)
    ceilings = [c for c in constraints if c.constraint_type == "mw_ceiling"]
    assert any(c.mw_limit == 200.0 for c in ceilings)


# ---------------------------------------------------------------------------
# extract_constraints — directional prohibition
# ---------------------------------------------------------------------------


def test_extract_upward_prohibition() -> None:
    """'upward dispatch instructions are not permitted' → directional_prohibition upward."""
    docs = [_doc(_DIRECTIONAL_UP_DOC_CONTENT)]
    constraints = extract_constraints(docs)
    directs = [c for c in constraints if c.constraint_type == "directional_prohibition"]
    assert any(c.prohibited_direction == "upward" for c in directs)


def test_extract_downward_prohibition() -> None:
    """'Downward dispatch is prohibited' → directional_prohibition downward."""
    docs = [_doc(_DIRECTIONAL_DOWN_DOC_CONTENT)]
    constraints = extract_constraints(docs)
    directs = [c for c in constraints if c.constraint_type == "directional_prohibition"]
    assert any(c.prohibited_direction == "downward" for c in directs)


def test_extract_no_upward_dispatch_pattern() -> None:
    """'no upward dispatch allowed' pattern is caught."""
    text = "No upward dispatch is permitted in the constrained zone during red periods."
    docs = [_doc(text)]
    constraints = extract_constraints(docs)
    directs = [c for c in constraints if c.constraint_type == "directional_prohibition"]
    assert any(c.prohibited_direction == "upward" for c in directs)


# ---------------------------------------------------------------------------
# extract_constraints — unit prohibition
# ---------------------------------------------------------------------------


def test_extract_bm_unit_prohibition() -> None:
    """'BM Unit DRAX1G must not be dispatched' → unit_prohibition for DRAX1G."""
    docs = [_doc(_UNIT_PROHIBITION_DOC_CONTENT)]
    constraints = extract_constraints(docs)
    units = [c for c in constraints if c.constraint_type == "unit_prohibition"]
    assert any(c.prohibited_unit == "DRAX1G" for c in units)


def test_extract_unit_cannot_be_dispatched_pattern() -> None:
    """'RATS1G cannot be dispatched' pattern → unit_prohibition for RATS1G."""
    docs = [_doc(_UNIT_PROHIBITION_ALT_CONTENT)]
    constraints = extract_constraints(docs)
    units = [c for c in constraints if c.constraint_type == "unit_prohibition"]
    assert any(c.prohibited_unit == "RATS1G" for c in units)


# ---------------------------------------------------------------------------
# extract_constraints — clean doc produces no constraints
# ---------------------------------------------------------------------------


def test_clean_doc_produces_no_constraints() -> None:
    """Document with no regulatory constraint language returns empty list."""
    docs = [_doc(_CLEAN_DOC_CONTENT)]
    assert extract_constraints(docs) == []


def test_empty_doc_list_produces_no_constraints() -> None:
    assert extract_constraints([]) == []


# ---------------------------------------------------------------------------
# validate_action — MW ceiling violations
# ---------------------------------------------------------------------------


def test_mw_ceiling_violation_when_action_exceeds_limit() -> None:
    """Action dispatching 600 MW violates a 500 MW ceiling constraint."""
    constraint = ExtractedConstraint(
        source_doc_id="doc-1",
        source_text="Output must not exceed 500 MW.",
        constraint_type="mw_ceiling",
        mw_limit=500.0,
        prohibited_direction=None,
        prohibited_unit=None,
    )
    result = validate_action(
        "Issue upward dispatch to DRAX1G for 600 MW in settlement period 28",
        [constraint],
    )
    assert not result.passed
    assert len(result.violations) == 1
    assert "600" in result.violations[0].reason
    assert "500" in result.violations[0].reason


def test_mw_ceiling_passes_when_action_is_within_limit() -> None:
    """Action dispatching 400 MW complies with a 500 MW ceiling."""
    constraint = ExtractedConstraint(
        source_doc_id="doc-1",
        source_text="Output must not exceed 500 MW.",
        constraint_type="mw_ceiling",
        mw_limit=500.0,
        prohibited_direction=None,
        prohibited_unit=None,
    )
    result = validate_action(
        "Issue upward dispatch to DRAX1G for 400 MW",
        [constraint],
    )
    assert result.passed
    assert result.violations == ()


def test_mw_ceiling_passes_when_action_has_no_mw_value() -> None:
    """Conservative: action with no explicit MW value does not trigger ceiling violation."""
    constraint = ExtractedConstraint(
        source_doc_id="doc-1",
        source_text="Output must not exceed 500 MW.",
        constraint_type="mw_ceiling",
        mw_limit=500.0,
        prohibited_direction=None,
        prohibited_unit=None,
    )
    result = validate_action("Ramp up DRAX1G as needed.", [constraint])
    assert result.passed


# ---------------------------------------------------------------------------
# validate_action — directional prohibition violations
# ---------------------------------------------------------------------------


def test_upward_prohibition_catches_upward_action() -> None:
    """Action containing 'upward dispatch' violates an upward prohibition."""
    constraint = ExtractedConstraint(
        source_doc_id="doc-1",
        source_text="Upward dispatch is not permitted during N-1.",
        constraint_type="directional_prohibition",
        mw_limit=None,
        prohibited_direction="upward",
        prohibited_unit=None,
    )
    result = validate_action(
        "Issue upward BM dispatch instruction to TORN1G for 200 MW",
        [constraint],
    )
    assert not result.passed
    assert "upward" in result.violations[0].reason


def test_upward_prohibition_catches_increase_keyword() -> None:
    """Action using 'increase' triggers upward prohibition."""
    constraint = ExtractedConstraint(
        source_doc_id="doc-1",
        source_text="Upward dispatch is prohibited.",
        constraint_type="directional_prohibition",
        mw_limit=None,
        prohibited_direction="upward",
        prohibited_unit=None,
    )
    result = validate_action("Increase output at HUMR1G by 150 MW", [constraint])
    assert not result.passed


def test_downward_prohibition_catches_downward_action() -> None:
    """Action containing 'downward' violates a downward prohibition."""
    constraint = ExtractedConstraint(
        source_doc_id="doc-1",
        source_text="Downward dispatch is prohibited during voltage support.",
        constraint_type="directional_prohibition",
        mw_limit=None,
        prohibited_direction="downward",
        prohibited_unit=None,
    )
    result = validate_action(
        "Issue downward BM instruction to STTG1G for 100 MW",
        [constraint],
    )
    assert not result.passed


def test_upward_prohibition_passes_for_downward_action() -> None:
    """Action with 'downward' does not violate an upward prohibition."""
    constraint = ExtractedConstraint(
        source_doc_id="doc-1",
        source_text="Upward dispatch is not permitted.",
        constraint_type="directional_prohibition",
        mw_limit=None,
        prohibited_direction="upward",
        prohibited_unit=None,
    )
    result = validate_action(
        "Issue downward BM instruction to DRAX1G for 200 MW",
        [constraint],
    )
    assert result.passed


# ---------------------------------------------------------------------------
# validate_action — unit prohibition violations
# ---------------------------------------------------------------------------


def test_unit_prohibition_catches_named_unit() -> None:
    """Action naming DRAX1G violates a DRAX1G unit prohibition."""
    constraint = ExtractedConstraint(
        source_doc_id="doc-1",
        source_text="BM Unit DRAX1G must not be dispatched.",
        constraint_type="unit_prohibition",
        mw_limit=None,
        prohibited_direction=None,
        prohibited_unit="DRAX1G",
    )
    result = validate_action(
        "Issue upward BM dispatch instruction to DRAX1G for 200 MW",
        [constraint],
    )
    assert not result.passed
    assert "DRAX1G" in result.violations[0].reason


def test_unit_prohibition_passes_for_different_unit() -> None:
    """Action naming TORN1G does not violate a DRAX1G prohibition."""
    constraint = ExtractedConstraint(
        source_doc_id="doc-1",
        source_text="BM Unit DRAX1G must not be dispatched.",
        constraint_type="unit_prohibition",
        mw_limit=None,
        prohibited_direction=None,
        prohibited_unit="DRAX1G",
    )
    result = validate_action(
        "Issue upward BM dispatch instruction to TORN1G for 200 MW",
        [constraint],
    )
    assert result.passed


# ---------------------------------------------------------------------------
# validate_action — multiple constraints, multiple violations
# ---------------------------------------------------------------------------


def test_multiple_violations_all_reported() -> None:
    """Two violations from different constraint types are both returned."""
    constraints = [
        ExtractedConstraint(
            source_doc_id="doc-1",
            source_text="Output must not exceed 300 MW.",
            constraint_type="mw_ceiling",
            mw_limit=300.0,
            prohibited_direction=None,
            prohibited_unit=None,
        ),
        ExtractedConstraint(
            source_doc_id="doc-2",
            source_text="BM Unit DRAX1G must not be dispatched.",
            constraint_type="unit_prohibition",
            mw_limit=None,
            prohibited_direction=None,
            prohibited_unit="DRAX1G",
        ),
    ]
    result = validate_action(
        "Issue upward dispatch to DRAX1G for 500 MW",
        constraints,
    )
    assert not result.passed
    assert len(result.violations) == 2


def test_no_constraints_always_passes() -> None:
    """With an empty constraint list, any action passes."""
    result = validate_action("Issue upward dispatch to DRAX1G for 9999 MW", [])
    assert result.passed
    assert result.violations == ()


# ---------------------------------------------------------------------------
# check_constraints — end-to-end convenience wrapper
# ---------------------------------------------------------------------------


def test_check_constraints_detects_ceiling_violation() -> None:
    """End-to-end: regulatory doc + action with MW overage → violation."""
    docs = [_doc(_MW_CEILING_DOC_CONTENT, doc_id="reg-001")]
    action = "Issue upward BM dispatch to DRAX1G for 800 MW in settlement period 30"
    result = check_constraints(action, docs)
    assert not result.passed
    assert any("800" in v.reason or "500" in v.reason for v in result.violations)


def test_check_constraints_detects_directional_violation() -> None:
    """End-to-end: upward-ban doc + upward action → violation."""
    docs = [_doc(_DIRECTIONAL_UP_DOC_CONTENT, doc_id="reg-002")]
    action = "Issue upward BM dispatch instruction to HUMR1G for 200 MW"
    result = check_constraints(action, docs)
    assert not result.passed
    assert any("upward" in v.reason for v in result.violations)


def test_check_constraints_detects_unit_violation() -> None:
    """End-to-end: unit-ban doc + action naming that unit → violation."""
    docs = [_doc(_UNIT_PROHIBITION_DOC_CONTENT, doc_id="reg-003")]
    action = "Issue upward dispatch to DRAX1G for 200 MW"
    result = check_constraints(action, docs)
    assert not result.passed
    assert any("DRAX1G" in v.reason for v in result.violations)


def test_check_constraints_passes_on_clean_docs() -> None:
    """Clean regulatory doc + compliant action → passed=True, no violations."""
    docs = [_doc(_CLEAN_DOC_CONTENT, doc_id="reg-004")]
    action = "Issue upward BM dispatch to TORN1G for 200 MW"
    result = check_constraints(action, docs)
    assert result.passed
    assert result.violations == ()


# ---------------------------------------------------------------------------
# ValidationResult immutability
# ---------------------------------------------------------------------------


def test_validation_result_is_frozen() -> None:
    result = ValidationResult(action="test", violations=(), passed=True)
    with pytest.raises((AttributeError, TypeError)):
        result.passed = False  # type: ignore[misc]


def test_extracted_constraint_is_frozen() -> None:
    c = ExtractedConstraint(
        source_doc_id="doc-1",
        source_text="must not exceed 500 MW",
        constraint_type="mw_ceiling",
        mw_limit=500.0,
        prohibited_direction=None,
        prohibited_unit=None,
    )
    with pytest.raises((AttributeError, TypeError)):
        c.mw_limit = 999.0  # type: ignore[misc]

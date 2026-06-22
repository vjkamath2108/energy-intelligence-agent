"""Regulatory constraint extraction and validation for EIA dispatch actions.

Parses retrieved regulation documents to extract operational constraints, then
checks a proposed dispatch action against each constraint before it is surfaced
to an operator. Violations flag the recommendation for human review.

Three constraint types are extracted and validated:

  mw_ceiling         — "must not exceed 500 MW" / "maximum 500 MW"
                       Violation: action dispatches more than the ceiling.

  directional_prohibition — "upward dispatch not permitted"
                             Violation: action is in the prohibited direction.

  unit_prohibition   — "BM unit DRAX1G must not be dispatched"
                       Violation: action names the prohibited unit.

Design decisions:
- No LLM for extraction — classical regex per CLAUDE.md architecture rules.
  ("Classical ML does structured work; LLM does explanation and NL query only.")
- Sentence-level extraction: each sentence in a document is tested independently
  to limit false positives from multi-clause sentences.
- Conservative matching: prefer false negatives over false positives for the MW
  ceiling check (the action must contain an explicit MW value that exceeds the
  limit — ambiguous actions do not trigger a violation).
- All data classes frozen=True — immutable after creation.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Literal

from src.models import RetrievedDocument

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns for constraint extraction from regulatory text
# ---------------------------------------------------------------------------

# MW ceiling: "must not exceed 500 MW", "limited to 500 MW", "maximum 500 MW"
_MW_CEILING_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"(?:must not|shall not|cannot|may not)\s+exceed\s+(\d+(?:\.\d+)?)\s*MW",
        re.IGNORECASE,
    ),
    re.compile(
        r"limit(?:ed)?\s+(?:to|of)\s+(\d+(?:\.\d+)?)\s*MW",
        re.IGNORECASE,
    ),
    re.compile(
        r"a\s+limit\s+of\s+(\d+(?:\.\d+)?)\s*MW",
        re.IGNORECASE,
    ),
    re.compile(
        r"maximum\s+(?:active\s+power\s+)?(?:output\s+)?(?:of\s+)?(\d+(?:\.\d+)?)\s*MW",
        re.IGNORECASE,
    ),
    re.compile(
        r"ceiling\s+of\s+(\d+(?:\.\d+)?)\s*MW",
        re.IGNORECASE,
    ),
]

# Directional prohibition: "upward dispatch not permitted", "no downward balancing"
_DIRECTIONAL_PROHIBITION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"(upward|downward)\s+(?:dispatch\s+instructions?|dispatch|balancing\s+services?|BM\s+instructions?)"
        r"\s+(?:is\s+|are\s+)?(?:not\s+permitted|prohibited|not\s+allowed|not\s+authorised)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:not\s+permitted|prohibited|not\s+allowed|not\s+authorised)"
        r".*?\b(upward|downward)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"no\s+(upward|downward)\s+(?:dispatch|balancing)",
        re.IGNORECASE,
    ),
]

# Unit prohibition: "DRAX1G must not be dispatched", "BM unit RATS1G shall not participate"
# BM unit IDs: 3-6 uppercase letters optionally followed by a digit and/or letter (e.g. DRAX1G)
_BM_UNIT_ID = r"\b([A-Z]{2,6}[0-9][A-Z]?)\b"
_UNIT_PROHIBITION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"BM\s+[Uu]nit\s+"
        + _BM_UNIT_ID
        + r"\s+(?:must\s+not|shall\s+not|cannot|may\s+not)"
        r"\s+(?:be\s+dispatched|participate|receive\s+dispatch)",
        re.IGNORECASE,
    ),
    re.compile(
        _BM_UNIT_ID + r"\s+(?:must\s+not|shall\s+not|cannot|is\s+not\s+permitted\s+to)"
        r"\s+(?:be\s+dispatched|generate|provide\s+balancing)",
        re.IGNORECASE,
    ),
    re.compile(
        r"dispatch(?:ing)?\s+(?:to|from)\s+" + _BM_UNIT_ID + r"\s+is\s+prohibited",
        re.IGNORECASE,
    ),
]

# MW value extraction from action text
_MW_VALUE = re.compile(r"(\d+(?:\.\d+)?)\s*MW", re.IGNORECASE)

# Direction signals in action text
_UPWARD_ACTION = re.compile(
    r"\b(?:upward|ramp(?:\s+|-)?up|increase|instructed\s+up)\b", re.IGNORECASE
)
_DOWNWARD_ACTION = re.compile(
    r"\b(?:downward|ramp(?:\s+|-)?down|decrease|reduce|cut\s+back|instructed\s+down)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

ConstraintType = Literal["mw_ceiling", "directional_prohibition", "unit_prohibition"]


@dataclass(frozen=True)
class ExtractedConstraint:
    """A single operational constraint parsed from a regulatory document.

    Args:
        source_doc_id: doc_id of the RetrievedDocument this was parsed from.
        source_text: The sentence fragment that contained the constraint.
        constraint_type: Semantic category of the constraint.
        mw_limit: Ceiling value in MW for mw_ceiling constraints; None otherwise.
        prohibited_direction: 'upward' or 'downward' for directional_prohibition; None otherwise.
        prohibited_unit: BM unit identifier for unit_prohibition; None otherwise.
    """

    source_doc_id: str
    source_text: str
    constraint_type: ConstraintType
    mw_limit: float | None
    prohibited_direction: Literal["upward", "downward"] | None
    prohibited_unit: str | None


@dataclass(frozen=True)
class ConstraintViolation:
    """A specific constraint that the proposed action violates.

    Args:
        constraint: The violated ExtractedConstraint.
        reason: Human-readable explanation of why the action violates the constraint.
    """

    constraint: ExtractedConstraint
    reason: str


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of validating a dispatch action against all extracted constraints.

    Args:
        action: The dispatch action string that was validated.
        violations: Tuple of all constraint violations found; empty when clean.
        passed: True when no violations were found.
    """

    action: str
    violations: tuple[ConstraintViolation, ...]
    passed: bool


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def _sentences(text: str) -> list[str]:
    """Split text into sentences on '.', '!', '?' boundaries.

    Sentence-level splitting limits false positives from multi-clause sentences
    where a constraint applies to a different subject than it appears to.

    Args:
        text: Regulatory document text.

    Returns:
        List of non-empty stripped sentence strings.
    """
    return [s.strip() for s in re.split(r"[.!?]", text) if s.strip()]


def _extract_mw_ceilings(sentence: str, doc_id: str) -> list[ExtractedConstraint]:
    """Extract all MW ceiling constraints from a single sentence.

    Args:
        sentence: One sentence from a regulatory document.
        doc_id: Source document identifier.

    Returns:
        List of mw_ceiling ExtractedConstraints; empty when none match.
    """
    results: list[ExtractedConstraint] = []
    for pattern in _MW_CEILING_PATTERNS:
        for match in pattern.finditer(sentence):
            try:
                mw = float(match.group(1))
            except (ValueError, IndexError):
                continue
            results.append(
                ExtractedConstraint(
                    source_doc_id=doc_id,
                    source_text=sentence,
                    constraint_type="mw_ceiling",
                    mw_limit=mw,
                    prohibited_direction=None,
                    prohibited_unit=None,
                )
            )
    return results


def _extract_directional_prohibitions(
    sentence: str, doc_id: str
) -> list[ExtractedConstraint]:
    """Extract directional dispatch prohibitions from a single sentence.

    Args:
        sentence: One sentence from a regulatory document.
        doc_id: Source document identifier.

    Returns:
        List of directional_prohibition ExtractedConstraints.
    """
    results: list[ExtractedConstraint] = []
    for pattern in _DIRECTIONAL_PROHIBITION_PATTERNS:
        for match in pattern.finditer(sentence):
            direction_raw = match.group(1).lower()
            if direction_raw not in ("upward", "downward"):
                continue
            direction: Literal["upward", "downward"] = (
                "upward" if direction_raw == "upward" else "downward"
            )
            results.append(
                ExtractedConstraint(
                    source_doc_id=doc_id,
                    source_text=sentence,
                    constraint_type="directional_prohibition",
                    mw_limit=None,
                    prohibited_direction=direction,
                    prohibited_unit=None,
                )
            )
    return results


def _extract_unit_prohibitions(sentence: str, doc_id: str) -> list[ExtractedConstraint]:
    """Extract BM unit dispatch prohibitions from a single sentence.

    Args:
        sentence: One sentence from a regulatory document.
        doc_id: Source document identifier.

    Returns:
        List of unit_prohibition ExtractedConstraints.
    """
    results: list[ExtractedConstraint] = []
    for pattern in _UNIT_PROHIBITION_PATTERNS:
        for match in pattern.finditer(sentence):
            try:
                unit = match.group(1).upper()
            except IndexError:
                continue
            results.append(
                ExtractedConstraint(
                    source_doc_id=doc_id,
                    source_text=sentence,
                    constraint_type="unit_prohibition",
                    mw_limit=None,
                    prohibited_direction=None,
                    prohibited_unit=unit,
                )
            )
    return results


def extract_constraints(docs: list[RetrievedDocument]) -> list[ExtractedConstraint]:
    """Parse all retrieved documents and return every identified constraint.

    Runs sentence-by-sentence to limit false positives from multi-clause text.
    Duplicate constraints (same type and value from different sentences in the
    same document) are included — deduplication is the caller's concern.

    Args:
        docs: Retrieved document chunks from the hybrid retriever.

    Returns:
        All ExtractedConstraints found across all documents, in document order.
    """
    all_constraints: list[ExtractedConstraint] = []
    for doc in docs:
        for sentence in _sentences(doc.content):
            all_constraints.extend(_extract_mw_ceilings(sentence, doc.doc_id))
            all_constraints.extend(
                _extract_directional_prohibitions(sentence, doc.doc_id)
            )
            all_constraints.extend(_extract_unit_prohibitions(sentence, doc.doc_id))

    if all_constraints:
        logger.debug(
            "Extracted %d constraints from %d documents",
            len(all_constraints),
            len(docs),
        )
    return all_constraints


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _action_mw_values(action: str) -> list[float]:
    """Extract all MW quantity values mentioned in the action string.

    Args:
        action: Proposed dispatch action text.

    Returns:
        List of MW float values; empty when no explicit MW values appear.
    """
    return [float(m.group(1)) for m in _MW_VALUE.finditer(action)]


def _check_mw_ceiling(
    action: str, constraint: ExtractedConstraint
) -> ConstraintViolation | None:
    """Return a violation if the action exceeds the MW ceiling, else None.

    Conservative: only flags when the action contains an explicit MW value
    that exceeds the ceiling. Ambiguous actions without MW values pass.

    Args:
        action: Proposed dispatch action text.
        constraint: An mw_ceiling ExtractedConstraint to check against.

    Returns:
        ConstraintViolation if violated, None if the action is compliant or ambiguous.
    """
    if constraint.mw_limit is None:
        return None
    mw_values = _action_mw_values(action)
    if not mw_values:
        return None
    action_mw = max(mw_values)
    if action_mw > constraint.mw_limit:
        return ConstraintViolation(
            constraint=constraint,
            reason=(
                f"Action dispatches {action_mw:.0f} MW which exceeds the regulatory "
                f"ceiling of {constraint.mw_limit:.0f} MW "
                f"(source: {constraint.source_doc_id})"
            ),
        )
    return None


def _check_directional_prohibition(
    action: str, constraint: ExtractedConstraint
) -> ConstraintViolation | None:
    """Return a violation if the action is in the prohibited dispatch direction, else None.

    Args:
        action: Proposed dispatch action text.
        constraint: A directional_prohibition ExtractedConstraint to check against.

    Returns:
        ConstraintViolation if the action direction matches the prohibition, else None.
    """
    direction = constraint.prohibited_direction
    if direction is None:
        return None

    pattern = _UPWARD_ACTION if direction == "upward" else _DOWNWARD_ACTION
    if pattern.search(action):
        return ConstraintViolation(
            constraint=constraint,
            reason=(
                f"Action involves {direction} dispatch which is prohibited "
                f"by the retrieved regulation (source: {constraint.source_doc_id})"
            ),
        )
    return None


def _check_unit_prohibition(
    action: str, constraint: ExtractedConstraint
) -> ConstraintViolation | None:
    """Return a violation if the action names a prohibited BM unit, else None.

    Args:
        action: Proposed dispatch action text.
        constraint: A unit_prohibition ExtractedConstraint to check against.

    Returns:
        ConstraintViolation if the prohibited unit appears in the action, else None.
    """
    unit = constraint.prohibited_unit
    if unit is None:
        return None
    pattern = re.compile(rf"\b{re.escape(unit)}\b", re.IGNORECASE)
    if pattern.search(action):
        return ConstraintViolation(
            constraint=constraint,
            reason=(
                f"Action involves BM unit {unit} which is prohibited "
                f"from dispatch (source: {constraint.source_doc_id})"
            ),
        )
    return None


_CHECKERS = {
    "mw_ceiling": _check_mw_ceiling,
    "directional_prohibition": _check_directional_prohibition,
    "unit_prohibition": _check_unit_prohibition,
}


def validate_action(
    action: str, constraints: list[ExtractedConstraint]
) -> ValidationResult:
    """Check the proposed action against every extracted constraint.

    Args:
        action: The dispatch action string from the LLM recommendation.
        constraints: All constraints extracted from the retrieved documents.

    Returns:
        ValidationResult with all violations found; passed=True when clean.
    """
    violations: list[ConstraintViolation] = []

    for constraint in constraints:
        checker = _CHECKERS.get(constraint.constraint_type)
        if checker is None:
            continue
        violation = checker(action, constraint)
        if violation is not None:
            violations.append(violation)

    if violations:
        logger.warning(
            "Constraint violations found (%d): %s",
            len(violations),
            "; ".join(v.reason for v in violations),
        )

    return ValidationResult(
        action=action,
        violations=tuple(violations),
        passed=not violations,
    )


def check_constraints(action: str, docs: list[RetrievedDocument]) -> ValidationResult:
    """Extract constraints from docs then validate the action in one call.

    Convenience wrapper for the common pipeline case. Equivalent to:
        validate_action(action, extract_constraints(docs))

    Args:
        action: The dispatch action string from the LLM recommendation.
        docs: Retrieved document chunks from the hybrid retriever.

    Returns:
        ValidationResult with all violations found; passed=True when clean.
    """
    constraints = extract_constraints(docs)
    return validate_action(action, constraints)

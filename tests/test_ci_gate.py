"""Tests for evaluation/ci_gate.py.

Verifies that:
  - The v1 prompt mock passes both gates.
  - The degraded prompt mock fails both gates.
  - The golden eval set itself is structurally valid.
"""

from __future__ import annotations

import pytest

from evaluation.ci_gate import (
    CITATION_COVERAGE_GATE,
    FAITHFULNESS_GATE,
    GOLDEN_EVAL_SET,
    run_eval,
)


# ---------------------------------------------------------------------------
# Golden eval set structure
# ---------------------------------------------------------------------------


def test_golden_eval_set_non_empty() -> None:
    """At least one eval case is defined."""
    assert len(GOLDEN_EVAL_SET) >= 1


def test_golden_eval_cases_have_context_docs() -> None:
    """Every eval case has at least one context document."""
    for case in GOLDEN_EVAL_SET:
        assert len(case.context_docs) >= 1, f"{case.name} has no context docs"


def test_golden_eval_cases_have_good_and_bad_responses() -> None:
    """Every eval case has non-empty good and bad action strings."""
    for case in GOLDEN_EVAL_SET:
        assert case.good.action, f"{case.name} good.action is empty"
        assert case.bad.action, f"{case.name} bad.action is empty"


def test_good_responses_cite_docs() -> None:
    """Good responses cite at least one document."""
    for case in GOLDEN_EVAL_SET:
        assert len(case.good.cited_doc_ids) >= 1, (
            f"{case.name} good response cites no documents"
        )


def test_bad_responses_cite_no_docs() -> None:
    """Bad (degraded) responses cite no documents."""
    for case in GOLDEN_EVAL_SET:
        assert len(case.bad.cited_doc_ids) == 0, (
            f"{case.name} bad response should cite no documents"
        )


# ---------------------------------------------------------------------------
# Gate outcomes
# ---------------------------------------------------------------------------


async def test_v1_prompt_passes_both_gates() -> None:
    """v1 prompt mock meets faithfulness >= 0.80 and citation coverage >= 0.90."""
    faithfulness, citation_coverage, _ = await run_eval("v1", mock=True)
    assert faithfulness >= FAITHFULNESS_GATE, (
        f"v1 faithfulness {faithfulness:.0%} < gate {FAITHFULNESS_GATE:.0%}"
    )
    assert citation_coverage >= CITATION_COVERAGE_GATE, (
        f"v1 citation coverage {citation_coverage:.0%} < gate {CITATION_COVERAGE_GATE:.0%}"
    )


async def test_degraded_prompt_fails_faithfulness_gate() -> None:
    """Degraded prompt mock fails faithfulness < 0.80 — proving the gate catches bad prompts."""
    faithfulness, _, _ = await run_eval("degraded", mock=True)
    assert faithfulness < FAITHFULNESS_GATE, (
        f"Degraded prompt should fail faithfulness gate but got {faithfulness:.0%}"
    )


async def test_degraded_prompt_fails_citation_gate() -> None:
    """Degraded prompt mock fails citation coverage < 0.90."""
    _, citation_coverage, _ = await run_eval("degraded", mock=True)
    assert citation_coverage < CITATION_COVERAGE_GATE, (
        f"Degraded prompt should fail citation gate but got {citation_coverage:.0%}"
    )


async def test_v1_per_case_all_pass() -> None:
    """Every individual case is grounded with the v1 prompt."""
    _, _, per_case = await run_eval("v1", mock=True)
    for name, is_faithful, _ in per_case:
        assert is_faithful, f"Case {name!r} failed grounding with v1 prompt"


async def test_degraded_per_case_all_fail() -> None:
    """Every individual case fails grounding with the degraded prompt."""
    _, _, per_case = await run_eval("degraded", mock=True)
    for name, is_faithful, _ in per_case:
        assert not is_faithful, f"Case {name!r} passed grounding with degraded prompt — bad responses should fail"

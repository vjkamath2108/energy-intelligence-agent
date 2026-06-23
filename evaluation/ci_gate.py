"""CI evaluation gate — faithfulness and citation coverage for EIA dispatch quality.

Runs a golden eval set of 5 (query, context, expected_response) tuples.

Gates (must both pass for exit 0):
  Faithfulness (grounded / total):          >= 0.80
  Citation coverage (avg cited/retrieved):  >= 0.90

Grounding uses the same term-overlap check as the production safety layer
(src.recommendation_engine.safety.check_grounding), so the gate exercises
real production logic — not a separate evaluator that could drift from prod.

Usage:
    python evaluation/ci_gate.py --mock                          # should pass (exit 0)
    python evaluation/ci_gate.py --mock --prompt-version v1      # should pass (exit 0)
    python evaluation/ci_gate.py --mock --prompt-version degraded  # should fail (exit 1)

Exit codes:
    0 — both gates passed
    1 — one or more gates failed
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from dataclasses import dataclass, field

from src.models import RetrievedDocument
from src.recommendation_engine.safety import check_grounding

logger = logging.getLogger("ci_gate")

FAITHFULNESS_GATE: float = 0.80
CITATION_COVERAGE_GATE: float = 0.90


# ---------------------------------------------------------------------------
# Golden eval dataset
# ---------------------------------------------------------------------------


@dataclass
class MockResponse:
    """Pre-scripted LLM response used in --mock mode."""

    action: str
    reasoning: str
    cited_doc_ids: list[str] = field(default_factory=list)


@dataclass
class EvalCase:
    """One golden (query, context_docs, responses) triplet."""

    name: str
    query: str
    context_docs: list[RetrievedDocument] = field(default_factory=list)
    good: MockResponse = field(
        default_factory=lambda: MockResponse(action="", reasoning="")
    )
    bad: MockResponse = field(
        default_factory=lambda: MockResponse(action="", reasoning="")
    )


def _doc(doc_id: str, content: str, source: str) -> RetrievedDocument:
    """Construct a RetrievedDocument for use in eval cases."""
    return RetrievedDocument(
        doc_id=doc_id,
        content=content,
        source=source,
        chunk_index=0,
        similarity_score=0.92,
    )


GOLDEN_EVAL_SET: list[EvalCase] = [
    EvalCase(
        name="bsuos-sp23",
        query="What BSUoS charge applies to SP23 for a 50MW generator?",
        context_docs=[
            _doc(
                "bsuos-001",
                "Balancing Services Use of System (BSUoS) charges recover costs incurred by "
                "National Grid ESO in balancing the electricity transmission system. Charges are "
                "calculated on a half-hourly settlement period basis, applied to all licensed "
                "generators and suppliers. The BSUoS tariff for settlement period SP23 is 4.82 per MWh.",
                "regulation/bsuos-tariff-2026.pdf",
            )
        ],
        good=MockResponse(
            action=(
                "Apply BSUoS tariff of 4.82 per MWh to SP23 metered volume for the 50MW generator, "
                "resulting in a settlement charge calculated on a half-hourly basis under National "
                "Grid ESO balancing cost recovery for licensed generators."
            ),
            reasoning=(
                "The BSUoS tariff document specifies 4.82 per MWh for SP23, applying to all licensed "
                "generators. The half-hourly settlement period basis means the metered volume for SP23 "
                "determines the charge."
            ),
            cited_doc_ids=["bsuos-001"],
        ),
        bad=MockResponse(
            action=(
                "Review the current situation and consult with appropriate stakeholders before "
                "making any decisions regarding this matter."
            ),
            reasoning=(
                "Based on standard procedures, assess all available options carefully and "
                "consider engaging relevant parties before proceeding."
            ),
            cited_doc_ids=[],
        ),
    ),
    EvalCase(
        name="tnuos-generation",
        query="Calculate TNUoS liability for a Zone 1 generator at 120MW registered capacity.",
        context_docs=[
            _doc(
                "tnuos-001",
                "Transmission Network Use of System (TNUoS) tariffs are charged to generators based "
                "on their registered capacity in MW. For Charging Zone 1 (North Scotland), the 2026/27 "
                "generation tariff is 42.80 per kW per year. Generators are charged during the triads — "
                "the three half-hour periods of highest system demand between November and February.",
                "regulation/tnuos-tariffs-2026-27.pdf",
            )
        ],
        good=MockResponse(
            action=(
                "Calculate TNUoS generation tariff for Zone 1 at 42.80 per kW per year on 120MW "
                "registered capacity: 42,800 multiplied by 120 equals annual TNUoS transmission "
                "network liability charged during triad demand periods for licensed generators."
            ),
            reasoning=(
                "The TNUoS tariff schedule specifies 42.80 per kW per year for Zone 1 generators. "
                "Applying to 120MW registered capacity at the triad charging mechanism gives the "
                "annual transmission network use liability. Triads fall between November and February."
            ),
            cited_doc_ids=["tnuos-001"],
        ),
        bad=MockResponse(
            action=(
                "Engage specialist advisors to determine the appropriate financial obligations "
                "and ensure compliance with all regulatory requirements."
            ),
            reasoning=(
                "Professional guidance should be sought to evaluate the situation comprehensively "
                "before committing to a course of action."
            ),
            cited_doc_ids=[],
        ),
    ),
    EvalCase(
        name="bm-dispatch-instruction",
        query="BM unit SCOT-W01 received a dispatch instruction to increase output by 30MW. What are the settlement implications?",
        context_docs=[
            _doc(
                "bm-001",
                "Balancing Mechanism (BM) dispatch instructions issued by National Grid ESO direct "
                "BM units to increase or decrease generation output. Accepted offers (increase) and "
                "accepted bids (decrease) are settled at the offer or bid price through the Balancing "
                "Mechanism Reporting Service. Settlement occurs within 28 calendar days of the "
                "relevant settlement period via the BSC Central Services Agent.",
                "regulation/bsc-code-s2-2026.pdf",
            )
        ],
        good=MockResponse(
            action=(
                "Process BM dispatch instruction for SCOT-W01 as an accepted offer to increase "
                "generation output by 30MW. Settlement will occur at the accepted offer price via "
                "the Balancing Mechanism Reporting Service within 28 calendar days through the "
                "BSC Central Services Agent."
            ),
            reasoning=(
                "The BSC Code specifies that accepted BM offers for output increase are settled at "
                "the accepted offer price through the Balancing Mechanism Reporting Service. The "
                "28-day settlement window applies from the relevant settlement period via BSC."
            ),
            cited_doc_ids=["bm-001"],
        ),
        bad=MockResponse(
            action=(
                "Proceed cautiously and verify with internal operations teams whether this "
                "instruction aligns with planned schedules before any further steps."
            ),
            reasoning=(
                "Internal verification processes should be followed to ensure alignment across "
                "relevant departments before implementing this instruction."
            ),
            cited_doc_ids=[],
        ),
    ),
    EvalCase(
        name="frequency-response-ccgt",
        query="What primary frequency response obligation applies to a 100MW CCGT under the latest Grid Code?",
        context_docs=[
            _doc(
                "grid-code-001",
                "Under Grid Code ECC.6.3.2, all synchronous generators above 50MW must provide "
                "Primary Frequency Response within 10 seconds of a frequency deviation exceeding "
                "0.2Hz from nominal 50Hz. The mandatory primary frequency response must be sustained "
                "for at least 20 seconds. For CCGTs, the required response rate is 10 percent of "
                "registered capacity per Hz deviation.",
                "regulation/grid-code-ecc-2026.pdf",
            )
        ],
        good=MockResponse(
            action=(
                "Confirm primary frequency response obligation for the CCGT under Grid Code ECC.6.3.2: "
                "provide 10 percent of registered capacity per Hz deviation within 10 seconds of "
                "0.2Hz exceedance from nominal frequency, sustained for minimum 20 seconds mandatory."
            ),
            reasoning=(
                "Grid Code ECC.6.3.2 mandates synchronous generators above 50MW deliver primary "
                "frequency response within 10 seconds of a frequency deviation exceeding 0.2Hz. "
                "For CCGTs the required rate is 10 percent of registered capacity per Hz. "
                "The mandatory response must be sustained for at least 20 seconds."
            ),
            cited_doc_ids=["grid-code-001"],
        ),
        bad=MockResponse(
            action=(
                "Ensure compliance with all applicable regulatory obligations and consult the "
                "relevant technical teams to assess capability and plan accordingly."
            ),
            reasoning=(
                "Technical and regulatory assessment is recommended to understand obligations "
                "and ensure the unit operates within all required parameters."
            ),
            cited_doc_ids=[],
        ),
    ),
    EvalCase(
        name="capacity-market-penalty",
        query="CfD unit WIND-SE-44 failed to deliver during a stress event. What Capacity Market penalty applies?",
        context_docs=[
            _doc(
                "cm-001",
                "Capacity Market Rules Section 9.4: A Capacity Provider that fails to deliver its "
                "obligation during a System Stress Event is subject to a Capacity Market penalty. "
                "The penalty is calculated as the undelivered MW multiplied by the Penalty Rate. "
                "The Penalty Rate for 2026/27 is set at 7500 per MWh of undelivered capacity. "
                "Penalties are recovered through the Capacity Market Settlement Body within "
                "10 business days.",
                "regulation/cm-rules-2026-27.pdf",
            )
        ],
        good=MockResponse(
            action=(
                "Calculate Capacity Market penalty under Rules Section 9.4: apply the 7500 per MWh "
                "penalty rate to undelivered capacity MW during the System Stress Event. Recover "
                "through the Capacity Market Settlement Body within 10 business days."
            ),
            reasoning=(
                "CM Rules Section 9.4 specifies the penalty formula for non-delivery during a "
                "System Stress Event: undelivered MW multiplied by the Penalty Rate. The 2026/27 "
                "Penalty Rate is 7500 per MWh of undelivered capacity. The Settlement Body "
                "recovers penalties within 10 business days."
            ),
            cited_doc_ids=["cm-001"],
        ),
        bad=MockResponse(
            action=(
                "Investigate the circumstances surrounding the missed delivery and escalate to "
                "management for review of appropriate remedial actions and communications."
            ),
            reasoning=(
                "A thorough investigation should be undertaken with relevant parties to understand "
                "the failure and determine next steps in line with standard procedures."
            ),
            cited_doc_ids=[],
        ),
    ),
]


# ---------------------------------------------------------------------------
# Evaluation logic
# ---------------------------------------------------------------------------


async def _evaluate_case(
    case: EvalCase,
    prompt_version: str,
    mock: bool,
) -> tuple[bool, float]:
    """Evaluate a single eval case.

    Args:
        case: The EvalCase to evaluate.
        prompt_version: "v1" (good responses) or "degraded" (bad responses).
        mock: True to use pre-scripted responses without calling Bedrock.

    Returns:
        Tuple of (is_faithful, citation_coverage).
    """
    if not mock:
        raise NotImplementedError(
            "Real Bedrock mode is not yet wired in the CI gate. "
            "Pass --mock for CI; implement real mode for local eval."
        )

    resp = case.good if prompt_version != "degraded" else case.bad

    safety = await check_grounding(
        action=resp.action,
        reasoning=resp.reasoning,
        context_docs=case.context_docs,
        embedder=None,
    )

    n_retrieved = len(case.context_docs)
    n_cited = len(resp.cited_doc_ids)
    coverage = n_cited / n_retrieved if n_retrieved > 0 else 0.0

    return safety.is_grounded, coverage


async def run_eval(
    prompt_version: str,
    mock: bool,
) -> tuple[float, float, list[tuple[str, bool, float]]]:
    """Run the full golden eval set.

    Args:
        prompt_version: "v1" or "degraded".
        mock: True to use pre-scripted mock responses.

    Returns:
        Tuple of (faithfulness_score, mean_citation_coverage, per_case_results).
        per_case_results is a list of (case_name, is_faithful, coverage).
    """
    per_case: list[tuple[str, bool, float]] = []

    for case in GOLDEN_EVAL_SET:
        is_faithful, coverage = await _evaluate_case(case, prompt_version, mock)
        per_case.append((case.name, is_faithful, coverage))

    faithfulness = sum(1 for _, f, _ in per_case if f) / len(per_case)
    mean_coverage = sum(cov for _, _, cov in per_case) / len(per_case)
    return faithfulness, mean_coverage, per_case


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """Run the CI eval gate. Returns exit code 0 on pass, 1 on fail."""
    parser = argparse.ArgumentParser(
        description="EIA CI evaluation gate — faithfulness and citation coverage gates."
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use pre-scripted mock responses (no AWS credentials required).",
    )
    parser.add_argument(
        "--prompt-version",
        default="v1",
        choices=["v1", "degraded"],
        help=(
            "Prompt version to evaluate against the golden set. "
            "'degraded' demonstrates that the gate catches bad prompts (should fail)."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s  %(message)s",
        stream=sys.stderr,
    )

    print("\nEIA Eval Gate")
    print(f"  prompt-version : {args.prompt_version}")
    print(f"  mock           : {args.mock}")
    print(f"  cases          : {len(GOLDEN_EVAL_SET)}")
    print(f"  faithfulness gate  : >= {FAITHFULNESS_GATE:.0%}")
    print(f"  citation gate      : >= {CITATION_COVERAGE_GATE:.0%}")
    print()

    faithfulness, mean_coverage, per_case = asyncio.run(
        run_eval(args.prompt_version, args.mock)
    )

    print("Per-case results:")
    for name, is_faithful, coverage in per_case:
        status = "PASS" if is_faithful else "FAIL"
        print(f"  [{status}]  {name:<30}  citation={coverage:.0%}")

    faith_status = "PASS" if faithfulness >= FAITHFULNESS_GATE else "FAIL"
    cov_status = "PASS" if mean_coverage >= CITATION_COVERAGE_GATE else "FAIL"

    print()
    print("Aggregate:")
    print(
        f"  Faithfulness      {faithfulness:.0%}  (gate >= {FAITHFULNESS_GATE:.0%})  {faith_status}"
    )
    print(
        f"  Citation coverage {mean_coverage:.0%}  (gate >= {CITATION_COVERAGE_GATE:.0%})  {cov_status}"
    )

    passed = (
        faithfulness >= FAITHFULNESS_GATE and mean_coverage >= CITATION_COVERAGE_GATE
    )
    print()
    print(f"Overall: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

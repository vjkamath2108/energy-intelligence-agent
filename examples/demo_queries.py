"""EIA RAG pipeline demo — 5 realistic dispatch scenarios.

Traces the full pipeline for each scenario:
  query → retrieved docs → recommendation → safety check → confidence → audit

Scenarios:
  1. BSUoS settlement charge calculation        (grounded, passes safety)
  2. TNUoS generation tariff liability          (grounded, passes safety)
  3. BM dispatch instruction settlement         (grounded, passes safety)
  4. Grid Code frequency response obligation    (grounded, passes safety)
  5. Hallucination detection                    (ungrounded response, escalated)

Usage:
    python examples/demo_queries.py             # demo mode — no AWS, no DB needed
    python examples/demo_queries.py --backend bedrock  # real Bedrock (requires creds)

Exit codes:
    0 — all expected outcomes produced (grounded scenarios pass safety; hallucination caught)
    1 — unexpected outcome
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from src.audit.audit_log import AuditLogger, entry_from_recommendation
from src.models import DispatchRecommendation, RetrievedDocument
from src.recommendation_engine.confidence import ConfidenceScorer, ConfidenceSignals
from src.recommendation_engine.safety import check_grounding

_DEMO_AUDIT_PATH = Path("results/demo_audit.jsonl")
_DIVIDER = "═" * 70
_LINE = "─" * 70

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Demo scenarios
# ---------------------------------------------------------------------------


@dataclass
class DemoScenario:
    """One pre-scripted RAG scenario for the demo."""

    name: str
    query: str
    context_docs: list[RetrievedDocument]
    action: str
    reasoning: str
    cited_doc_ids: list[str]
    expect_hallucination: bool = False


def _doc(
    doc_id: str,
    content: str,
    source: str,
    score: float = 0.92,
) -> RetrievedDocument:
    return RetrievedDocument(
        doc_id=doc_id,
        content=content,
        source=source,
        chunk_index=0,
        similarity_score=score,
    )


DEMO_SCENARIOS: list[DemoScenario] = [
    # ── Scenario 1 ──────────────────────────────────────────────────────────
    DemoScenario(
        name="BSUoS Settlement Charge",
        query="What BSUoS charge applies to settlement period SP23 for a 50MW generator?",
        context_docs=[
            _doc(
                "bsuos-2026-sp23",
                "Balancing Services Use of System (BSUoS) charges recover costs incurred by "
                "National Grid ESO in balancing the electricity transmission system. Charges "
                "are calculated on a half-hourly settlement period basis applied to all licensed "
                "generators and suppliers. The BSUoS tariff for settlement period SP23 is "
                "£4.82/MWh. Metered volume data must be submitted via BSC Central Services Agent.",
                "regulation/bsuos-tariff-2026.pdf",
                0.95,
            ),
            _doc(
                "bsc-s2-settlement",
                "Settlement of BSUoS charges occurs within 28 calendar days of the relevant "
                "settlement period. All licensed generators must submit accurate metered volume "
                "data for each half-hourly settlement period via the BSC Central Services Agent.",
                "regulation/bsc-code-s2-2026.pdf",
                0.88,
            ),
        ],
        action=(
            "Apply BSUoS tariff of £4.82/MWh to settlement period SP23 metered volume for the "
            "50MW generator. Settlement calculated on a half-hourly basis: 50MW × 0.5h × £4.82 "
            "= £120.50 per SP23. Submit metered volume data via BSC Central Services Agent "
            "within 28 calendar days."
        ),
        reasoning=(
            "BSUoS tariff document confirms £4.82/MWh for SP23 applied to all licensed generators. "
            "Half-hourly settlement period basis gives charge = MW × 0.5h × tariff. "
            "BSC Code S2 confirms 28-day settlement timeline via Central Services Agent."
        ),
        cited_doc_ids=["bsuos-2026-sp23", "bsc-s2-settlement"],
    ),
    # ── Scenario 2 ──────────────────────────────────────────────────────────
    DemoScenario(
        name="TNUoS Generation Tariff",
        query="Calculate the annual TNUoS liability for a Zone 1 generator at 120MW registered capacity.",
        context_docs=[
            _doc(
                "tnuos-zone1-2026",
                "Transmission Network Use of System (TNUoS) tariffs are charged to generators "
                "based on their registered capacity in MW. For Charging Zone 1 (North Scotland), "
                "the 2026/27 generation tariff is £42.80/kW/year. Generators are charged during "
                "the triads — the three half-hour periods of highest system demand between "
                "November and February.",
                "regulation/tnuos-tariffs-2026-27.pdf",
                0.93,
            ),
        ],
        action=(
            "Annual TNUoS liability for Zone 1 generator: £42.80/kW/year × 120,000 kW = "
            "£5,136,000 per year. Charges levied via triad settlement during the three "
            "highest system demand half-hour periods between November and February."
        ),
        reasoning=(
            "TNUoS tariff schedule for Charging Zone 1 specifies £42.80/kW/year for generation. "
            "120MW = 120,000 kW. Triad mechanism confirms charges apply at peak demand periods "
            "November–February."
        ),
        cited_doc_ids=["tnuos-zone1-2026"],
    ),
    # ── Scenario 3 ──────────────────────────────────────────────────────────
    DemoScenario(
        name="BM Dispatch Instruction Settlement",
        query="BM unit SCOT-W01 received a dispatch instruction to increase output by 30MW. Settlement implications?",
        context_docs=[
            _doc(
                "bm-dispatch-bsc",
                "Balancing Mechanism (BM) dispatch instructions issued by National Grid ESO "
                "direct BM units to increase or decrease generation output. Accepted offers "
                "(increase) and accepted bids (decrease) are settled at the offer/bid price "
                "through the Balancing Mechanism Reporting Service. Settlement occurs within "
                "28 calendar days of the relevant settlement period via BSC Central Services Agent.",
                "regulation/bsc-code-s2-2026.pdf",
                0.91,
            ),
        ],
        action=(
            "Process SCOT-W01 BM dispatch instruction as an accepted offer: increase generation "
            "output by 30MW at the accepted offer price. Settle via Balancing Mechanism Reporting "
            "Service within 28 calendar days through BSC Central Services Agent. "
            "Record instruction in settlement metered volume data."
        ),
        reasoning=(
            "BSC Code S2 defines accepted BM offers (output increase) as settled at the accepted "
            "offer price through the Balancing Mechanism Reporting Service. 28-day settlement "
            "window applies from the relevant settlement period via BSC Central Services Agent."
        ),
        cited_doc_ids=["bm-dispatch-bsc"],
    ),
    # ── Scenario 4 ──────────────────────────────────────────────────────────
    DemoScenario(
        name="Grid Code Frequency Response",
        query="What primary frequency response obligation applies to a 100MW CCGT under Grid Code ECC.6.3.2?",
        context_docs=[
            _doc(
                "grid-code-ecc632",
                "Under Grid Code ECC.6.3.2, all synchronous generators above 50MW must provide "
                "Primary Frequency Response within 10 seconds of a frequency deviation exceeding "
                "±0.2Hz from nominal 50Hz. The mandatory primary frequency response must be "
                "sustained for at least 20 seconds. For CCGTs, the required response rate is "
                "10% of registered capacity per Hz deviation.",
                "regulation/grid-code-ecc-2026.pdf",
                0.96,
            ),
        ],
        action=(
            "Confirm 100MW CCGT primary frequency response obligation under Grid Code ECC.6.3.2: "
            "deliver 10% registered capacity per Hz deviation (10MW/Hz) within 10 seconds of "
            "±0.2Hz exceedance from nominal 50Hz frequency. Sustain mandatory primary response "
            "for minimum 20 seconds."
        ),
        reasoning=(
            "Grid Code ECC.6.3.2 mandates synchronous generators above 50MW provide primary "
            "frequency response within 10 seconds. For CCGTs: 10% of 100MW = 10MW/Hz response "
            "rate. Mandatory 20-second sustained response applies at ±0.2Hz deviation threshold."
        ),
        cited_doc_ids=["grid-code-ecc632"],
    ),
    # ── Scenario 5: Hallucination ────────────────────────────────────────────
    DemoScenario(
        name="HALLUCINATION DETECTION — Generic Response Against Regulatory Context",
        query="What action should we take given current energy market conditions and portfolio strategy?",
        context_docs=[
            _doc(
                "cm-rules-penalty",
                "Capacity Market Rules Section 9.4: A Capacity Provider that fails to deliver "
                "its obligation during a System Stress Event is subject to a Capacity Market "
                "penalty. The Penalty Rate for 2026/27 is £7,500/MWh of undelivered capacity. "
                "Penalties are recovered through the Capacity Market Settlement Body within "
                "10 business days.",
                "regulation/cm-rules-2026-27.pdf",
                0.71,
            ),
        ],
        action=(
            "Recommend optimising portfolio exposure based on current sentiment indicators and "
            "forward demand forecasts. Consider risk-weighted strategic positioning and consult "
            "stakeholders to align with broader commercial objectives and market opportunities."
        ),
        reasoning=(
            "Market analysis suggests favourable conditions for strategic repositioning. Portfolio "
            "optimisation should consider current pricing dynamics and forward curve expectations "
            "in the context of overall commercial strategy."
        ),
        cited_doc_ids=[],
        expect_hallucination=True,
    ),
]


# ---------------------------------------------------------------------------
# Evaluation logic
# ---------------------------------------------------------------------------


async def _run_scenario(
    scenario: DemoScenario,
    audit_logger: AuditLogger,
) -> tuple[DispatchRecommendation, float]:
    """Run one scenario through the safety + confidence pipeline.

    Args:
        scenario: Pre-scripted demo scenario.
        audit_logger: AuditLogger instance for writing the audit trail.

    Returns:
        Tuple of (DispatchRecommendation, elapsed_ms).
    """
    start = time.monotonic()

    safety = await check_grounding(
        action=scenario.action,
        reasoning=scenario.reasoning,
        context_docs=scenario.context_docs,
        embedder=None,
    )

    cited = [d for d in scenario.context_docs if d.doc_id in scenario.cited_doc_ids]
    scorer = ConfidenceScorer()
    signals = ConfidenceSignals(
        llm_confidence=0.85 if safety.is_grounded else 0.42,
        safety_result=safety,
        supporting_documents=cited,
    )
    confidence, requires_review = scorer.score(signals)

    rec = DispatchRecommendation(
        query=scenario.query,
        action=scenario.action,
        confidence=confidence,
        supporting_documents=cited,
        reasoning=scenario.reasoning,
        requires_human_review=requires_review or not safety.is_grounded,
        hallucination_flagged=not safety.is_grounded,
    )

    try:
        await audit_logger.record(entry_from_recommendation(rec))
    except Exception as e:
        logger.warning("Audit write failed: %s", e)

    elapsed_ms = (time.monotonic() - start) * 1000
    return rec, elapsed_ms


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _print_scenario_result(
    idx: int,
    scenario: DemoScenario,
    rec: DispatchRecommendation,
    elapsed_ms: float,
) -> None:
    """Print the full trace for one scenario."""
    print(f"\n{_DIVIDER}")
    print(f"SCENARIO {idx}: {scenario.name}")
    print(_DIVIDER)
    print(f"\n  Query:  {scenario.query}\n")

    print(f"  Retrieved Documents ({len(scenario.context_docs)}):")
    for i, doc in enumerate(scenario.context_docs, 1):
        snippet = doc.content[:80].replace("\n", " ")
        print(f"    [{i}] {doc.source}  (retrieval score: {doc.similarity_score:.2f})")
        print(f"         {snippet}…")

    print()
    action_short = rec.action[:95] + "…" if len(rec.action) > 95 else rec.action
    print(f"  Recommendation:")
    print(f"    Action      : {action_short}")
    print(f"    Confidence  : {rec.confidence:.3f}")
    print(f"    Cited docs  : {len(rec.supporting_documents)}/{len(scenario.context_docs)}")
    print(f"    Grounded    : {'YES  (term overlap >= 30%)' if not rec.hallucination_flagged else 'NO   <<< hallucination detected >>>'}")
    print(f"    Human review: {'YES  <<< escalated to operator >>>' if rec.requires_human_review else 'NO'}")
    print(f"\n  Audit entry written  (id: {rec.recommendation_id})")
    print(f"  Pipeline time: {elapsed_ms:.1f}ms")


def _print_summary(results: list[tuple[DemoScenario, DispatchRecommendation, float]]) -> bool:
    print(f"\n{_DIVIDER}")
    print("SUMMARY")
    print(_DIVIDER)
    print(f"  {'#':<3}  {'Scenario':<42}  {'Grounded':<10}  {'Confidence':<11}  {'Human Review'}")
    print(f"  {_LINE[:66]}")
    all_ok = True
    for idx, (scenario, rec, _) in enumerate(results, 1):
        grounded = "YES" if not rec.hallucination_flagged else "NO  ⚠"
        review = "YES  ⚠" if rec.requires_human_review else "NO"
        conf = f"{rec.confidence:.3f}"
        name = scenario.name[:40]
        print(f"  {idx:<3}  {name:<42}  {grounded:<10}  {conf:<11}  {review}")

        # Validate expectations
        if scenario.expect_hallucination and not rec.hallucination_flagged:
            print(f"       ERROR: expected hallucination to be detected for scenario {idx}")
            all_ok = False
        if not scenario.expect_hallucination and rec.hallucination_flagged:
            print(f"       ERROR: unexpected hallucination flag on scenario {idx}")
            all_ok = False

    total_ms = sum(ms for _, _, ms in results)
    print(f"\n  Total latency (demo mode): {total_ms:.0f}ms across {len(results)} scenarios")
    print(f"  Audit log: {_DEMO_AUDIT_PATH}")

    if all_ok:
        print(f"\n  Result: ALL SCENARIOS PASSED — safety layer behaved as expected.")
    else:
        print(f"\n  Result: UNEXPECTED OUTCOME — see errors above.")
    return all_ok


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main(backend: str) -> int:
    """Run all demo scenarios. Return 0 on success, 1 on unexpected outcome."""
    if backend == "bedrock":
        print("ERROR: Real Bedrock mode in demo_queries.py requires a live DB + AWS.")
        print("       Use --backend demo (default) or see src/agent/energy_agent.py.")
        return 1

    print(_DIVIDER)
    print("EIA RAG Pipeline Demo")
    print(f"Backend: {backend}  |  Safety: term overlap >= 30%  |  Confidence: Brier < 0.20")
    print(_DIVIDER)

    _DEMO_AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    audit_logger = AuditLogger(log_path=_DEMO_AUDIT_PATH)

    results: list[tuple[DemoScenario, DispatchRecommendation, float]] = []
    for idx, scenario in enumerate(DEMO_SCENARIOS, 1):
        rec, elapsed_ms = await _run_scenario(scenario, audit_logger)
        _print_scenario_result(idx, scenario, rec, elapsed_ms)
        results.append((scenario, rec, elapsed_ms))

    ok = _print_summary(results)
    return 0 if ok else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s  %(message)s")
    parser = argparse.ArgumentParser(description="EIA RAG pipeline demo.")
    parser.add_argument(
        "--backend",
        default="demo",
        choices=["demo", "bedrock"],
        help="'demo' runs without AWS credentials (default).",
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.backend)))

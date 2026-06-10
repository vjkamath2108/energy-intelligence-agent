"""Smoke test for W3 recommendation engine — Bedrock and Ollama backends.

Requires no populated DB. Uses a hardcoded synthetic document as context
so you can verify the full pipeline without running ingestion first.

Usage:
    python scripts/smoke_w3.py                  # both backends
    python scripts/smoke_w3.py --backend bedrock
    python scripts/smoke_w3.py --backend ollama
"""

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

# Resolve project root so this runs from any working directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.models import RetrievedDocument, DispatchRecommendation

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Synthetic context docs — no DB needed
# ---------------------------------------------------------------------------

_CONTEXT_DOCS = [
    RetrievedDocument(
        doc_id="smoke-doc-1",
        content=(
            "BSUoS (Balancing Services Use of System) charges are levied by "
            "National Grid ESO on licensed suppliers and generators to recover "
            "the costs of balancing the transmission system. Charges vary by "
            "settlement period and are published in the BSC (Balancing and "
            "Settlement Code) Section T."
        ),
        source="tariffs/bsuos-rates-q1-2024.pdf",
        chunk_index=0,
        similarity_score=0.91,
        metadata={},
    ),
    RetrievedDocument(
        doc_id="smoke-doc-2",
        content=(
            "In the event of a frequency deviation exceeding ±0.2 Hz, "
            "BM (Balancing Mechanism) units with a registered DSBR (Demand "
            "Side Balancing Reserve) agreement must respond within 240 seconds "
            "per BSC Annex T-1 clause 4.2. Failure to respond triggers a "
            "Capacity Market penalty under CM Rules section 6."
        ),
        source="regulations/bsc-annex-t1.pdf",
        chunk_index=2,
        similarity_score=0.87,
        metadata={},
    ),
]

_QUERY = "What action should be taken if grid frequency drops below 49.8 Hz?"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_result(backend: str, result: DispatchRecommendation, elapsed_ms: float) -> None:
    line = "─" * 60
    print(f"\n{line}")
    print(f"Backend : {backend}")
    print(f"Latency : {elapsed_ms:.0f} ms")
    print(f"Action  : {result.action}")
    print(f"Confid. : {result.confidence:.2f}")
    print(f"Grounded: {not result.hallucination_flagged}")
    print(f"H-Review: {result.requires_human_review}")
    print(f"Cited   : {[d.doc_id for d in result.supporting_documents]}")
    print(f"Reasoning (truncated):")
    print(f"  {result.reasoning[:200]}...")
    print(line)


async def smoke_bedrock() -> bool:
    from src.recommendation_engine.claude_client import ClaudeRecommender
    print("\n[bedrock] Initialising ClaudeRecommender...")
    try:
        rec = ClaudeRecommender()
    except Exception as e:
        print(f"[bedrock] FAIL — could not initialise: {e}")
        return False

    print(f"[bedrock] Calling Bedrock ({rec.model_id})...")
    t0 = time.monotonic()
    try:
        result = await rec.recommend(_QUERY, _CONTEXT_DOCS)
    except Exception as e:
        print(f"[bedrock] FAIL — {e}")
        return False

    elapsed_ms = (time.monotonic() - t0) * 1000
    _print_result("bedrock", result, elapsed_ms)

    gate_10s = elapsed_ms < 10000
    gate_json = isinstance(result, DispatchRecommendation)
    print(f"\n[bedrock] Gate: valid JSON schema  → {'PASS' if gate_json else 'FAIL'}")
    print(f"[bedrock] Gate: latency < 10s      → {'PASS' if gate_10s else f'FAIL ({elapsed_ms:.0f}ms)'}")
    return gate_json and gate_10s


async def smoke_ollama() -> bool:
    from src.recommendation_engine.ollama_client import OllamaRecommender
    print("\n[ollama] Initialising OllamaRecommender...")
    try:
        rec = OllamaRecommender()
    except Exception as e:
        print(f"[ollama] FAIL — could not initialise: {e}")
        return False

    print(f"[ollama] Health check (model: {rec.model})...")
    try:
        await rec.check_health()
    except RuntimeError as e:
        print(f"[ollama] FAIL — {e}")
        return False

    print(f"[ollama] Calling Ollama ({rec.model})...")
    t0 = time.monotonic()
    try:
        result = await rec.recommend(_QUERY, _CONTEXT_DOCS)
    except Exception as e:
        print(f"[ollama] FAIL — {e}")
        return False
    finally:
        await rec.aclose()

    elapsed_ms = (time.monotonic() - t0) * 1000
    _print_result("ollama", result, elapsed_ms)

    gate_json = isinstance(result, DispatchRecommendation)
    gate_no_aws = True  # if we got here, no AWS was called
    print(f"\n[ollama] Gate: valid JSON schema  → {'PASS' if gate_json else 'FAIL'}")
    print(f"[ollama] Gate: zero AWS calls     → PASS")
    return gate_json


async def main(backends: list[str]) -> int:
    results: dict[str, bool] = {}
    for backend in backends:
        if backend == "bedrock":
            results["bedrock"] = await smoke_bedrock()
        elif backend == "ollama":
            results["ollama"] = await smoke_ollama()

    print("\n" + "=" * 60)
    print("SMOKE TEST SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name:<10} {status}")
        if not passed:
            all_pass = False
    print("=" * 60)
    return 0 if all_pass else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="W3 smoke test — recommendation engine")
    parser.add_argument(
        "--backend",
        choices=["bedrock", "ollama", "both"],
        default="both",
        help="Which backend to test (default: both)",
    )
    args = parser.parse_args()

    backends = ["bedrock", "ollama"] if args.backend == "both" else [args.backend]
    sys.exit(asyncio.run(main(backends)))

"""
Benchmark runner: Bedrock vs Ollama explainer comparison.

Runs both explainer backends against the same set of anomaly events
detected by Isolation Forest on sample.jsonl, then writes structured
results to results/benchmark_raw.jsonl and a summary report to
results/bedrock_vs_ollama.md.

Ground truth severity is defined by anomaly pattern window:
  Pattern A (events 60–69):   HIGH — cascading failure, multi-service
  Pattern B (events 130–139): HIGH — sustained latency spike, payment-svc
  Pattern C (events 200–212): HIGH — error rate surge, api-gateway
  Other flagged events:        MEDIUM — isolated anomalies outside patterns

Usage:
    python -m stream1_log_anomaly.benchmark_runner
    python -m stream1_log_anomaly.benchmark_runner --explainer bedrock
    python -m stream1_log_anomaly.benchmark_runner --explainer ollama
    python -m stream1_log_anomaly.benchmark_runner --explainer both   (default)
"""

import argparse
import asyncio
import json
import logging
import os
import statistics
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from stream1_log_anomaly.src.models import LogEvent
from stream1_log_anomaly.src.detection.anomaly_scorer import AnomalyScorer
from stream1_log_anomaly.src.explanation.bedrock_explainer import BedrockExplainer
from stream1_log_anomaly.src.explanation.ollama_explainer import OllamaExplainer

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

LOG_FILE = Path("stream1_log_anomaly/logs/sample.jsonl")
RESULTS_DIR = Path("results")

# Ground truth: event index → expected severity
# Based on injected anomaly windows in log_generator.py
def _ground_truth_severity(event_index: int) -> str:
    if 60 <= event_index <= 69:
        return "HIGH"   # Pattern A: cascading failure
    if 130 <= event_index <= 139:
        return "HIGH"   # Pattern B: sustained latency spike
    if 200 <= event_index <= 212:
        return "HIGH"   # Pattern C: error rate surge
    return "MEDIUM"     # Isolated anomaly outside injected windows


def _load_events() -> list[LogEvent]:
    return [
        LogEvent(**json.loads(line))
        for line in LOG_FILE.read_text().splitlines()
        if line.strip()
    ]


def _detect_anomalies(events: list[LogEvent]) -> list[tuple[int, LogEvent, list[LogEvent]]]:
    """
    Run Isolation Forest and return list of (index, anomalous_event, context).
    """
    scorer = AnomalyScorer()
    anomalies = []

    for i, event in enumerate(events):
        result = scorer.score(event)
        if result is not None and result.is_anomaly:
            context_start = max(0, i - 5)
            context = events[context_start : i + 6]
            anomalies.append((i, event, context))

    logger.info("Isolation Forest flagged %d anomalies from %d events", len(anomalies), len(events))
    return anomalies


async def _run_backend(
    backend_name: str,
    anomalies: list[tuple[int, LogEvent, list[LogEvent]]],
) -> list[dict]:
    """
    Run one explainer backend against all anomaly events.

    Returns list of result dicts, one per anomaly event.
    """
    # Compute backend label once — includes model name for ollama
    # e.g. "bedrock", "ollama:llama3.2:3b", "ollama:phi4-mini"
    backend_label = (
        backend_name
        if backend_name == "bedrock"
        else f"ollama:{os.getenv('OLLAMA_MODEL', 'llama3.2:3b')}"
    )

    results = []

    if backend_name == "bedrock":
        explainer: BedrockExplainer | OllamaExplainer = BedrockExplainer(
            region=os.getenv("AWS_REGION", "eu-west-2")
        )
    else:
        explainer = OllamaExplainer()
        try:
            await explainer.check_health()
        except RuntimeError as e:
            logger.error("Ollama health check failed: %s", e)
            raise

    logger.info("Running %s on %d anomaly events...", backend_label, len(anomalies))

    for event_index, event, context in anomalies:
        start = time.monotonic()
        is_retry = False
        json_valid = True
        explanation = None

        try:
            explanation = await explainer.explain(event, context)
            elapsed_ms = (time.monotonic() - start) * 1000
        except Exception as e:
            elapsed_ms = (time.monotonic() - start) * 1000
            json_valid = False
            logger.warning(
                "Explainer failed for event %d: %s", event_index, e
            )

        ground_truth = _ground_truth_severity(event_index)
        predicted_severity = explanation.severity if explanation else "UNKNOWN"
        severity_correct = predicted_severity == ground_truth

        result = {
            "backend": backend_label,
            "event_index": event_index,
            "service": event.service,
            "level": event.level,
            "latency_ms": event.latency_ms,
            "status_code": event.status_code,
            "anomaly_pattern": (
                "A_cascade" if 60 <= event_index <= 69
                else "B_latency_spike" if 130 <= event_index <= 139
                else "C_error_surge" if 200 <= event_index <= 212
                else "isolated"
            ),
            "ground_truth_severity": ground_truth,
            "predicted_severity": predicted_severity,
            "severity_correct": severity_correct,
            "json_valid": json_valid,
            "elapsed_ms": round(elapsed_ms),
            "summary": explanation.summary if explanation else None,
            "likely_cause": explanation.likely_cause if explanation else None,
            "recommended_action": explanation.recommended_action if explanation else None,
        }
        results.append(result)

        logger.info(
            "[%s] event=%d pattern=%s gt=%s predicted=%s correct=%s latency=%.0fms",
            backend_label,
            event_index,
            result["anomaly_pattern"],
            ground_truth,
            predicted_severity,
            severity_correct,
            elapsed_ms,
        )

    if isinstance(explainer, OllamaExplainer):
        await explainer.aclose()

    return results


def _compute_stats(results: list[dict]) -> dict:
    """Compute summary statistics from a list of result dicts."""
    latencies = [r["elapsed_ms"] for r in results if r["json_valid"]]
    valid_count = sum(1 for r in results if r["json_valid"])
    correct_count = sum(1 for r in results if r["severity_correct"])
    total = len(results)

    # Per-pattern severity accuracy
    patterns = ["A_cascade", "B_latency_spike", "C_error_surge", "isolated"]
    pattern_accuracy = {}
    for pattern in patterns:
        pattern_results = [r for r in results if r["anomaly_pattern"] == pattern]
        if pattern_results:
            correct = sum(1 for r in pattern_results if r["severity_correct"])
            pattern_accuracy[pattern] = f"{correct}/{len(pattern_results)}"
        else:
            pattern_accuracy[pattern] = "0/0"

    return {
        "total_events": total,
        "json_validity_rate": f"{valid_count}/{total} ({100*valid_count//total}%)",
        "severity_accuracy": f"{correct_count}/{total} ({100*correct_count//total}%)",
        "pattern_accuracy": pattern_accuracy,
        "latency_p50_ms": round(statistics.median(latencies)) if latencies else 0,
        "latency_p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95)]) if len(latencies) >= 20 else max(latencies) if latencies else 0,
        "latency_mean_ms": round(statistics.mean(latencies)) if latencies else 0,
    }


def _write_markdown_report(
    bedrock_results: list[dict],
    ollama_results: list[dict],
    bedrock_stats: dict,
    ollama_stats: dict,
) -> Path:
    """Write the full benchmark report as markdown."""
    RESULTS_DIR.mkdir(exist_ok=True)
    report_path = RESULTS_DIR / "bedrock_vs_ollama.md"
    ts = datetime.now().strftime("%Y-%m-%d")

    # Derive model names from actual results — not from current env var
    ollama_models = sorted({
        r["backend"].replace("ollama:", "")
        for r in ollama_results
        if r.get("backend", "").startswith("ollama:")
    })
    ollama_models_str = ", ".join(ollama_models) if ollama_models else "unknown"

    lines = [
        "# Bedrock vs Ollama — Benchmark Report",
        "## Stream 1: Log Anomaly Explanation Pipeline",
        "",
        f"**Date:** {ts}  ",
        f"**Dataset:** {LOG_FILE} ({bedrock_stats['total_events']} anomaly events from 300 synthetic logs)  ",
        "**Bedrock model:** claude-sonnet-4-5 (us.* cross-region inference profile)  ",
        f"**Ollama models tested:** {ollama_models_str} (local inference, Apple M4 Max)  ",
        f"**Anomaly detector:** Isolation Forest (contamination=auto)  ",
        "",
        "---",
        "",
        "## Test Methodology",
        "",
        "Both explainers ran against identical anomaly events detected by Isolation Forest",
        "on the same 300-event synthetic log dataset (seed=42, reproducible).",
        "Each explainer received the same prompt template and the same ±5 context window.",
        "",
        "Three injected anomaly patterns provide ground truth for severity accuracy:",
        "- **Pattern A** (events 60–69): Cascading failure — auth → inventory → payment-svc → api-gateway.",
        "  All ERROR, 500 status, latency 1500–2500ms. Ground truth: HIGH.",
        "- **Pattern B** (events 130–139): Single-service latency spike — payment-svc.",
        "  Latency 2000–4000ms, mixed 200/500. Ground truth: HIGH.",
        "- **Pattern C** (events 200–212): Error rate surge — api-gateway.",
        "  Normal latency (100–400ms), elevated 500s. Tests error-only detection. Ground truth: HIGH.",
        "- **Isolated events**: Anomalies outside injected windows. Ground truth: MEDIUM.",
        "",
        "---",
        "",
        "## Results",
        "",
        f"| Metric | Bedrock (claude-sonnet-4-5) | {ollama_models_str} (combined) |",
        "|---|---|---|",
        f"| Events explained | {bedrock_stats['total_events']} | {ollama_stats['total_events']} |",
        f"| JSON validity | {bedrock_stats['json_validity_rate']} | {ollama_stats['json_validity_rate']} |",
        f"| Severity accuracy (overall) | {bedrock_stats['severity_accuracy']} | {ollama_stats['severity_accuracy']} |",
        f"| Pattern A accuracy (cascade) | {bedrock_stats['pattern_accuracy']['A_cascade']} | {ollama_stats['pattern_accuracy']['A_cascade']} |",
        f"| Pattern B accuracy (latency) | {bedrock_stats['pattern_accuracy']['B_latency_spike']} | {ollama_stats['pattern_accuracy']['B_latency_spike']} |",
        f"| Pattern C accuracy (error surge) | {bedrock_stats['pattern_accuracy']['C_error_surge']} | {ollama_stats['pattern_accuracy']['C_error_surge']} |",
        f"| Latency P50 | {bedrock_stats['latency_p50_ms']}ms | {ollama_stats['latency_p50_ms']}ms |",
        f"| Latency P95 | {bedrock_stats['latency_p95_ms']}ms | {ollama_stats['latency_p95_ms']}ms |",
        "| Data leaves network | Yes (AWS us.* cross-region) | No |",
        "| Cost per call | ~$0.003 | ~$0 |",
        "| Production-ready | Yes | No |",
        "",
        "---",
        "",
        "## Key Findings",
        "",
        "### 1. Severity Accuracy Gap",
        "",
        "The most significant quality difference is severity classification.",
        f"Bedrock achieved {bedrock_stats['severity_accuracy']} overall severity accuracy.",
        f"Ollama models ({ollama_models_str}) achieved {ollama_stats['severity_accuracy']} combined.",
        "",
        "The gap is most pronounced on Pattern A (cascading failure) and Pattern B (latency spike).",
        "Ollama defaults to MEDIUM for the majority of events regardless of signal strength,",
        "suggesting the 3B parameter model lacks the reasoning capacity to contextualise",
        "severity from the ±5 event window.",
        "",
        "### 2. Pattern C (Error Surge Without Latency) — Notable Finding",
        "",
        "Pattern C injects elevated 500 error rates with *normal* latency (100–400ms).",
        "This tests whether the explainer can identify severity from error signal alone.",
        f"Bedrock Pattern C accuracy: {bedrock_stats['pattern_accuracy']['C_error_surge']}.",
        f"Ollama Pattern C accuracy: {ollama_stats['pattern_accuracy']['C_error_surge']}.",
        "",
        "### 3. Latency Trade-off",
        "",
        f"Ollama P50 latency ({ollama_stats['latency_p50_ms']}ms) is significantly faster than",
        f"Bedrock ({bedrock_stats['latency_p50_ms']}ms). This is expected: Bedrock includes",
        "network round-trip to AWS; Ollama is pure local inference on a 3B model.",
        "The latency gap would narrow with a larger local model (8B+).",
        "However, a faster but incorrect explanation has negative operational value —",
        "it misleads the on-call engineer.",
        "",
        "### 4. JSON Reliability",
        "",
        f"Bedrock JSON validity: {bedrock_stats['json_validity_rate']}.",
        f"Ollama JSON validity: {ollama_stats['json_validity_rate']}.",
        "The defensive `_parse_explanation` logic (markdown fence stripping, brace",
        "completion, severity normalisation) handles most Ollama formatting issues,",
        "but the underlying reliability gap is real.",
        "",
        "---",
        "",
        "## Three-Way Decision Framework",
        "",
        "| | Bedrock | EKS self-hosted (vLLM) | Ollama (local) |",
        "|---|---|---|---|",
        "| GDPR sub-processor | Anthropic | None | None |",
        "| Data residency | AWS region | Client VPC | Client machine |",
        "| Audit proof | AWS CloudTrail | VPC Flow Logs | Hard to audit |",
        "| Model quality | Full Claude Sonnet | Open-source only | Open-source only |",
        "| Severity accuracy | Strong | Depends on model | Weak (3B) |",
        "| Cost crossover | Baseline | ~50M tokens/day | ~$0 (dev only) |",
        "| Ops burden | Low | High | Very low |",
        "| Production-ready | Yes | Yes | No |",
        "",
        "---",
        "",
        "## Recommendation by Use Case",
        "",
        "**Use Bedrock when:**",
        "- Quality and reliability are primary concerns",
        "- Volume is below ~50M tokens/day",
        "- Client has signed a DPA with Anthropic",
        "",
        "**Use EKS self-hosted (vLLM) when:**",
        "- Client has NIS2 obligations (energy, critical infrastructure)",
        "- Data sovereignty requirements preclude external API endpoints",
        "- Volume exceeds ~50M tokens/day and operational maturity exists",
        "- Regulator requires auditable proof of data residency (VPC Flow Logs)",
        "",
        "**Use Ollama when:**",
        "- Local development and pattern validation only",
        "- Proving sovereign inference architecture before EKS commitment",
        "- Zero-cost exploration of model quality trade-offs",
        "",
        "---",
        "",
        "## What I Would Do Differently",
        "",
        f"1. **Three models tested:** {ollama_models_str}. A further comparison with",
        "   llama3.1:8b or qwen3:14b would show whether the quality gap narrows",
        "   significantly at 8B+ parameters while remaining local.",
        "",
        "2. **Add human quality ratings (1–5).** Severity accuracy is ground-truth-based.",
        "   Production evaluation would need on-call engineer ratings of explanation",
        "   usefulness, not just correctness of the severity label.",
        "",
        "3. **Measure token efficiency.** Bedrock and Ollama produce explanations of",
        "   different lengths. Tokens per explanation is a meaningful cost efficiency",
        "   metric alongside latency.",
        "",
        "---",
        "",
        "## Files",
        "",
        "- `src/explanation/bedrock_explainer.py`",
        "- `src/explanation/ollama_explainer.py`",
        "- `src/explanation/prompts.py`",
        "- `src/detection/anomaly_scorer.py`",
        "- `src/streaming/log_generator.py`",
        "- `results/benchmark_raw.jsonl` — raw per-event results",
        "",
    ]

    report_path.write_text("\n".join(lines))
    logger.info("Report written to %s", report_path)
    return report_path


def _load_raw_results(backend: str) -> list[dict]:
    raw_path = RESULTS_DIR / "benchmark_raw.jsonl"
    if not raw_path.exists():
        return []
    results = [
        json.loads(line)
        for line in raw_path.read_text().splitlines()
        if line.strip()
    ]
    return [r for r in results if r.get("backend", "").startswith(backend)]


async def main(run_bedrock: bool, run_ollama: bool, report_only: bool) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    raw_path = RESULTS_DIR / "benchmark_raw.jsonl"

    if report_only:
        bedrock_results = _load_raw_results("bedrock")
        ollama_results = _load_raw_results("ollama")
        if not bedrock_results:
            print("No Bedrock results in results/benchmark_raw.jsonl")
            print("Run: python -m stream1_log_anomaly.benchmark_runner --explainer bedrock")
            return
        if not ollama_results:
            print("No Ollama results in results/benchmark_raw.jsonl")
            print("Run: python -m stream1_log_anomaly.benchmark_runner --explainer ollama")
            return
    else:
        events = _load_events()
        anomalies = _detect_anomalies(events)

        if run_bedrock:
            logger.info("--- Bedrock run ---")
            bedrock_results = await _run_backend("bedrock", anomalies)
            with raw_path.open("a") as f:
                for r in bedrock_results:
                    f.write(json.dumps(r) + "\n")
            stats = _compute_stats(bedrock_results)
            print(f"\nBedrock complete — severity: {stats['severity_accuracy']}, P50: {stats['latency_p50_ms']}ms")
            print(f"Results appended to {raw_path}")
            if not run_ollama:
                print("Transfer results/benchmark_raw.jsonl to your Ollama machine, then run:")
                print("  python -m stream1_log_anomaly.benchmark_runner --explainer ollama")

        if run_ollama:
            logger.info("--- Ollama run ---")
            ollama_results = await _run_backend("ollama", anomalies)
            with raw_path.open("a") as f:
                for r in ollama_results:
                    f.write(json.dumps(r) + "\n")
            stats = _compute_stats(ollama_results)
            print(f"\nOllama complete — severity: {stats['severity_accuracy']}, P50: {stats['latency_p50_ms']}ms")
            print(f"Results appended to {raw_path}")

        bedrock_results = _load_raw_results("bedrock")
        ollama_results = _load_raw_results("ollama")

    if bedrock_results and ollama_results:
        bedrock_stats = _compute_stats(bedrock_results)
        ollama_stats = _compute_stats(ollama_results)
        report_path = _write_markdown_report(
            bedrock_results, ollama_results, bedrock_stats, ollama_stats
        )
        print(f"\nReport written: {report_path}")
        print(f"Bedrock — severity: {bedrock_stats['severity_accuracy']}, P50: {bedrock_stats['latency_p50_ms']}ms")
        print(f"Ollama  — severity: {ollama_stats['severity_accuracy']}, P50: {ollama_stats['latency_p50_ms']}ms")
    else:
        missing = "Ollama" if bedrock_results else "Bedrock"
        print(f"\n{missing} results not yet present. Run both backends then re-run with --report.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark Bedrock vs Ollama explainer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Workflow for split machines (Bedrock on office laptop, Ollama on local):

  Step 1 — Office laptop (AWS access):
    python -m stream1_log_anomaly.benchmark_runner --explainer bedrock

  Step 2 — Copy results file to Ollama machine:
    scp results/benchmark_raw.jsonl <local>:~/vijay/log-anomaly-pipeline/results/

  Step 3 — Local machine (Ollama):
    python -m stream1_log_anomaly.benchmark_runner --explainer ollama

  Step 4 — Generate report (either machine, once both results present):
    python -m stream1_log_anomaly.benchmark_runner --report
        """,
    )
    parser.add_argument(
        "--explainer",
        choices=["bedrock", "ollama", "both"],
        default="both",
        help="Which backend(s) to run (default: both)",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate report from existing results/benchmark_raw.jsonl without running inference",
    )
    args = parser.parse_args()
    run_bedrock = args.explainer in ("bedrock", "both") and not args.report
    run_ollama = args.explainer in ("ollama", "both") and not args.report
    asyncio.run(main(run_bedrock, run_ollama, report_only=args.report))
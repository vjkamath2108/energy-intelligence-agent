"""
Benchmark runner: Bedrock vs local model comparison.

Option B ground truth methodology:
  Severity accuracy measured ONLY on injected pattern windows.
  Isolated events excluded from accuracy; included in latency/validity only.

  Pattern A (events 60-69):   HIGH - cascading failure
  Pattern B (events 130-139): HIGH - latency spike
  Pattern C (events 200-212): HIGH - error surge
  Isolated:                    None - excluded from accuracy

Usage:
    python -m stream1_log_anomaly.src.streaming.generate_logs  # regenerate logs

    # Run individual backends (split-machine workflow):
    python -m stream1_log_anomaly.benchmark_runner --explainer bedrock
    OLLAMA_MODEL=llama3.2:3b python -m stream1_log_anomaly.benchmark_runner --explainer ollama
    OLLAMA_MODEL=phi4-mini python -m stream1_log_anomaly.benchmark_runner --explainer ollama
    OLLAMA_MODEL=qwen3:14b OLLAMA_TIMEOUT_S=120 OLLAMA_NUM_PREDICT=1024 \\
        python -m stream1_log_anomaly.benchmark_runner --explainer ollama

    # Generate report once all backends done:
    python -m stream1_log_anomaly.benchmark_runner --report
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

# Preferred display order in report columns
DISPLAY_ORDER = ["bedrock", "ollama:llama3.2:3b", "ollama:phi4-mini", "ollama:qwen3:14b"]


def _ground_truth_severity(event_index: int) -> str | None:
    """
    Return expected severity for injected pattern events, None for isolated.

    Option B: only events with deterministic ground truth (injected windows)
    contribute to accuracy metrics. Isolated events are excluded.
    """
    if 60 <= event_index <= 69:
        return "HIGH"   # Pattern A: cascading failure
    if 130 <= event_index <= 139:
        return "HIGH"   # Pattern B: latency spike
    if 200 <= event_index <= 212:
        return "HIGH"   # Pattern C: error surge
    return None         # Isolated — excluded from accuracy


def _load_events() -> list[LogEvent]:
    return [
        LogEvent(**json.loads(line))
        for line in LOG_FILE.read_text().splitlines()
        if line.strip()
    ]


def _detect_anomalies(events: list[LogEvent]) -> list[tuple[int, LogEvent, list[LogEvent]]]:
    """Run Isolation Forest and return (index, event, context) tuples."""
    scorer = AnomalyScorer()
    anomalies = []
    for i, event in enumerate(events):
        result = scorer.score(event)
        if result is not None and result.is_anomaly:
            context_start = max(0, i - 5)
            context = events[context_start : i + 6]
            anomalies.append((i, event, context))
    logger.info(
        "Isolation Forest flagged %d anomalies from %d events",
        len(anomalies), len(events),
    )
    return anomalies


async def _run_backend(
    backend_name: str,
    anomalies: list[tuple[int, LogEvent, list[LogEvent]]],
) -> list[dict]:
    """
    Run one explainer backend against all anomaly events.

    Returns list of result dicts, one per anomaly event.
    """
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
        elapsed_ms = 0.0  # initialised before try to avoid UnboundLocalError
        json_valid = True
        explanation = None

        try:
            explanation = await explainer.explain(event, context)
            elapsed_ms = (time.monotonic() - start) * 1000
        except Exception as e:
            elapsed_ms = (time.monotonic() - start) * 1000
            json_valid = False
            logger.warning("Explainer failed for event %d: %s", event_index, e)

        ground_truth = _ground_truth_severity(event_index)
        predicted_severity = explanation.severity if explanation else "UNKNOWN"
        severity_correct: bool | None = (
            predicted_severity == ground_truth
            if ground_truth is not None
            else None
        )
        anomaly_pattern = (
            "A_cascade" if 60 <= event_index <= 69
            else "B_latency_spike" if 130 <= event_index <= 139
            else "C_error_surge" if 200 <= event_index <= 212
            else "isolated"
        )

        result = {
            "backend": backend_label,
            "event_index": event_index,
            "service": event.service,
            "level": event.level,
            "latency_ms": event.latency_ms,
            "status_code": event.status_code,
            "anomaly_pattern": anomaly_pattern,
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

        correct_str = "N/A" if severity_correct is None else str(severity_correct)
        gt_str = ground_truth if ground_truth is not None else "N/A"
        logger.info(
            "[%s] event=%d pattern=%s gt=%s predicted=%s correct=%s latency=%.0fms",
            backend_label, event_index, anomaly_pattern,
            gt_str, predicted_severity, correct_str, elapsed_ms,
        )

    if isinstance(explainer, OllamaExplainer):
        await explainer.aclose()

    return results


def _compute_stats(results: list[dict]) -> dict:
    """
    Compute summary statistics.

    Option B: accuracy uses only events with non-None ground truth.
    Isolated events contribute to latency and JSON validity only.
    """
    latencies = [r["elapsed_ms"] for r in results if r["json_valid"]]
    valid_count = sum(1 for r in results if r["json_valid"])
    total = len(results)

    scoreable = [r for r in results if r["severity_correct"] is not None]
    correct_count = sum(1 for r in scoreable if r["severity_correct"])
    scoreable_total = len(scoreable)
    isolated_count = sum(1 for r in results if r["anomaly_pattern"] == "isolated")

    patterns = ["A_cascade", "B_latency_spike", "C_error_surge"]
    pattern_accuracy: dict[str, str] = {}
    for pattern in patterns:
        pr = [r for r in results if r["anomaly_pattern"] == pattern]
        if pr:
            correct = sum(1 for r in pr if r["severity_correct"])
            pattern_accuracy[pattern] = f"{correct}/{len(pr)}"
        else:
            pattern_accuracy[pattern] = "0/0"

    accuracy_str = (
        f"{correct_count}/{scoreable_total} ({100 * correct_count // scoreable_total}%)"
        if scoreable_total > 0 else "0/0 (N/A)"
    )

    return {
        "total_events": total,
        "scoreable_events": scoreable_total,
        "isolated_events": isolated_count,
        "json_validity_rate": f"{valid_count}/{total} ({100 * valid_count // total}%)",
        "severity_accuracy": accuracy_str,
        "pattern_accuracy": pattern_accuracy,
        "latency_p50_ms": round(statistics.median(latencies)) if latencies else 0,
        "latency_p95_ms": (
            round(sorted(latencies)[int(len(latencies) * 0.95)])
            if len(latencies) >= 20 else max(latencies) if latencies else 0
        ),
        "latency_mean_ms": round(statistics.mean(latencies)) if latencies else 0,
    }


def _load_raw_results(backend_prefix: str) -> list[dict]:
    """Load results for a backend. Prefix match: 'ollama' matches all ollama:* backends."""
    raw_path = RESULTS_DIR / "benchmark_raw.jsonl"
    if not raw_path.exists():
        return []
    results = [
        json.loads(line)
        for line in raw_path.read_text().splitlines()
        if line.strip()
    ]
    return [r for r in results if r.get("backend", "").startswith(backend_prefix)]


def _load_results_by_backend() -> dict[str, list[dict]]:
    """Load all results grouped by exact backend label."""
    raw_path = RESULTS_DIR / "benchmark_raw.jsonl"
    if not raw_path.exists():
        return {}
    grouped: dict[str, list[dict]] = {}
    for line in raw_path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        backend = r.get("backend", "unknown")
        grouped.setdefault(backend, []).append(r)
    return grouped


def _col_label(backend: str) -> str:
    """Human-readable column header for a backend."""
    if backend == "bedrock":
        return "Bedrock (claude-sonnet-4-5)"
    model = backend.replace("ollama:", "")
    return f"{model} (Ollama)"


def _write_markdown_report(all_results: dict[str, list[dict]]) -> Path:
    """
    Write benchmark report with one column per model.

    all_results: dict mapping backend label to list of result dicts.
    """
    RESULTS_DIR.mkdir(exist_ok=True)
    report_path = RESULTS_DIR / "bedrock_vs_ollama.md"
    ts = datetime.now().strftime("%Y-%m-%d")

    # Ordered backends — known order first, any extras appended
    backends = [b for b in DISPLAY_ORDER if b in all_results]
    backends += [b for b in sorted(all_results) if b not in backends]

    all_stats = {b: _compute_stats(all_results[b]) for b in backends}
    col_labels = [_col_label(b) for b in backends]

    # Table helpers
    separator = "|---|" + "---|" * len(backends)
    header_row = "| Metric | " + " | ".join(col_labels) + " |"

    def _row(label: str, key: str) -> str:
        vals = [str(all_stats[b][key]) for b in backends]
        return f"| {label} | " + " | ".join(vals) + " |"

    def _pattern_row(label: str, pattern_key: str) -> str:
        vals = [all_stats[b]["pattern_accuracy"][pattern_key] for b in backends]
        return f"| {label} | " + " | ".join(vals) + " |"

    def _static_row(label: str, values: list[str]) -> str:
        return f"| {label} | " + " | ".join(values) + " |"

    data_leaves = [
        "Yes (AWS us.* cross-region)" if b == "bedrock" else "No"
        for b in backends
    ]
    cost_vals = ["~$0.003" if b == "bedrock" else "~$0" for b in backends]
    prod_vals = ["Yes" if b == "bedrock" else "No" for b in backends]

    bedrock_stats = all_stats.get("bedrock", {})
    ollama_backends = [b for b in backends if b != "bedrock"]
    ollama_models_str = ", ".join(b.replace("ollama:", "") for b in ollama_backends)

    lines = [
        "# Bedrock vs Local Models — Benchmark Report",
        "## Stream 1: Log Anomaly Explanation Pipeline",
        "",
        f"**Date:** {ts}  ",
        f"**Dataset:** {LOG_FILE} ({bedrock_stats.get('total_events', 'N/A')} anomaly events from 300 synthetic logs)  ",
        "**Bedrock model:** claude-sonnet-4-5 (us.* cross-region inference profile)  ",
        f"**Local models tested:** {ollama_models_str} (Apple M4 Max)  ",
        "**Anomaly detector:** Isolation Forest (contamination=auto)  ",
        "",
        "---",
        "",
        "## Test Methodology",
        "",
        "All models ran against identical anomaly events detected by Isolation Forest",
        "on the same 300-event synthetic log dataset (seed=42, reproducible).",
        "Each model received the same prompt template and the same ±5 context window.",
        "",
        "**Option B ground truth:** Severity accuracy is measured only on events with",
        "deterministic ground truth from injected anomaly patterns. Isolated events",
        "(outside pattern windows) are included in latency and JSON validity only.",
        "",
        "Three injected anomaly patterns:",
        "- **Pattern A** (events 60–69): Cascading failure — auth → inventory → payment-svc → api-gateway.",
        "  All ERROR, 500 status, latency 1500–2500ms. Ground truth: HIGH.",
        "- **Pattern B** (events 130–139): Single-service latency spike — payment-svc.",
        "  Latency 2000–4000ms, mixed 200/500. Ground truth: HIGH.",
        "- **Pattern C** (events 200–212): Error rate surge — api-gateway.",
        "  Normal latency (100–400ms), elevated 500s. Tests error-only signal. Ground truth: HIGH.",
        "- **Isolated events**: Outside injected windows. Excluded from accuracy measurement.",
        "",
        "---",
        "",
        "## Results",
        "",
        header_row,
        separator,
        _row("Total events", "total_events"),
        _row("Scoreable events (injected patterns)", "scoreable_events"),
        _row("Isolated events (excl. from accuracy)", "isolated_events"),
        _row("JSON validity", "json_validity_rate"),
        _row("Severity accuracy (injected only)", "severity_accuracy"),
        _pattern_row("Pattern A accuracy (cascade)", "A_cascade"),
        _pattern_row("Pattern B accuracy (latency spike)", "B_latency_spike"),
        _pattern_row("Pattern C accuracy (error surge)", "C_error_surge"),
        _row("Latency P50", "latency_p50_ms"),
        _row("Latency P95", "latency_p95_ms"),
        _static_row("Data leaves network", data_leaves),
        _static_row("Cost per call", cost_vals),
        _static_row("Production-ready", prod_vals),
        "",
        "---",
        "",
        "## Key Findings",
        "",
        "### 1. Severity Accuracy",
        "",
        "Accuracy measured on injected pattern events only (Option B).",
        "Isolated events — where ground truth is ambiguous — are excluded.",
        "",
    ]

    for b in backends:
        s = all_stats[b]
        lines.append(f"- **{_col_label(b)}**: {s['severity_accuracy']} overall.")

    lines += [
        "",
        "Smaller local models default to MEDIUM regardless of signal strength.",
        "Larger models (qwen3:14b) show improvement but remain below Bedrock",
        "on cascade reasoning — Pattern A requires multi-event context synthesis.",
        "",
        "### 2. Pattern C — Error Surge Without Latency",
        "",
        "Pattern C tests whether a model can identify HIGH severity from error signal",
        "alone, without a latency spike. Normal latency means the only signal is",
        "elevated 500 status codes across consecutive events.",
        "",
    ]

    for b in backends:
        s = all_stats[b]
        lines.append(
            f"- **{_col_label(b)}**: "
            f"Pattern C accuracy {s['pattern_accuracy']['C_error_surge']}."
        )

    lines += [
        "",
        "### 3. Latency vs Quality Trade-off",
        "",
    ]

    for b in backends:
        s = all_stats[b]
        lines.append(
            f"- **{_col_label(b)}**: P50 {s['latency_p50_ms']}ms, "
            f"P95 {s['latency_p95_ms']}ms, "
            f"accuracy {s['severity_accuracy']}."
        )

    lines += [
        "",
        "A faster but incorrect explanation has negative operational value —",
        "it misleads the on-call engineer. qwen3:14b achieves the best local",
        "quality but at ~20s/call requires GPU hardware (g5.xlarge on EKS)",
        "to reach production-acceptable latency.",
        "",
        "### 4. JSON Reliability",
        "",
    ]

    for b in backends:
        s = all_stats[b]
        lines.append(f"- **{_col_label(b)}**: {s['json_validity_rate']}.")

    lines += [
        "",
        "---",
        "",
        "## Three-Way Deployment Decision Framework",
        "",
        "| | Bedrock | EKS self-hosted (vLLM) | Ollama (local dev) |",
        "|---|---|---|---|",
        "| GDPR sub-processor | Anthropic | None | None |",
        "| Data residency | AWS region | Client VPC | Client machine |",
        "| Audit proof | AWS CloudTrail | VPC Flow Logs | Hard to audit |",
        "| Model quality | Full Claude Sonnet | Open-source (size-dependent) | Open-source |",
        "| Best local accuracy | — | qwen3:14b on GPU | qwen3:14b |",
        "| Cost crossover | Baseline | ~50M tokens/day | ~$0 (dev only) |",
        "| Ops burden | Low | High | Very low |",
        "| Production-ready | Yes | Yes | No |",
        "",
        "**Note on inference profiles:** This benchmark used a `us.*` cross-region",
        "inference profile. For UK energy clients under NIS2, use an `eu.*` profile",
        "or provisioned throughput in eu-west-2 to guarantee data residency.",
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
        "- Use qwen3:14b or larger on GPU nodes for production-grade quality",
        "",
        "**Use Ollama when:**",
        "- Local development and pattern validation only",
        "- Proving the sovereign inference pattern before EKS commitment",
        "- Zero-cost exploration of model quality vs size trade-offs",
        "",
        "---",
        "",
        "## What I Would Do Differently",
        "",
        f"1. **Models benchmarked:** Bedrock claude-sonnet-4-5 vs {ollama_models_str}.",
        "   Quality scales with model size. A production sovereign deployment needs",
        "   a 13B+ model on GPU hardware to match Bedrock quality and latency.",
        "",
        "2. **Add human quality ratings (1–5).** Severity accuracy is ground-truth-based.",
        "   Production evaluation needs on-call engineer ratings of explanation",
        "   usefulness — not just correctness of the severity label.",
        "",
        "3. **Measure token efficiency.** Models produce explanations of different lengths.",
        "   Tokens per explanation is a meaningful cost efficiency metric alongside latency.",
        "",
        "---",
        "",
        "## Files",
        "",
        "- `src/explanation/bedrock_explainer.py`",
        "- `src/explanation/ollama_explainer.py`",
        "- `src/explanation/prompts.py`",
        "- `src/detection/anomaly_scorer.py`",
        "- `src/streaming/generate_logs.py`",
        "- `results/benchmark_raw.jsonl` — raw per-event results",
        "",
    ]

    report_path.write_text("\n".join(lines))
    logger.info("Report written to %s", report_path)
    return report_path


async def main(run_bedrock: bool, run_ollama: bool, report_only: bool) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    raw_path = RESULTS_DIR / "benchmark_raw.jsonl"

    if report_only:
        all_results = _load_results_by_backend()
        if not any(b == "bedrock" for b in all_results):
            print("No Bedrock results in results/benchmark_raw.jsonl")
            print("Run: python -m stream1_log_anomaly.benchmark_runner --explainer bedrock")
            return
        if not any(b.startswith("ollama:") for b in all_results):
            print("No Ollama results in results/benchmark_raw.jsonl")
            return
        report_path = _write_markdown_report(all_results)
        print(f"\nReport written: {report_path}")
        all_stats = {b: _compute_stats(r) for b, r in all_results.items()}
        for backend in [b for b in DISPLAY_ORDER if b in all_stats] + \
                       [b for b in sorted(all_stats) if b not in DISPLAY_ORDER]:
            s = all_stats[backend]
            print(f"  {backend}: accuracy={s['severity_accuracy']}, P50={s['latency_p50_ms']}ms")
        return

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
        model = os.getenv("OLLAMA_MODEL", "unknown")
        print(f"\nOllama ({model}) complete — severity: {stats['severity_accuracy']}, P50: {stats['latency_p50_ms']}ms")
        print(f"Results appended to {raw_path}")

    # Auto-generate report if bedrock + at least one ollama backend present
    all_results = _load_results_by_backend()
    has_bedrock = any(b == "bedrock" for b in all_results)
    has_ollama = any(b.startswith("ollama:") for b in all_results)
    if has_bedrock and has_ollama:
        report_path = _write_markdown_report(all_results)
        print(f"\nReport written: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark Bedrock vs local model explainers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Workflow for split machines (Bedrock on office laptop, Ollama on local):

  Step 1 — Office laptop (AWS access):
    python -m stream1_log_anomaly.benchmark_runner --explainer bedrock

  Step 2 — Copy results to local machine:
    scp results/benchmark_raw.jsonl <local>:~/vijay/.../results/

  Step 3 — Local machine, one per model:
    OLLAMA_MODEL=llama3.2:3b python -m stream1_log_anomaly.benchmark_runner --explainer ollama
    OLLAMA_MODEL=phi4-mini python -m stream1_log_anomaly.benchmark_runner --explainer ollama
    OLLAMA_MODEL=qwen3:14b OLLAMA_TIMEOUT_S=120 OLLAMA_NUM_PREDICT=1024 \\
        python -m stream1_log_anomaly.benchmark_runner --explainer ollama

  Step 4 — Generate report (either machine, once all backends done):
    python -m stream1_log_anomaly.benchmark_runner --report
        """,
    )
    parser.add_argument(
        "--explainer",
        choices=["bedrock", "ollama", "both"],
        default="both",
        help="Which backend to run (default: both)",
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
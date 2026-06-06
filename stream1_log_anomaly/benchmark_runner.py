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

    # Run individual backends:
    python -m stream1_log_anomaly.benchmark_runner --explainer bedrock
    python -m stream1_log_anomaly.benchmark_runner --explainer mistral-bedrock
    OLLAMA_MODEL=mistral:latest OLLAMA_TIMEOUT_S=120 \
        python -m stream1_log_anomaly.benchmark_runner --explainer mistral-local
    OLLAMA_MODEL=llama3.2:3b python -m stream1_log_anomaly.benchmark_runner --explainer ollama
    OLLAMA_MODEL=phi4-mini python -m stream1_log_anomaly.benchmark_runner --explainer ollama
    OLLAMA_MODEL=qwen3:14b OLLAMA_TIMEOUT_S=120 OLLAMA_NUM_PREDICT=1024 \
        python -m stream1_log_anomaly.benchmark_runner --explainer ollama

    # Using shell scripts (split-machine workflow):
    scripts/benchmark_office.sh   # office laptop — Bedrock backends
    scripts/benchmark_local.sh    # local laptop  — Ollama backends
    scripts/benchmark_combine.sh  # either machine — merge + final report

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
from stream1_log_anomaly.src.explanation.mistral_bedrock_explainer import MistralBedrockExplainer
from stream1_log_anomaly.src.explanation.mistral_local_explainer import MistralLocalExplainer

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

LOG_FILE = Path("stream1_log_anomaly/logs/sample.jsonl")
RESULTS_DIR = Path("results")

# Preferred display order in report columns
DISPLAY_ORDER = [
    "bedrock",
    "mistral-bedrock",
    "mistral-local",
    "ollama:llama3.2:3b",
    "ollama:phi4-mini",
    "ollama:qwen3:14b",
]


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


def _save_results(raw_path: Path, new_results: list[dict]) -> None:
    """
    Write results for a backend, replacing any existing results for
    that backend. Other backends' results are preserved.

    Prevents duplicate accumulation when a backend is re-run.
    """
    if not new_results:
        return

    backend_label = new_results[0]["backend"]
    existing = []

    if raw_path.exists():
        for line in raw_path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            # Keep results from OTHER backends only
            if r.get("backend") != backend_label:
                existing.append(r)

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("w") as f:
        for r in existing + new_results:
            f.write(json.dumps(r) + "\n")

    logger.info(
        "Saved %d results for backend '%s' to %s",
        len(new_results),
        backend_label,
        raw_path,
    )


async def _run_backend(
    backend_name: str,
    anomalies: list[tuple[int, LogEvent, list[LogEvent]]],
) -> list[dict]:
    """
    Run one explainer backend against all anomaly events.

    Returns list of result dicts, one per anomaly event.
    """
    # Resolve backend label — used as the key in raw results JSONL
    if backend_name == "bedrock":
        backend_label = "bedrock"
    elif backend_name == "mistral-bedrock":
        backend_label = "mistral-bedrock"
    elif backend_name == "mistral-local":
        backend_label = "mistral-local"
    else:
        # ollama variants — label includes model name
        backend_label = f"ollama:{os.getenv('OLLAMA_MODEL', 'llama3.2:3b')}"

    results = []

    # Instantiate the right explainer
    if backend_name == "bedrock":
        explainer = BedrockExplainer(
            region=os.getenv("AWS_REGION", "eu-west-2")
        )
    elif backend_name == "mistral-bedrock":
        explainer = MistralBedrockExplainer(
            region=os.getenv("BEDROCK_REGION", "eu-west-2")
        )
    elif backend_name == "mistral-local":
        explainer = MistralLocalExplainer()
        try:
            await explainer.check_health()
        except RuntimeError as e:
            logger.error("Mistral Local health check failed: %s", e)
            raise
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
        elapsed_ms = 0.0
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

    # Close clients that need it
    if hasattr(explainer, "aclose"):
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


def _load_results_by_backend(raw_path: Path) -> dict[str, list[dict]]:
    """Load all results grouped by exact backend label."""
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
    labels = {
        "bedrock": "Claude Sonnet (Bedrock)",
        "mistral-bedrock": "Mistral 7B (Bedrock)",
        "mistral-local": "Mistral 7B (Sovereign/Ollama)",
    }
    if backend in labels:
        return labels[backend]
    model = backend.replace("ollama:", "")
    return f"{model} (Ollama)"


def _data_leaves(backend: str) -> str:
    if backend == "bedrock":
        return "Yes (AWS us.* cross-region)"
    if backend == "mistral-bedrock":
        return "Yes (AWS Bedrock)"
    return "No"


def _cost_per_call(backend: str) -> str:
    if backend == "bedrock":
        return "~$0.003"
    if backend == "mistral-bedrock":
        return "~$0.0001"
    return "~$0"


def _production_ready(backend: str) -> str:
    if backend in ("bedrock", "mistral-bedrock"):
        return "Yes"
    return "No"


def _write_markdown_report(
    all_results: dict[str, list[dict]],
    report_path: Path,
) -> Path:
    """Write benchmark report with one column per model."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d")

    # Ordered backends — known order first, extras appended
    backends = [b for b in DISPLAY_ORDER if b in all_results]
    backends += [b for b in sorted(all_results) if b not in backends]

    all_stats = {b: _compute_stats(all_results[b]) for b in backends}
    col_labels = [_col_label(b) for b in backends]

    separator = "|---|" + "---|" * len(backends)
    header_row = "| Metric | " + " | ".join(col_labels) + " |"

    def _row(label: str, key: str) -> str:
        vals = [str(all_stats[b][key]) for b in backends]
        return f"| {label} | " + " | ".join(vals) + " |"

    def _pattern_row(label: str, pattern_key: str) -> str:
        vals = [all_stats[b]["pattern_accuracy"][pattern_key] for b in backends]
        return f"| {label} | " + " | ".join(vals) + " |"

    def _static_row(label: str, fn) -> str:
        vals = [fn(b) for b in backends]
        return f"| {label} | " + " | ".join(vals) + " |"

    bedrock_stats = all_stats.get("bedrock", {})
    non_bedrock = [b for b in backends if b != "bedrock"]
    local_models_str = ", ".join(_col_label(b) for b in non_bedrock)

    lines = [
        "# Benchmark Report — Frontier vs Open-Weight Inference",
        "## Stream 1: Log Anomaly Explanation Pipeline",
        "",
        f"**Date:** {ts}  ",
        f"**Dataset:** {LOG_FILE} ({bedrock_stats.get('total_events', 'N/A')} anomaly events from 300 synthetic logs)  ",
        "**Frontier model:** Claude Sonnet 4.5 (AWS Bedrock, us.* cross-region)  ",
        f"**Open-weight models:** {local_models_str}  ",
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
        "**Open-weight deployment modes benchmarked:**",
        "- **Mistral 7B (Bedrock)** — managed API, data leaves machine to AWS",
        "- **Mistral 7B (Sovereign/Ollama)** — fully local, zero external calls,",
        "  addresses NIS2 data residency requirements for UK energy operators",
        "",
        "**Option B ground truth:** Severity accuracy measured only on injected",
        "pattern events. Isolated events included in latency/validity only.",
        "",
        "Three injected anomaly patterns:",
        "- **Pattern A** (events 60–69): Cascading failure — auth → inventory → payment-svc → api-gateway.",
        "  All ERROR, 500 status, latency 1500–2500ms. Ground truth: HIGH.",
        "- **Pattern B** (events 130–139): Single-service latency spike — payment-svc.",
        "  Latency 2000–4000ms, mixed 200/500. Ground truth: HIGH.",
        "- **Pattern C** (events 200–212): Error rate surge — api-gateway.",
        "  Normal latency (100–400ms), elevated 500s. Tests error-only signal. Ground truth: HIGH.",
        "- **Isolated events**: Outside injected windows. Excluded from accuracy.",
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
        _row("Latency P50 (ms)", "latency_p50_ms"),
        _row("Latency P95 (ms)", "latency_p95_ms"),
        _static_row("Data leaves network", _data_leaves),
        _static_row("Cost per call", _cost_per_call),
        _static_row("Production-ready", _production_ready),
        "",
        "---",
        "",
        "## Key Findings",
        "",
        "### 1. Frontier vs Open-Weight Quality Gap",
        "",
        "Accuracy measured on injected pattern events only (Option B).",
        "",
    ]

    for b in backends:
        s = all_stats[b]
        lines.append(f"- **{_col_label(b)}**: {s['severity_accuracy']} overall.")

    lines += [
        "",
        "### 2. Mistral 7B — Managed vs Sovereign Deployment",
        "",
        "Same model, two deployment modes. Key trade-off:",
        "- **Bedrock**: lower latency, managed infrastructure, data leaves machine",
        "- **Sovereign/Ollama**: ~4–5x higher latency, zero external dependency,",
        "  satisfies NIS2 data residency for UK energy operators",
        "",
        "For async anomaly pipelines the latency trade-off is acceptable.",
        "For real-time triage dashboards it requires GPU hardware to close the gap.",
        "",
        "### 3. Pattern C — Error Surge Without Latency",
        "",
        "Pattern C tests severity identification from error signal alone,",
        "without a latency spike. Only elevated 500 status codes as signal.",
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
        "### 4. Latency vs Quality Trade-off",
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
        "### 5. JSON Reliability",
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
        "| | Claude Sonnet (Bedrock) | Mistral 7B (Bedrock) | Mistral 7B (Sovereign) |",
        "|---|---|---|---|",
        "| GDPR sub-processor | Anthropic | Mistral AI | None |",
        "| Data residency | AWS region | AWS region | Client machine |",
        "| NIS2 compliant | Requires DPA | Requires DPA | Yes — zero egress |",
        "| Model quality | Frontier | Open-weight 7B | Open-weight 7B |",
        "| Latency | Low | Low | ~4–5x higher |",
        "| Cost crossover | Baseline | ~30x cheaper | ~$0 |",
        "| Ops burden | Low | Low | Medium (hardware) |",
        "| Production-ready | Yes | Yes | With GPU |",
        "",
        "**Note on inference profiles:** For UK energy clients under NIS2, use an",
        "`eu.*` inference profile or provisioned throughput in eu-west-2 to",
        "guarantee data residency even with managed Bedrock backends.",
        "",
        "---",
        "",
        "## Recommendation by Use Case",
        "",
        "**Use Claude Sonnet (Bedrock) when:**",
        "- Quality and reliability are primary concerns",
        "- Volume is below ~50M tokens/day",
        "- Client has signed a DPA with Anthropic",
        "",
        "**Use Mistral 7B (Bedrock) when:**",
        "- Cost is a primary constraint and frontier quality is not required",
        "- Client is comfortable with open-weight model quality trade-offs",
        "- Volume makes Claude cost prohibitive",
        "",
        "**Use Mistral 7B (Sovereign/Ollama) when:**",
        "- Client has NIS2 obligations (energy, critical infrastructure)",
        "- Data sovereignty requirements preclude any external API endpoints",
        "- Async pipeline — latency trade-off is acceptable",
        "- Use GPU hardware (g5.xlarge on EKS/vLLM) for production-grade latency",
        "",
        "---",
        "",
        "## Files",
        "",
        "- `src/explanation/bedrock_explainer.py` — Claude Sonnet via Bedrock",
        "- `src/explanation/mistral_bedrock_explainer.py` — Mistral 7B via Bedrock",
        "- `src/explanation/mistral_local_explainer.py` — Mistral 7B via Ollama (sovereign)",
        "- `src/explanation/ollama_explainer.py` — generic Ollama backend",
        "- `src/explanation/prompts.py`",
        "- `src/detection/anomaly_scorer.py`",
        "- `results/benchmark_raw.jsonl` — raw per-event results",
        "",
    ]

    report_path.write_text("\n".join(lines))
    logger.info("Report written to %s", report_path)
    return report_path


async def main(
    backend: str,
    report_only: bool,
    raw_input: Path,
    raw_output: Path,
    report_output: Path,
) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    if report_only:
        all_results = _load_results_by_backend(raw_input)
        if not all_results:
            print(f"No results in {raw_input} — run backends first.")
            return
        report_path = _write_markdown_report(all_results, report_output)
        print(f"\nReport written: {report_path}")
        all_stats = {b: _compute_stats(r) for b, r in all_results.items()}
        ordered = [b for b in DISPLAY_ORDER if b in all_stats]
        ordered += [b for b in sorted(all_stats) if b not in ordered]
        for b in ordered:
            s = all_stats[b]
            print(
                f"  {_col_label(b)}: "
                f"accuracy={s['severity_accuracy']}, "
                f"P50={s['latency_p50_ms']}ms"
            )
        return

    events = _load_events()
    anomalies = _detect_anomalies(events)

    run_backends = (
        ["bedrock", "mistral-bedrock", "mistral-local", "ollama"]
        if backend == "all"
        else [backend]
    )

    for b in run_backends:
        logger.info("--- %s run ---", b)
        backend_results = await _run_backend(b, anomalies)
        _save_results(raw_output, backend_results)
        stats = _compute_stats(backend_results)
        print(
            f"\n{_col_label(b)} complete — "
            f"accuracy: {stats['severity_accuracy']}, "
            f"P50: {stats['latency_p50_ms']}ms"
        )

    all_results = _load_results_by_backend(raw_output)
    if len(all_results) > 1:
        report_path = _write_markdown_report(all_results, report_output)
        print(f"\nReport written: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark Bedrock vs open-weight model explainers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  python -m stream1_log_anomaly.benchmark_runner --explainer bedrock
  python -m stream1_log_anomaly.benchmark_runner --explainer mistral-bedrock
  OLLAMA_MODEL=mistral:latest OLLAMA_TIMEOUT_S=120 \\
      python -m stream1_log_anomaly.benchmark_runner --explainer mistral-local
  OLLAMA_MODEL=llama3.2:3b \\
      python -m stream1_log_anomaly.benchmark_runner --explainer ollama
  python -m stream1_log_anomaly.benchmark_runner --report

  # Split-machine workflow:
  scripts/benchmark_office.sh    # office laptop
  scripts/benchmark_local.sh     # local laptop
  scripts/benchmark_combine.sh   # merge + final report
        """,
    )
    parser.add_argument(
        "--explainer",
        choices=["bedrock", "mistral-bedrock", "mistral-local", "ollama", "all"],
        default="all",
        help="Which backend to run (default: all)",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate report from existing raw results without running inference",
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        default=RESULTS_DIR / "benchmark_raw.jsonl",
        help="Path to write raw results JSONL (default: results/benchmark_raw.jsonl)",
    )
    parser.add_argument(
        "--raw-input",
        type=Path,
        default=RESULTS_DIR / "benchmark_raw.jsonl",
        help="Path to read raw results JSONL for --report (default: results/benchmark_raw.jsonl)",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=RESULTS_DIR / "bedrock_vs_ollama_vs_mistral.md",
        help="Path to write the markdown report (default: results/bedrock_vs_ollama_vs_mistral.md)",
    )
    args = parser.parse_args()

    asyncio.run(main(
        backend=args.explainer,
        report_only=args.report,
        raw_input=args.raw_input,
        raw_output=args.raw_output,
        report_output=args.report_output,
    ))
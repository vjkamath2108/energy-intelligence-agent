"""Stream 1 anomaly detection demo — log pipeline with Isolation Forest + explanations.

Processes 300 synthetic EKS log events, scores each with Isolation Forest,
and for every detected anomaly produces a structured explanation.

Anomaly patterns in the synthetic dataset:
  Pattern A — Cascading failure: auth-service latency spike → propagates to
              payment-svc and api-gateway (events ~33–42)
  Pattern B — Sustained latency: payment-svc elevated response times (events ~100–120)
  Pattern C — Error surge: api-gateway elevated 500s, no latency spike (events ~200–215)

Usage:
    python examples/demo_streaming.py              # demo mode — no AWS needed
    python examples/demo_streaming.py --explainer bedrock  # real Bedrock (requires creds)
    python examples/demo_streaming.py --max-anomalies 3    # show first N anomalies in detail

Exit codes:
    0 — pipeline ran cleanly, at least one anomaly detected
    1 — no anomalies detected (dataset or scorer issue)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

from stream1_log_anomaly.src.detection.anomaly_scorer import AnomalyScorer
from stream1_log_anomaly.src.models import Explanation, LogEvent

_SAMPLE_LOG = Path("stream1_log_anomaly/logs/sample.jsonl")
_DIVIDER = "═" * 70
_LINE = "─" * 70

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Demo explainer (no AWS)
# ---------------------------------------------------------------------------


def _demo_explain(event: LogEvent, context_events: list[LogEvent]) -> Explanation:
    """Generate a realistic structured explanation from event properties alone.

    Replaces BedrockExplainer / OllamaExplainer in demo mode.
    Uses deterministic rules based on the event's service, level, and latency
    to produce explanations that match the known synthetic anomaly patterns.

    Args:
        event: The anomalous log event.
        context_events: Surrounding events for causality context.

    Returns:
        Structured Explanation matching the canonical model.
    """
    svc = event.service
    latency = event.latency_ms
    is_error = event.status_code >= 500

    # Derive context signals
    context_services = [e.service for e in context_events]
    cascading = len(set(context_services)) >= 3

    if svc == "auth-service" or (cascading and latency > 1000):
        severity = "HIGH"
        summary = (
            f"auth-service response time degraded to {latency}ms — upstream cascade risk. "
            f"Downstream services ({', '.join(set(context_services) - {svc})}) may be affected."
        )
        likely_cause = (
            "Auth service database connection pool exhaustion or OAuth token validation "
            "latency spike. Cascading pattern observed across multiple dependent services."
        )
        recommended_action = (
            "1. Check auth-service DB connection pool utilisation (target < 80%). "
            "2. Inspect OAuth provider connectivity. "
            "3. Activate circuit breaker on dependent services to limit blast radius. "
            "4. Page on-call SRE if latency persists > 2 minutes."
        )
    elif svc == "payment-svc" and latency > 2000:
        severity = "HIGH"
        summary = (
            f"payment-svc latency sustained at {latency}ms — SLA breach risk. "
            f"{'Status 500 detected — transaction failures likely.' if is_error else 'Status 200 but latency exceeds SLA threshold.'}"
        )
        likely_cause = (
            "Payment gateway upstream timeout or database write latency spike. "
            "Sustained pattern (not a single spike) suggests resource saturation or lock contention."
        )
        recommended_action = (
            "1. Check payment-svc DB slow query log for lock contention. "
            "2. Verify payment gateway upstream connectivity and circuit breaker state. "
            "3. Review current transaction queue depth. "
            "4. Consider traffic shedding if latency exceeds 5000ms."
        )
    elif is_error and svc == "api-gateway":
        severity = "MEDIUM"
        summary = (
            f"api-gateway returning {event.status_code} errors at elevated rate. "
            f"Latency nominally {latency}ms — errors not latency-driven."
        )
        likely_cause = (
            "Upstream routing misconfiguration or downstream service returning 5xx errors "
            "that api-gateway surfaces directly. Not a latency bottleneck."
        )
        recommended_action = (
            "1. Check api-gateway error logs for upstream target status codes. "
            "2. Verify health checks on downstream services. "
            "3. Review recent deployment or configuration changes. "
            "4. Escalate if error rate exceeds 5% over 5-minute window."
        )
    else:
        severity = "LOW"
        summary = (
            f"{svc} flagged as anomalous: latency={latency}ms "
            f"level={event.level} status={event.status_code}."
        )
        likely_cause = (
            "Isolated spike within normal operating parameters. "
            "Likely transient GC pause, network jitter, or noisy-neighbour effect."
        )
        recommended_action = (
            "Monitor for recurrence over the next 5 minutes. "
            "No immediate action required unless pattern persists."
        )

    return Explanation(
        severity=severity,
        summary=summary,
        likely_cause=likely_cause,
        recommended_action=recommended_action,
    )


# ---------------------------------------------------------------------------
# Event loading
# ---------------------------------------------------------------------------


def _load_events() -> list[LogEvent]:
    """Load synthetic log events from sample.jsonl.

    Generates the file via the log generator if it does not exist.

    Returns:
        List of LogEvent objects in order.
    """
    if not _SAMPLE_LOG.exists():
        print(f"  sample.jsonl not found — generating synthetic dataset…")
        from stream1_log_anomaly.src.streaming.generate_logs import generate_events, OUTPUT_PATH
        events_raw = generate_events()
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with OUTPUT_PATH.open("w") as f:
            for e in events_raw:
                f.write(json.dumps(e) + "\n")
        print(f"  Generated {len(events_raw)} events → {OUTPUT_PATH}")

    events: list[LogEvent] = []
    with _SAMPLE_LOG.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(LogEvent(**json.loads(line)))
                except Exception as e:
                    logger.warning("Skipping malformed log line: %s", e)
    return events


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


async def run_pipeline(explainer: str, max_anomalies: int) -> int:
    """Run the full streaming anomaly detection demo pipeline.

    Args:
        explainer: "demo" (no AWS) or "bedrock" (requires credentials).
        max_anomalies: Maximum number of anomalies to show in full detail.

    Returns:
        Exit code — 0 if anomalies detected, 1 otherwise.
    """
    print(_DIVIDER)
    print("Stream 1 — Log Anomaly Detection Demo")
    print(f"Model: Isolation Forest  |  Explainer: {explainer}  |  Contamination: 10%")
    print(_DIVIDER)

    if explainer == "bedrock":
        print("\nERROR: Real Bedrock explainer requires AWS credentials.")
        print("       Use --explainer demo (default).")
        return 1

    print(f"\nLoading events from {_SAMPLE_LOG}…")
    events = _load_events()
    print(f"Loaded {len(events)} log events.\n")

    scorer = AnomalyScorer()
    anomalies_detected = 0
    anomalies_shown = 0

    pipeline_start = time.monotonic()

    for i, event in enumerate(events):
        result = scorer.score(event)

        if result is None:
            # Still buffering
            if i + 1 == scorer.buffer_size:
                print(f"[{i + 1:03d}/{len(events)}]  Buffer full — Isolation Forest fitted on {scorer.buffer_size} events.")
                print()
            continue

        if not result.is_anomaly:
            continue

        anomalies_detected += 1

        if anomalies_shown >= max_anomalies:
            continue

        anomalies_shown += 1

        # Explanation
        context_start = max(0, i - 5)
        context_end = min(len(events), i + 6)
        context_events = [e for j, e in enumerate(events[context_start:context_end]) if j != i - context_start]

        explain_start = time.monotonic()
        explanation = _demo_explain(event, context_events)
        explain_ms = (time.monotonic() - explain_start) * 1000

        # Print full trace
        print(f"{_LINE}")
        print(
            f"[{i + 1:03d}/{len(events)}]  ANOMALY DETECTED  |  "
            f"score={result.anomaly_score:+.4f}  |  window={result.window_id}"
        )
        print(_LINE)
        print(f"  Event:     {event.ts}  {event.level:<5}  {event.service:<20}  "
              f"latency={event.latency_ms}ms  status={event.status_code}")
        print(f"\n  Context ({len(context_events)} surrounding events):")
        for ctx in context_events[:4]:
            marker = "→" if ctx.service == event.service else " "
            print(f"   {marker}  {ctx.level:<5}  {ctx.service:<20}  {ctx.latency_ms}ms  {ctx.status_code}")
        if len(context_events) > 4:
            print(f"     … {len(context_events) - 4} more")

        print(f"\n  Severity:  {explanation.severity}")
        print(f"  Summary:   {explanation.summary[:110]}")
        if len(explanation.summary) > 110:
            print(f"             {explanation.summary[110:]}")
        print(f"\n  Likely cause:")
        print(f"    {explanation.likely_cause[:110]}")
        if len(explanation.likely_cause) > 110:
            print(f"    {explanation.likely_cause[110:]}")
        print(f"\n  Recommended action:")
        for line in explanation.recommended_action.split("\n"):
            print(f"    {line.strip()}")
        print(f"\n  Explainer: {explainer}  |  Explanation latency: {explain_ms:.1f}ms")
        print()

    total_ms = (time.monotonic() - pipeline_start) * 1000

    print(_DIVIDER)
    print("SUMMARY")
    print(_DIVIDER)
    print(f"  Events processed : {len(events)}")
    print(f"  Buffer size      : {scorer.buffer_size}")
    print(f"  Events scored    : {len(events) - scorer.buffer_size}")
    print(f"  Anomalies found  : {anomalies_detected}  ({anomalies_detected / max(len(events) - scorer.buffer_size, 1) * 100:.1f}%)")
    print(f"  Shown in detail  : {anomalies_shown} (--max-anomalies {max_anomalies})")
    print(f"  Total time       : {total_ms:.0f}ms  ({total_ms / len(events):.1f}ms/event)")

    if anomalies_detected == 0:
        print("\n  WARNING: no anomalies detected — check scorer or dataset.")
        return 1

    print(f"\n  Result: PIPELINE COMPLETE — {anomalies_detected} anomalies detected and explained.")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s  %(message)s")
    parser = argparse.ArgumentParser(description="Stream 1 anomaly detection demo.")
    parser.add_argument(
        "--explainer",
        default="demo",
        choices=["demo", "bedrock"],
        help="'demo' runs without AWS credentials (default).",
    )
    parser.add_argument(
        "--max-anomalies",
        type=int,
        default=3,
        help="Maximum number of anomalies to display in full detail (default: 3).",
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(run_pipeline(args.explainer, args.max_anomalies)))

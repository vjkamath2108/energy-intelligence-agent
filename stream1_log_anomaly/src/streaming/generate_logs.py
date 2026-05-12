"""
Synthetic log generator for Stream 1 benchmark dataset.

Generates 300 events with three distinct anomaly patterns:
  - Pattern A: Cascading failure (auth → inventory → payment-svc → api-gateway)
  - Pattern B: Single-service latency spike (payment-svc sustained high latency)
  - Pattern C: Error rate surge (api-gateway elevated 500s, no latency spike)

Design decisions:
- Fixed random seed for reproducibility — benchmark runs on identical data.
- Anomaly patterns are deterministic (fixed event windows), not random injection.
- Service sequence in cascades is explicit, not random.choice() — matches
  realistic propagation order for an auth-dependent microservice stack.
- Normal baseline: 70% INFO, 20% WARN, 10% ERROR; latency 10–300ms; 90% 200s.
- Anomaly windows tuned so Isolation Forest flags ~30 events across 300
  (contamination ~10%), matching the benchmark spec.

Usage:
    python -m stream1_log_anomaly.src.streaming.log_generator
    # writes to stream1_log_anomaly/logs/sample.jsonl
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

SEED = 42  # fixed for reproducibility
OUTPUT_PATH = Path("stream1_log_anomaly/logs/sample.jsonl")

SERVICES = ["api-gateway", "auth-service", "payment-svc", "inventory", "order-svc"]
LEVELS_NORMAL = ["INFO"] * 14 + ["WARN"] * 4 + ["ERROR"] * 2


def _normal_event(service: str, ts: datetime, rng: random.Random) -> dict:
    """Generate a single normal baseline event."""
    return {
        "ts": ts.isoformat(),
        "service": service,
        "level": rng.choice(LEVELS_NORMAL),
        "latency_ms": rng.randint(10, 300),
        "status_code": rng.choice([200] * 9 + [500] * 1),
    }


def generate_events(n: int = 300, seed: int = SEED) -> list[dict]:
    """
    Generate n synthetic log events with injected anomaly patterns.

    Anomaly windows (deterministic):
      Pattern A — Cascading failure: events 60–69
        auth-service fails first, propagates to inventory, payment-svc,
        api-gateway in sequence. All ERROR, 500, latency 1500–2500ms.
      Pattern B — Latency spike: events 130–139
        payment-svc sustained high latency (2000–4000ms) but 200 status.
        Single service, no cascade. Tests latency-only anomaly detection.
      Pattern C — Error surge: events 200–212
        api-gateway elevated error rate (500s) without latency spike.
        Latency stays normal (100–400ms). Tests error-rate-only detection.

    Args:
        n: Total number of events to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of event dicts ready to serialise as JSONL.
    """
    rng = random.Random(seed)
    base_time = datetime(2026, 5, 1, 8, 0, 0)
    events = []

    # Cascade service order — explicit propagation sequence
    cascade_services = [
        "auth-service",
        "auth-service",
        "inventory",
        "inventory",
        "payment-svc",
        "payment-svc",
        "api-gateway",
        "api-gateway",
        "api-gateway",
        "api-gateway",
    ]

    for i in range(n):
        ts = base_time + timedelta(seconds=i)
        service = rng.choice(SERVICES)

        # Pattern A: Cascading failure (events 60–69)
        if 60 <= i <= 69:
            event = {
                "ts": ts.isoformat(),
                "service": cascade_services[i - 60],
                "level": "ERROR",
                "latency_ms": rng.randint(1500, 2500),
                "status_code": 500,
            }

        # Pattern B: Single-service latency spike (events 130–139)
        elif 130 <= i <= 139:
            event = {
                "ts": ts.isoformat(),
                "service": "payment-svc",
                "level": rng.choice(["WARN"] * 3 + ["ERROR"] * 7),
                "latency_ms": rng.randint(2000, 4000),
                "status_code": rng.choice([200] * 6 + [500] * 4),
            }

        # Pattern C: Error rate surge, no latency spike (events 200–212)
        elif 200 <= i <= 212:
            event = {
                "ts": ts.isoformat(),
                "service": "api-gateway",
                "level": rng.choice(["WARN"] * 2 + ["ERROR"] * 8),
                "latency_ms": rng.randint(100, 400),  # latency stays normal
                "status_code": rng.choice([500] * 8 + [200] * 2),
            }

        # Normal baseline
        else:
            event = _normal_event(service, ts, rng)

        events.append(event)

    return events


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    events = generate_events()

    with OUTPUT_PATH.open("w") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")

    # Summary for verification
    anomaly_windows = {
        "Pattern A (cascade, events 60–69)": sum(
            1 for i, e in enumerate(events) if 60 <= i <= 69
        ),
        "Pattern B (latency spike, events 130–139)": sum(
            1 for i, e in enumerate(events) if 130 <= i <= 139
        ),
        "Pattern C (error surge, events 200–212)": sum(
            1 for i, e in enumerate(events) if 200 <= i <= 212
        ),
    }

    print(f"Generated {len(events)} events → {OUTPUT_PATH}")
    print(f"Seed: {SEED} (fixed — reruns produce identical data)")
    print()
    for pattern, count in anomaly_windows.items():
        print(f"  {pattern}: {count} events injected")
    print()
    print("Run the pipeline to verify Isolation Forest detection rate:")
    print("  python -m stream1_log_anomaly.main --explainer ollama")


if __name__ == "__main__":
    main()
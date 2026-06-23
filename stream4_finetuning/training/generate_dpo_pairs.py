"""Generate DPO preference pairs from Stream 1 benchmark outputs.

The pipeline produces its own training signal: when multiple models explain
the same anomaly, pairs where a high-accuracy model (chosen) is correct and a
low-accuracy model (rejected) is wrong become DPO preference pairs for fine-tuning.

Source data:
  results/benchmark_raw.jsonl      — per-event model outputs and accuracy labels
  stream1_log_anomaly/logs/sample.jsonl  — original log events for prompt reconstruction

Output:
  stream4_finetuning/data/dpo_pairs.jsonl  — TRL DPOTrainer-compatible pairs

DPO pair selection criteria:
  chosen  = mistral-bedrock (100% severity accuracy on injected patterns)
  rejected = ollama:llama3.2:3b (6% severity accuracy on injected patterns)
  filter  = both severity_correct=True (chosen) and severity_correct=False (rejected)
             on injected anomaly patterns (not isolated events)

Usage:
    python -m stream4_finetuning.training.generate_dpo_pairs
    python -m stream4_finetuning.training.generate_dpo_pairs \\
        --benchmark results/benchmark_raw.jsonl \\
        --logs stream1_log_anomaly/logs/sample.jsonl \\
        --output stream4_finetuning/data/dpo_pairs.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_CHOSEN_BACKEND = "mistral-bedrock"
_REJECTED_BACKEND = "ollama:llama3.2:3b"

_SYSTEM_PROMPT = (
    "You are an expert site reliability engineer analysing Kubernetes log anomalies. "
    "Given an anomalous log event and its surrounding context, produce a structured "
    "incident explanation as a JSON object with four fields: severity (HIGH, MEDIUM, "
    "or LOW), summary (one sentence describing what happened), likely_cause (the most "
    "probable root cause), and recommended_action (what an engineer should do next). "
    "Respond ONLY with valid JSON. No markdown, no explanation, no preamble."
)


@dataclass(frozen=True)
class DPOPair:
    """A single DPO preference pair for fine-tuning.

    Fields match the TRL DPOTrainer dataset schema:
    https://huggingface.co/docs/trl/dpo_trainer#expected-dataset-format
    """

    prompt: str
    chosen: str
    rejected: str
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Serialise to a dict for JSONL output."""
        return {
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "metadata": self.metadata,
        }


def _build_prompt(
    anomalous_event: dict[str, object],
    context_events: list[dict[str, object]],
) -> str:
    """Reconstruct the inference prompt from log event dicts.

    Mirrors the logic in stream1_log_anomaly/src/explanation/prompts.py
    so that fine-tuned outputs are compatible with the production pipeline.

    Args:
        anomalous_event: The event flagged as anomalous.
        context_events: Surrounding ±5 events for causality context.

    Returns:
        Formatted prompt string.
    """
    context_str = json.dumps(context_events, indent=2)
    event_str = json.dumps(anomalous_event, indent=2)

    return (
        f"Surrounding context events (±5 around the anomaly):\n{context_str}\n\n"
        f"Anomalous event flagged by Isolation Forest:\n{event_str}\n\n"
        f'Respond ONLY with a JSON object starting with {{ and ending with }}.'
    )


def _build_completion(record: dict[str, object]) -> str:
    """Build the completion JSON string from a benchmark record.

    Args:
        record: A row from benchmark_raw.jsonl.

    Returns:
        JSON string matching the Explanation schema.
    """
    return json.dumps(
        {
            "severity": record["predicted_severity"],
            "summary": record["summary"],
            "likely_cause": record["likely_cause"],
            "recommended_action": record["recommended_action"],
        },
        ensure_ascii=False,
    )


def load_log_events(logs_path: Path) -> list[dict[str, object]]:
    """Load all log events from a JSONL file.

    Args:
        logs_path: Path to the JSONL log file.

    Returns:
        List of log event dicts in order.
    """
    events: list[dict[str, object]] = []
    with logs_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    logger.info("Loaded %d log events from %s", len(events), logs_path)
    return events


def load_benchmark_records(benchmark_path: Path) -> dict[str, dict[int, dict[str, object]]]:
    """Load benchmark records grouped by backend and event index.

    Args:
        benchmark_path: Path to benchmark_raw.jsonl.

    Returns:
        Nested dict: backend → event_index → record.
    """
    records: dict[str, dict[int, dict[str, object]]] = {}
    with benchmark_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r: dict[str, object] = json.loads(line)
            backend = str(r["backend"])
            idx = int(str(r["event_index"]))
            records.setdefault(backend, {})[idx] = r

    for backend, events in records.items():
        logger.info("Loaded %d records for backend '%s'", len(events), backend)
    return records


def generate_dpo_pairs(
    log_events: list[dict[str, object]],
    records: dict[str, dict[int, dict[str, object]]],
    chosen_backend: str = _CHOSEN_BACKEND,
    rejected_backend: str = _REJECTED_BACKEND,
    context_window: int = 5,
) -> list[DPOPair]:
    """Generate DPO preference pairs from benchmark outputs.

    Selects events where chosen_backend is correct and rejected_backend is wrong
    on injected anomaly patterns (excludes isolated events).

    Args:
        log_events: Original log events for prompt reconstruction.
        records: Benchmark records grouped by backend and event index.
        chosen_backend: Backend whose outputs become the preferred (chosen) response.
        rejected_backend: Backend whose outputs become the dispreferred (rejected) response.
        context_window: Number of events on each side of the anomaly for context.

    Returns:
        List of DPOPair instances ready for JSONL serialisation.
    """
    chosen_records = records.get(chosen_backend, {})
    rejected_records = records.get(rejected_backend, {})

    if not chosen_records:
        raise ValueError(f"No records found for chosen backend '{chosen_backend}'")
    if not rejected_records:
        raise ValueError(f"No records found for rejected backend '{rejected_backend}'")

    pairs: list[DPOPair] = []

    for idx in sorted(chosen_records):
        chosen_rec = chosen_records[idx]
        rejected_rec = rejected_records.get(idx)

        if rejected_rec is None:
            continue

        if chosen_rec.get("anomaly_pattern") == "isolated":
            continue

        chosen_correct = bool(chosen_rec.get("severity_correct"))
        rejected_correct = bool(rejected_rec.get("severity_correct"))

        if not chosen_correct or rejected_correct:
            continue

        context_start = max(0, idx - context_window)
        context_end = min(len(log_events), idx + context_window + 1)
        context = [e for i, e in enumerate(log_events[context_start:context_end], context_start) if i != idx]
        anomalous = log_events[idx] if idx < len(log_events) else {}

        prompt = _build_prompt(anomalous, context)
        chosen_completion = _build_completion(chosen_rec)
        rejected_completion = _build_completion(rejected_rec)

        pair = DPOPair(
            prompt=prompt,
            chosen=chosen_completion,
            rejected=rejected_completion,
            metadata={
                "event_index": idx,
                "anomaly_pattern": str(chosen_rec.get("anomaly_pattern", "")),
                "ground_truth_severity": chosen_rec.get("ground_truth_severity"),
                "chosen_backend": chosen_backend,
                "rejected_backend": rejected_backend,
                "source": "stream1_benchmark_raw",
            },
        )
        pairs.append(pair)
        logger.debug(
            "DPO pair %d — pattern=%s chosen_severity=%s rejected_severity=%s",
            idx,
            chosen_rec.get("anomaly_pattern"),
            chosen_rec.get("predicted_severity"),
            rejected_rec.get("predicted_severity"),
        )

    logger.info(
        "Generated %d DPO pairs (chosen=%s, rejected=%s)",
        len(pairs),
        chosen_backend,
        rejected_backend,
    )
    return pairs


def write_dpo_pairs(pairs: list[DPOPair], output_path: Path) -> None:
    """Write DPO pairs to a JSONL file.

    Args:
        pairs: List of DPOPair instances to write.
        output_path: Destination path; parent directory is created if needed.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for pair in pairs:
            f.write(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")
    logger.info("Wrote %d DPO pairs to %s", len(pairs), output_path)


def main() -> None:
    """CLI entry point for DPO pair generation."""
    parser = argparse.ArgumentParser(
        description="Generate DPO preference pairs from Stream 1 benchmark outputs"
    )
    parser.add_argument(
        "--benchmark",
        default="results/benchmark_raw.jsonl",
        help="Path to benchmark_raw.jsonl",
    )
    parser.add_argument(
        "--logs",
        default="stream1_log_anomaly/logs/sample.jsonl",
        help="Path to original log events JSONL",
    )
    parser.add_argument(
        "--output",
        default="stream4_finetuning/data/dpo_pairs.jsonl",
        help="Output path for DPO pairs JSONL",
    )
    parser.add_argument(
        "--chosen-backend",
        default=_CHOSEN_BACKEND,
        help=f"Backend to use as 'chosen' (default: {_CHOSEN_BACKEND})",
    )
    parser.add_argument(
        "--rejected-backend",
        default=_REJECTED_BACKEND,
        help=f"Backend to use as 'rejected' (default: {_REJECTED_BACKEND})",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s  %(message)s")

    log_events = load_log_events(Path(args.logs))
    records = load_benchmark_records(Path(args.benchmark))
    pairs = generate_dpo_pairs(
        log_events, records, args.chosen_backend, args.rejected_backend
    )

    if not pairs:
        logger.warning("No DPO pairs generated — check backend names and filter criteria")
        return

    write_dpo_pairs(pairs, Path(args.output))
    print(f"\nGenerated {len(pairs)} DPO pairs → {args.output}")
    print(f"  Chosen  : {args.chosen_backend}")
    print(f"  Rejected: {args.rejected_backend}")


if __name__ == "__main__":
    main()

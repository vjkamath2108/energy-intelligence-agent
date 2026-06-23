"""Generate SFT (supervised fine-tuning) dataset from high-quality benchmark outputs.

Takes the outputs from the best-performing model (mistral-bedrock, 100% accuracy)
on injected anomaly patterns and formats them as instruction-following pairs
for SFTTrainer.

The SFT phase teaches the base model the task format and domain vocabulary before
DPO alignment. Using the pipeline's own production outputs (rather than
human-annotated gold labels) means the SFT and DPO datasets stay consistent.

Source data:
  results/benchmark_raw.jsonl      — per-event model outputs and accuracy labels
  stream1_log_anomaly/logs/sample.jsonl  — original log events for prompt reconstruction

Output:
  stream4_finetuning/data/sft_dataset.jsonl  — Alpaca-format instruction-following pairs

Dataset schema (Alpaca format, compatible with trl.SFTTrainer):
    {
        "instruction": "<system prompt>",
        "input": "<user prompt with event + context>",
        "output": "<JSON explanation>",
        "metadata": { ... }
    }

Usage:
    python -m stream4_finetuning.training.generate_sft_dataset
    python -m stream4_finetuning.training.generate_sft_dataset \\
        --benchmark results/benchmark_raw.jsonl \\
        --logs stream1_log_anomaly/logs/sample.jsonl \\
        --output stream4_finetuning/data/sft_dataset.jsonl \\
        --include-isolated
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_SOURCE_BACKEND = "mistral-bedrock"

_INSTRUCTION = (
    "You are an expert site reliability engineer analysing Kubernetes log anomalies. "
    "Given an anomalous log event and its surrounding context, produce a structured "
    "incident explanation as a JSON object with four fields: severity (HIGH, MEDIUM, "
    "or LOW), summary (one sentence describing what happened), likely_cause (the most "
    "probable root cause), and recommended_action (what an engineer should do next). "
    "Respond ONLY with valid JSON. No markdown, no explanation, no preamble."
)


@dataclass(frozen=True)
class SFTExample:
    """A single supervised fine-tuning example.

    Follows the Alpaca instruction-following format:
    https://github.com/tatsu-lab/stanford_alpaca
    """

    instruction: str
    input: str
    output: str
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Serialise to a dict for JSONL output."""
        return {
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output,
            "metadata": self.metadata,
        }


def _build_input(
    anomalous_event: dict[str, object],
    context_events: list[dict[str, object]],
) -> str:
    """Build the user input string from log event dicts.

    Args:
        anomalous_event: The event flagged as anomalous.
        context_events: Surrounding ±5 events for causality context.

    Returns:
        Formatted input string (without the system instruction).
    """
    context_str = json.dumps(context_events, indent=2)
    event_str = json.dumps(anomalous_event, indent=2)

    return (
        f"Surrounding context events (±5 around the anomaly):\n{context_str}\n\n"
        f"Anomalous event flagged by Isolation Forest:\n{event_str}"
    )


def _build_output(record: dict[str, object]) -> str:
    """Build the target output JSON string from a benchmark record.

    Args:
        record: A row from benchmark_raw.jsonl with summary/cause/action fields.

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


def generate_sft_dataset(
    log_events: list[dict[str, object]],
    records: dict[str, dict[int, dict[str, object]]],
    source_backend: str = _SOURCE_BACKEND,
    include_isolated: bool = False,
    context_window: int = 5,
) -> list[SFTExample]:
    """Generate SFT instruction-following examples from benchmark outputs.

    Selects outputs from source_backend where the severity label is correct
    (on injected pattern events). Optionally includes isolated events where
    no ground-truth label is available (treated as unlabelled training data).

    Args:
        log_events: Original log events for prompt reconstruction.
        records: Benchmark records grouped by backend → event_index → record.
        source_backend: Backend whose outputs to use as training targets.
        include_isolated: Whether to include isolated events without ground truth.
        context_window: Number of events on each side of the anomaly for context.

    Returns:
        List of SFTExample instances ready for JSONL serialisation.
    """
    source_records = records.get(source_backend, {})
    if not source_records:
        raise ValueError(f"No records found for backend '{source_backend}'")

    examples: list[SFTExample] = []

    for idx in sorted(source_records):
        record = source_records[idx]
        is_isolated = record.get("anomaly_pattern") == "isolated"

        if is_isolated and not include_isolated:
            continue

        if not is_isolated and not bool(record.get("severity_correct")):
            continue

        context_start = max(0, idx - context_window)
        context_end = min(len(log_events), idx + context_window + 1)
        context = [
            e
            for i, e in enumerate(log_events[context_start:context_end], context_start)
            if i != idx
        ]
        anomalous = log_events[idx] if idx < len(log_events) else {}

        example = SFTExample(
            instruction=_INSTRUCTION,
            input=_build_input(anomalous, context),
            output=_build_output(record),
            metadata={
                "event_index": idx,
                "anomaly_pattern": str(record.get("anomaly_pattern", "")),
                "ground_truth_severity": record.get("ground_truth_severity"),
                "severity_correct": record.get("severity_correct"),
                "source_backend": source_backend,
            },
        )
        examples.append(example)

    logger.info(
        "Generated %d SFT examples from backend '%s' (include_isolated=%s)",
        len(examples),
        source_backend,
        include_isolated,
    )
    return examples


def write_sft_dataset(examples: list[SFTExample], output_path: Path) -> None:
    """Write SFT examples to a JSONL file.

    Args:
        examples: List of SFTExample instances to write.
        output_path: Destination path; parent directory is created if needed.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for example in examples:
            f.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")
    logger.info("Wrote %d SFT examples to %s", len(examples), output_path)


def main() -> None:
    """CLI entry point for SFT dataset generation."""
    parser = argparse.ArgumentParser(
        description="Generate SFT dataset from high-quality Stream 1 benchmark outputs"
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
        default="stream4_finetuning/data/sft_dataset.jsonl",
        help="Output path for SFT dataset JSONL",
    )
    parser.add_argument(
        "--source-backend",
        default=_SOURCE_BACKEND,
        help=f"Backend to use as training signal source (default: {_SOURCE_BACKEND})",
    )
    parser.add_argument(
        "--include-isolated",
        action="store_true",
        help="Include isolated events without ground-truth severity labels",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s  %(message)s")

    from stream4_finetuning.training.generate_dpo_pairs import (
        load_benchmark_records,
        load_log_events,
    )

    log_events = load_log_events(Path(args.logs))
    records = load_benchmark_records(Path(args.benchmark))
    examples = generate_sft_dataset(
        log_events,
        records,
        source_backend=args.source_backend,
        include_isolated=args.include_isolated,
    )

    if not examples:
        logger.warning("No SFT examples generated")
        return

    write_sft_dataset(examples, Path(args.output))
    print(f"\nGenerated {len(examples)} SFT examples → {args.output}")
    print(f"  Source backend : {args.source_backend}")
    print(f"  Include isolated: {args.include_isolated}")


if __name__ == "__main__":
    main()

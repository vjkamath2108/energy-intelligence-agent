"""
Stream 1 — Log Anomaly Pipeline entry point.

Wires together:
    tail_consumer → anomaly_scorer → bedrock_explainer

Usage:
    python -m stream1_log_anomaly.main
    python -m stream1_log_anomaly.main --explainer ollama   # not yet implemented
    python -m stream1_log_anomaly.main --log-file path/to/file.jsonl

Pipeline flow:
    1. tail_file() reads JSONL log file → asyncio.Queue
    2. pipeline() consumes queue → AnomalyScorer.score()
    3. On anomaly: BedrockExplainer.explain() → print structured output
"""

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from stream1_log_anomaly.src.models import LogEvent


from stream1_log_anomaly.src.streaming.tail_consumer import tail_file, consume_queue
from stream1_log_anomaly.src.detection.anomaly_scorer import AnomalyScorer
from stream1_log_anomaly.src.models import Explanation
from stream1_log_anomaly.src.explanation.bedrock_explainer import BedrockExplainer

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_LOG_FILE = Path("stream1_log_anomaly/logs/sample.jsonl")


def _print_explanation(event: LogEvent, explanation: Explanation, event_index: int) -> None:
    """Print a formatted anomaly explanation to stdout."""
    print(f"\n{'='*60}")
    print(f"🚨 ANOMALY [{event_index:03d}] — {event.service} | {event.level} | {event.latency_ms}ms")
    print(f"{'='*60}")
    print(f"Severity:  {explanation.severity}")
    print(f"Summary:   {explanation.summary}")
    print(f"Cause:     {explanation.likely_cause}")
    print(f"Action:    {explanation.recommended_action}")
    print(f"{'='*60}\n")


async def pipeline(
    queue: asyncio.Queue[LogEvent],
    explainer: BedrockExplainer,
    stop_event: asyncio.Event,
) -> None:
    """
    Consume events from queue, score for anomalies, explain if flagged.

    Maintains a rolling buffer of recent events for context window.
    Exits cleanly when stop_event is set and queue is drained.

    Args:
        queue: Source queue from tail_consumer.
        explainer: Configured explainer backend (Bedrock or Ollama).
        stop_event: Signals pipeline to exit after draining queue.
    """
    scorer = AnomalyScorer()
    recent_events: list[LogEvent] = []   # rolling window for ±5 context
    event_index = 0
    anomalies_found = 0

    async for event in consume_queue(queue):
        recent_events.append(event)
        if len(recent_events) > 50:
            recent_events.pop(0)

        result = scorer.score(event)

        if result is None:
            logger.debug("Buffering event %d", event_index)
        elif result.is_anomaly:
            anomalies_found += 1
            # ±5 context window, clamped to available events
            context_start = max(0, len(recent_events) - 6)
            context = recent_events[context_start:-1]

            logger.info(
                "Anomaly detected — calling explainer (event %d, score=%.4f)",
                event_index,
                result.anomaly_score,
            )
            try:
                explanation = await explainer.explain(event, context)
                _print_explanation(event, explanation, event_index)
            except Exception as e:
                logger.error("Explainer failed for event %d: %s", event_index, e)

        event_index += 1

        # Exit when stop_event is set and queue is empty
        if stop_event.is_set() and queue.empty():
            logger.info(
                "Pipeline complete — %d events processed, %d anomalies found.",
                event_index,
                anomalies_found,
            )
            return


async def main(log_file: Path, explainer_backend: str) -> None:
    """
    Entry point — wires tail_consumer, scorer, and explainer.

    Args:
        log_file: Path to JSONL log file to tail.
        explainer_backend: 'bedrock' or 'ollama' (ollama not yet implemented).
    """
    if not log_file.exists():
        raise FileNotFoundError(f"Log file not found: {log_file}")

    if explainer_backend == "ollama":
        raise NotImplementedError(
            "Ollama explainer not yet implemented — coming in Sprint 1. "
            "Run with --explainer bedrock"
        )

    explainer = BedrockExplainer(region=os.getenv("AWS_REGION", "eu-west-2"))

    queue: asyncio.Queue[LogEvent] = asyncio.Queue()
    stop_event = asyncio.Event()

    logger.info("Starting pipeline — log file: %s, explainer: %s", log_file, explainer_backend)

    await asyncio.gather(
        tail_file(log_file, queue, stop_event=stop_event),
        pipeline(queue, explainer, stop_event),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream 1 — Log Anomaly Pipeline")
    parser.add_argument(
        "--log-file",
        type=Path,
        default=DEFAULT_LOG_FILE,
        help="Path to JSONL log file (default: stream1_log_anomaly/logs/sample.jsonl)",
    )
    parser.add_argument(
        "--explainer",
        choices=["bedrock", "ollama"],
        default="bedrock",
        help="Explainer backend (default: bedrock)",
    )
    args = parser.parse_args()

    asyncio.run(main(args.log_file, args.explainer))
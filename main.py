"""EIA streaming pipeline entry point.

Wires the log anomaly streaming pipeline into the EIA project context:
    source (file | redis) → anomaly_scorer → explainer (bedrock | ollama)

Usage:
    python main.py
    python main.py --mode file --explainer bedrock
    python main.py --mode file --explainer ollama
    python main.py --mode redis --explainer bedrock
    python main.py --mode redis --explainer ollama --stream-key log_events

Modes:
    file  — tail a JSONL log file; processes existing lines then polls for new ones.
    redis — consume from a Redis Stream via XREAD blocking reads.

Explainer backends:
    bedrock — AWS Bedrock Claude Sonnet. Requires AWS credentials. Data leaves
              the machine to AWS eu-west-2. Set AWS_REGION / AWS_PROFILE in .env.
    ollama  — Local Ollama instance. Zero AWS calls. Requires Ollama running
              and the configured model pulled (default: llama3.2:3b).
              Set OLLAMA_MODEL env var to switch models.

Environment variables (all optional, override via .env):
    AWS_REGION       — AWS region for Bedrock (default: eu-west-2)
    REDIS_URL        — Redis connection URL (default: redis://localhost:6379)
    STREAM_KEY       — Redis stream key (default: log_events)
    OLLAMA_BASE_URL  — Ollama server URL (default: http://localhost:11434)
    OLLAMA_MODEL     — Ollama model name (default: llama3.2:3b)
"""

import argparse
import asyncio
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from src.instrumentation.langfuse_client import trace_anomaly_detection
from stream1_log_anomaly.src.detection.anomaly_scorer import AnomalyScorer
from stream1_log_anomaly.src.explanation.bedrock_explainer import BedrockExplainer
from stream1_log_anomaly.src.explanation.ollama_explainer import OllamaExplainer
from stream1_log_anomaly.src.models import Explanation, LogEvent
from stream1_log_anomaly.src.streaming.redis_consumer import redis_consumer
from stream1_log_anomaly.src.streaming.tail_consumer import consume_queue, tail_file

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_LOG_FILE = Path("stream1_log_anomaly/logs/sample.jsonl")


def _print_explanation(event: LogEvent, explanation: Explanation, event_index: int) -> None:
    """Print a formatted anomaly explanation to stdout."""
    print(f"\n{'=' * 60}")
    print(f"ANOMALY [{event_index:03d}] — {event.service} | {event.level} | {event.latency_ms}ms")
    print(f"{'=' * 60}")
    print(f"Severity:  {explanation.severity}")
    print(f"Summary:   {explanation.summary}")
    print(f"Cause:     {explanation.likely_cause}")
    print(f"Action:    {explanation.recommended_action}")
    print(f"{'=' * 60}\n")


async def pipeline(
    queue: asyncio.Queue[LogEvent],
    explainer: BedrockExplainer | OllamaExplainer,
    stop_event: asyncio.Event,
) -> None:
    """Consume events from queue, score for anomalies, explain if flagged.

    Maintains a rolling buffer of recent events for context window.
    Exits cleanly when stop_event is set and queue is drained.

    Args:
        queue: Source queue populated by tail_file() or redis_consumer().
        explainer: Configured explainer backend (Bedrock or Ollama).
        stop_event: Signals pipeline to exit after draining queue.
    """
    scorer = AnomalyScorer()
    recent_events: list[LogEvent] = []
    event_index = 0
    anomalies_found = 0

    async for event in consume_queue(queue):
        recent_events.append(event)
        if len(recent_events) > 50:
            recent_events.pop(0)

        async with trace_anomaly_detection(window_size=len(recent_events)) as det_span:
            result = scorer.score(event)
            if result is not None:
                det_span["is_anomaly"] = result.is_anomaly
                det_span["anomaly_score"] = result.anomaly_score
                det_span["event_id"] = event.ts

        if result is None:
            logger.debug("Buffering event %d", event_index)
        elif result.is_anomaly:
            anomalies_found += 1
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

        if stop_event.is_set() and queue.empty():
            logger.info(
                "Pipeline complete — %d events processed, %d anomalies found.",
                event_index,
                anomalies_found,
            )
            return


async def _build_explainer(
    backend: str,
) -> BedrockExplainer | OllamaExplainer:
    """Instantiate and health-check the chosen explainer backend.

    Args:
        backend: 'bedrock' or 'ollama'.

    Returns:
        Ready-to-use explainer instance.

    Raises:
        SystemExit: If the explainer backend is not reachable.
    """
    if backend == "ollama":
        explainer: BedrockExplainer | OllamaExplainer = OllamaExplainer()
        try:
            await explainer.check_health()  # type: ignore[union-attr]
        except RuntimeError as e:
            logger.error("Ollama health check failed: %s", e)
            raise SystemExit(1)
        return explainer

    return BedrockExplainer(region=os.getenv("AWS_REGION", "eu-west-2"))


async def run_file_mode(
    log_file: Path,
    explainer: BedrockExplainer | OllamaExplainer,
) -> None:
    """Run the pipeline in file-tail mode.

    Args:
        log_file: Path to JSONL log file to tail.
        explainer: Configured explainer backend.
    """
    if not log_file.exists():
        raise FileNotFoundError(f"Log file not found: {log_file}")

    queue: asyncio.Queue[LogEvent] = asyncio.Queue()
    stop_event = asyncio.Event()

    logger.info("Mode: file — tailing %s", log_file)

    await asyncio.gather(
        tail_file(log_file, queue, stop_event=stop_event),
        pipeline(queue, explainer, stop_event),
    )


async def run_redis_mode(
    stream_key: str,
    explainer: BedrockExplainer | OllamaExplainer,
) -> None:
    """Run the pipeline in Redis Streams mode.

    Args:
        stream_key: Redis stream key to consume from.
        explainer: Configured explainer backend.
    """
    queue: asyncio.Queue[LogEvent] = asyncio.Queue()
    stop_event = asyncio.Event()

    logger.info("Mode: redis — consuming stream '%s'", stream_key)

    await asyncio.gather(
        redis_consumer(queue, stop_event=stop_event, stream_key=stream_key),
        pipeline(queue, explainer, stop_event),
    )


async def main(
    mode: str,
    explainer_backend: str,
    log_file: Path,
    stream_key: str,
) -> None:
    """Wire source → anomaly_scorer → explainer based on CLI flags.

    Args:
        mode: 'file' or 'redis'.
        explainer_backend: 'bedrock' or 'ollama'.
        log_file: JSONL log file path (file mode only).
        stream_key: Redis stream key (redis mode only).
    """
    explainer = await _build_explainer(explainer_backend)

    try:
        if mode == "file":
            await run_file_mode(log_file, explainer)
        else:
            await run_redis_mode(stream_key, explainer)
    finally:
        if isinstance(explainer, OllamaExplainer):
            await explainer.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EIA streaming pipeline — log anomaly detection")
    parser.add_argument(
        "--mode",
        choices=["file", "redis"],
        default="file",
        help="Event source: tail a JSONL file (file) or consume a Redis Stream (redis). Default: file",
    )
    parser.add_argument(
        "--explainer",
        choices=["bedrock", "ollama"],
        default="bedrock",
        help="Explanation backend: AWS Bedrock Claude (bedrock) or local Ollama (ollama). Default: bedrock",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=DEFAULT_LOG_FILE,
        help=f"JSONL log file to tail (file mode only). Default: {DEFAULT_LOG_FILE}",
    )
    parser.add_argument(
        "--stream-key",
        default=os.getenv("STREAM_KEY", "log_events"),
        help="Redis stream key to consume (redis mode only). Default: log_events",
    )
    args = parser.parse_args()

    asyncio.run(
        main(
            mode=args.mode,
            explainer_backend=args.explainer,
            log_file=args.log_file,
            stream_key=args.stream_key,
        )
    )

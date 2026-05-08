"""
Asyncio file tail consumer.

Reads a JSONL log file from the beginning on startup, then continuously
polls for new lines. Parsed events are placed into an asyncio.Queue for
downstream consumers (anomaly scorer, etc.).

Production note: polling with asyncio.sleep is sufficient for a demo pipeline.
A production system on Linux would use inotify (watchdog library) to avoid
any polling overhead.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import AsyncIterator
from stream1_log_anomaly.src.models import LogEvent

logger = logging.getLogger(__name__)


async def tail_file(
    filepath: Path,
    queue: asyncio.Queue[LogEvent],
    poll_interval: float = 0.1,
    stop_event: asyncio.Event | None = None,
) -> None:
    """
    Tail a JSONL file and push parsed events onto a queue.

    Reads all existing lines on startup, then polls for new lines.
    Malformed JSON lines are logged and skipped — they do not crash the consumer.

    Args:
        filepath: Path to the JSONL log file.
        queue: Destination queue for parsed log event dicts.
        poll_interval: Seconds between polls when no new data. Default 0.1s.
        stop_event: If set, consumer exits cleanly when the event is set.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Log file not found: {filepath}")

    with filepath.open("r") as f:
        while True:
            line = f.readline()

            if line:
                line = line.strip()
                if line:  # skip blank lines
                    try:
                        #event = json.loads(line) - commented out to use pydantic model instead in sprint 1
                        event = LogEvent(**json.loads(line))
                        await queue.put(event)
                    except json.JSONDecodeError:
                        logger.warning("Skipping malformed line: %s", line[:120])
            else:
                # No new data — check stop condition, then wait
                if stop_event and stop_event.is_set():
                    logger.info("Stop event set — tail consumer exiting.")
                    return
                await asyncio.sleep(poll_interval)


async def consume_queue(queue: asyncio.Queue[dict]) -> AsyncIterator[LogEvent]:
    """
    Async generator that yields events from the queue.

    Usage:
        async for event in consume_queue(queue):
            process(event)
    """
    while True:
        event = await queue.get()
        yield event
        queue.task_done()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    async def _smoke_test() -> None:
        path = Path("stream1_log_anomaly/logs/sample.jsonl")
        q: asyncio.Queue[LogEvent] = asyncio.Queue()
        stop = asyncio.Event()
        count = 0

        async def drain() -> None:
            nonlocal count
            async for event in consume_queue(q):
                print(f"[{count:03d}] {event.level} — {event.service} — latency: {event.latency_ms} status:{event.status_code}")
                count += 1
                if count >= 50:
                    stop.set()
                    return

        await asyncio.gather(
            tail_file(path, q, stop_event=stop),
            drain(),
        )
        print(f"\nDone. {count} events processed.")

    asyncio.run(_smoke_test())
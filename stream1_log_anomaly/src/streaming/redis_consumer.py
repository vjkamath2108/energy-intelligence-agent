"""
Redis Streams consumer for log event ingestion.

Replaces file tail as the event source. Reads from a Redis Stream
using XREAD with blocking mode, parses events into LogEvent models,
and pushes onto an asyncio.Queue for downstream consumers.

Design decisions:
- XREAD with block=0 for efficient blocking reads — no polling overhead.
- Consumer tracks last_id to resume from where it left off on restart.
- asyncio.Queue is the internal handoff — scorer/explainer unchanged.
- File tail remains available via --mode file flag in main.py.
"""

import asyncio
import json
import logging
import os

import redis.asyncio as aioredis

from stream1_log_anomaly.src.models import LogEvent

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
STREAM_KEY = os.getenv("STREAM_KEY", "log_events")
BLOCK_MS = 1000  # block for 1s waiting for new events before looping


async def redis_consumer(
    queue: asyncio.Queue[LogEvent],
    stop_event: asyncio.Event | None = None,
    stream_key: str = STREAM_KEY,
) -> None:
    """
    Read from a Redis Stream and push LogEvent objects onto a queue.

    Starts reading from the beginning of the stream ('0-0') on startup.
    Uses blocking XREAD — efficient, no polling overhead.

    Args:
        queue: Destination queue for parsed LogEvent objects.
        stop_event: If set, consumer exits cleanly when the event is set.
        stream_key: Redis stream key to consume from.
    """
    client = aioredis.from_url(REDIS_URL, decode_responses=True)
    last_id = "0-0"  # start from beginning of stream

    logger.info("Redis consumer started — stream=%s", stream_key)

    try:
        while True:
            if stop_event and stop_event.is_set():
                logger.info("Stop event set — Redis consumer exiting.")
                break

            # Blocking read — waits up to BLOCK_MS for new entries
            results = await client.xread(
                {stream_key: last_id},
                block=BLOCK_MS,
                count=10,  # process up to 10 events per read
            )

            if not results:
                continue  # timeout, loop and check stop_event

            for _, messages in results:
                for message_id, fields in messages:
                    last_id = message_id  # advance cursor

                    try:
                        # fields is a flat dict — parse JSON payload
                        raw = json.loads(fields["payload"])
                        event = LogEvent(**raw)
                        await queue.put(event)
                        logger.debug(
                            "Consumed event %s — %s %s",
                            message_id,
                            event.level,
                            event.service,
                        )
                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        logger.warning("Skipping malformed message %s: %s", message_id, e)

    finally:
        await client.aclose()
        logger.info("Redis client closed.")


async def publish_log_events(
    filepath: str,
    stream_key: str = STREAM_KEY,
) -> None:
    """
    Publish log events from a JSONL file into a Redis Stream.

    Used for testing — loads sample.jsonl and XADDs each event.
    In production, the log generator writes directly to Redis.

    Args:
        filepath: Path to JSONL file containing log events.
        stream_key: Redis stream key to publish to.
    """
    client = aioredis.from_url(REDIS_URL, decode_responses=True)

    try:
        with open(filepath) as f:
            lines = [l.strip() for l in f if l.strip()]

        for line in lines:
            await client.xadd(stream_key, {"payload": line})
            await asyncio.sleep(0.05)  # 50ms between events — realistic pace

        logger.info("Published %d events to stream %s", len(lines), stream_key)
    finally:
        await client.aclose()


if __name__ == "__main__":
    import asyncio
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    async def _smoke_test() -> None:
        stream_key = "log_events_test"
        q: asyncio.Queue[LogEvent] = asyncio.Queue()
        stop = asyncio.Event()
        received = 0

        # Publish events first, then consume
        await publish_log_events(
            "stream1_log_anomaly/logs/sample.jsonl",
            stream_key=stream_key,
        )

        async def drain() -> None:
            nonlocal received
            while True:
                event = await q.get()
                print(f"[{received:03d}] {event.level} — {event.service} — latency: {event.latency_ms}ms")
                received += 1
                if received >= 50:
                    stop.set()
                    return

        await asyncio.gather(
            redis_consumer(q, stop_event=stop, stream_key=stream_key),
            drain(),
        )

        print(f"\nDone. {received} events consumed from Redis Stream.")

    asyncio.run(_smoke_test())
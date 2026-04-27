"""COGRAM EVENTS — event-driven change notifications via Redis pub/sub.

Producers (cache writes, trainer runs, budget updates, narration events,
ingest writes) call `publish(channel, payload)`. Consumers (dashboard, future
plugins) call `subscribe(pattern, handler)` to react.

No polling. Events fire on actual writes only.

Channels:
  cogram:events:graph_change       — entity/edge created/updated
  cogram:events:cache_change       — Engram cache hit/miss/evict/feedback
  cogram:events:spend_change       — daily_spend updated
  cogram:events:training_change    — adapter_runs row written
  cogram:events:narrative_change   — vLLM narrative produced/refreshed
  cogram:events:session_change     — Redis active_memory subgraph mutated
  cogram:events:webhook            — external HTTP webhook fired
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, AsyncIterator, Callable, Optional

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
EVENT_PREFIX = "cogram:events:"

_pub_client = None
_sub_client = None


async def _pub():
    global _pub_client
    if _pub_client is not None:
        return _pub_client
    try:
        import redis.asyncio as redis_asyncio
    except ImportError:
        return None
    _pub_client = redis_asyncio.from_url(REDIS_URL, decode_responses=True)
    return _pub_client


async def _sub():
    global _sub_client
    if _sub_client is not None:
        return _sub_client
    try:
        import redis.asyncio as redis_asyncio
    except ImportError:
        return None
    _sub_client = redis_asyncio.from_url(REDIS_URL, decode_responses=True)
    return _sub_client


# ---------------------------------------------------------------------------
# Publish (called by any writer when data changes)
# ---------------------------------------------------------------------------

async def publish(channel: str, payload: dict | None = None) -> None:
    """Publish a change event. Fire-and-forget; never blocks the writer."""
    r = await _pub()
    if r is None:
        return
    msg = {
        "ts": time.time(),
        "channel": channel,
        "payload": payload or {},
    }
    full_channel = channel if channel.startswith(EVENT_PREFIX) else EVENT_PREFIX + channel
    try:
        await r.publish(full_channel, json.dumps(msg, default=str))
    except Exception:
        # Never let a publish failure break the writer
        pass


def publish_sync(channel: str, payload: dict | None = None) -> None:
    """Sync wrapper for use in non-async contexts (e.g. graphiti monkey-patches)."""
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(publish(channel, payload))
    except RuntimeError:
        # No running loop — fire-and-forget in a fresh one
        try:
            asyncio.run(publish(channel, payload))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Subscribe (used by dashboard to forward events to browser via SSE)
# ---------------------------------------------------------------------------

async def stream(pattern: str = "cogram:events:*") -> AsyncIterator[dict]:
    """Async generator yielding events matching the pattern.

    Yields dicts: {"ts": float, "channel": str, "payload": dict}
    Yields a heartbeat every 30s to keep the connection alive when idle.
    """
    r = await _sub()
    if r is None:
        return
    pubsub = r.pubsub()
    await pubsub.psubscribe(pattern)
    last_heartbeat = time.time()
    try:
        while True:
            try:
                message = await asyncio.wait_for(pubsub.get_message(ignore_subscribe_messages=True), timeout=5.0)
            except asyncio.TimeoutError:
                message = None

            if message and message.get("type") in ("pmessage", "message"):
                try:
                    data = json.loads(message["data"])
                except (json.JSONDecodeError, TypeError):
                    data = {"raw": message.get("data")}
                # strip prefix from channel for cleaner client-side switching
                if "channel" in data and data["channel"].startswith(EVENT_PREFIX):
                    data["channel"] = data["channel"][len(EVENT_PREFIX):]
                yield data
                last_heartbeat = time.time()
            elif time.time() - last_heartbeat > 30:
                # Heartbeat to keep SSE connection alive
                yield {"ts": time.time(), "channel": "heartbeat", "payload": {}}
                last_heartbeat = time.time()
    finally:
        await pubsub.punsubscribe(pattern)
        await pubsub.aclose()


# ---------------------------------------------------------------------------
# Convenience constants for callers (avoid string typos)
# ---------------------------------------------------------------------------

GRAPH_CHANGE = "cogram:events:graph_change"
CACHE_CHANGE = "cogram:events:cache_change"
SPEND_CHANGE = "cogram:events:spend_change"
TRAINING_CHANGE = "cogram:events:training_change"
NARRATIVE_CHANGE = "cogram:events:narrative_change"
INTENT_ANNOTATED = "cogram:events:intent_annotated"
PROFILE_CHANGE = "cogram:events:profile_change"
SESSION_CHANGE = "cogram:events:session_change"
WEBHOOK = "cogram:events:webhook"

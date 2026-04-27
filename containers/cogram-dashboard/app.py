"""Cogram dashboard — live observability for the running stack.

Endpoints:
  GET /              — HTML UI with force-graph + live metrics
  GET /health        — liveness
  GET /api/events    — SSE stream: emits only when data changes server-side
  GET /api/stats     — one-shot JSON counters (manual refresh)
  GET /api/graph     — graph payload for force-graph viz {nodes, links}
  GET /api/recent_events — last 50 audit events from Engram
  GET /api/training_runs — last 20 LoRA training runs

Streaming model: server polls Postgres + Neo4j every CHECK_INTERVAL seconds,
hashes the resulting state, and only pushes an event when the hash changes.
Client opens ONE long-lived SSE connection. No periodic fetches.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator

# Silence Neo4j's "property doesn't exist" notifications. These flood the log
# while no node has yet been narrated. Neo4j emits them via several channels;
# we silence all known ones.
logging.getLogger("neo4j").setLevel(logging.ERROR)
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)
logging.getLogger("neo4j.io").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*property key does not exist.*")
warnings.filterwarnings("ignore", message=".*Received notification.*")

import asyncpg
import httpx
import redis.asyncio as redis_asyncio
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from neo4j import AsyncGraphDatabase


NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")
REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")
POSTGRES_DSN = os.environ.get("POSTGRES_DSN", "postgresql://cogram:cogram@postgres:5432/cogram")
MCP_URL = os.environ.get("MCP_URL", "http://cogram-mcp:7800")
TRAINER_URL = os.environ.get("TRAINER_URL", "http://cogram-trainer:7900")


app = FastAPI(title="cogram-dashboard", version="0.1.0")

_neo4j_driver = None
_redis_client = None
_pg_pool = None
_http_client = None


async def _neo4j():
    global _neo4j_driver
    if _neo4j_driver is None:
        # Try every known disable mechanism for neo4j 5.x and 6.x drivers
        for kwargs in [
            {"notifications_min_severity": "OFF",
             "notifications_disabled_classifications": ["UNRECOGNIZED", "DEPRECATION"]},
            {"notifications_min_severity": "OFF"},
            {"notifications_disabled_categories": ["UNRECOGNIZED", "DEPRECATION"]},
            {},
        ]:
            try:
                _neo4j_driver = AsyncGraphDatabase.driver(
                    NEO4J_URI,
                    auth=(NEO4J_USER, NEO4J_PASSWORD),
                    **kwargs,
                )
                break
            except TypeError:
                continue
    return _neo4j_driver


async def _redis():
    global _redis_client
    if _redis_client is None:
        _redis_client = redis_asyncio.from_url(REDIS_URL, decode_responses=True)
    return _redis_client


async def _pg():
    global _pg_pool
    if _pg_pool is None:
        _pg_pool = await asyncpg.create_pool(POSTGRES_DSN, min_size=1, max_size=5)
    return _pg_pool


async def _http():
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(timeout=httpx.Timeout(5.0))
    return _http_client


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    return {"ok": True, "service": "cogram-dashboard"}


@app.get("/api/stats")
async def stats() -> dict:
    out: dict = {"timestamp": datetime.now(timezone.utc).isoformat()}

    # Neo4j counts
    try:
        driver = await _neo4j()
        async with driver.session() as session:
            res = await session.run(
                "MATCH (n) RETURN labels(n)[0] AS kind, count(*) AS n"
            )
            out["nodes"] = {r["kind"]: r["n"] async for r in res}
            res2 = await session.run(
                "MATCH ()-[r:RELATES_TO]->() RETURN count(r) AS n"
            )
            row2 = await res2.single()
            out["relates_to_edges"] = row2["n"] if row2 else 0
            res3 = await session.run(
                "MATCH ()-[r:RELATES_TO]->() WHERE r.intent_meta IS NOT NULL "
                "RETURN count(r) AS n"
            )
            row3 = await res3.single()
            out["edges_with_intent"] = row3["n"] if row3 else 0
            res4 = await session.run(
                "MATCH (n:Entity) WHERE n.vllm_narrative IS NOT NULL "
                "RETURN count(n) AS n"
            )
            row4 = await res4.single()
            out["nodes_with_narrative"] = row4["n"] if row4 else 0
    except Exception as e:
        out["neo4j_error"] = str(e)

    # Redis (active subgraph sessions)
    try:
        r = await _redis()
        keys = []
        cursor = 0
        while True:
            cursor, batch = await r.scan(cursor, match="cogram:session:*:active", count=100)
            keys.extend(batch)
            if cursor == 0:
                break
        out["active_sessions"] = len(keys)
    except Exception as e:
        out["redis_error"] = str(e)

    # Postgres / Engram
    try:
        pool = await _pg()
        async with pool.acquire() as conn:
            cache_count = await conn.fetchval(
                "SELECT COUNT(*) FROM engram.patterns WHERE namespace='cogram'"
            )
            out["cache_entries"] = int(cache_count or 0)
            avg_conf = await conn.fetchval(
                "SELECT AVG(confidence) FROM engram.patterns WHERE namespace='cogram'"
            )
            out["cache_avg_confidence"] = round(float(avg_conf or 1.0), 3)
            today = datetime.now(timezone.utc).date()
            spend = await conn.fetchrow(
                "SELECT calls, tokens_in, tokens_out, cost_usd FROM engram.daily_spend WHERE day=$1",
                today,
            )
            out["spend_today"] = (
                {
                    "calls": spend["calls"],
                    "tokens_in": spend["tokens_in"],
                    "tokens_out": spend["tokens_out"],
                    "cost_usd": round(float(spend["cost_usd"]), 4),
                }
                if spend else {"calls": 0, "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0}
            )
    except Exception as e:
        out["postgres_error"] = str(e)

    # Trainer status
    try:
        client = await _http()
        resp = await client.get(f"{TRAINER_URL}/status")
        if resp.status_code == 200:
            out["trainer"] = resp.json()
    except Exception as e:
        out["trainer_error"] = str(e)

    return out


@app.get("/api/graph")
async def graph(limit: int = 200) -> dict:
    """Force-graph payload {nodes:[{id,label,color}], links:[{source,target,label}]}."""
    try:
        driver = await _neo4j()
        async with driver.session() as session:
            res = await session.run(
                """
                MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
                RETURN
                    a.uuid AS aid, coalesce(a.name, '') AS aname,
                    b.uuid AS bid, coalesce(b.name, '') AS bname,
                    coalesce(r.fact, '') AS fact,
                    r.intent_meta AS intent_meta,
                    a.vllm_narrative IS NOT NULL AS a_has_narrative,
                    b.vllm_narrative IS NOT NULL AS b_has_narrative
                LIMIT $limit
                """,
                limit=limit,
            )
            nodes: dict = {}
            links: list = []
            async for r in res:
                if r["aid"] not in nodes:
                    nodes[r["aid"]] = {
                        "id": r["aid"],
                        "label": r["aname"][:40],
                        "has_narrative": r["a_has_narrative"],
                    }
                if r["bid"] not in nodes:
                    nodes[r["bid"]] = {
                        "id": r["bid"],
                        "label": r["bname"][:40],
                        "has_narrative": r["b_has_narrative"],
                    }
                pat = ""
                if r["intent_meta"]:
                    try:
                        pat = json.loads(r["intent_meta"]).get("cognitive_pattern", "")[:30]
                    except (TypeError, json.JSONDecodeError):
                        pat = ""
                links.append({
                    "source": r["aid"],
                    "target": r["bid"],
                    "label": pat or r["fact"][:40],
                })
        return {"nodes": list(nodes.values()), "links": links}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/recent_events")
async def recent_events(limit: int = 50) -> list:
    try:
        pool = await _pg()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT pattern_id, decision::text AS decision, source, created_at "
                "FROM engram.audit ORDER BY created_at DESC LIMIT $1",
                limit,
            )
        return [
            {
                "pattern_id": r["pattern_id"][:16] if r["pattern_id"] else None,
                "decision": r["decision"],
                "source": r["source"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/training_runs")
async def training_runs(limit: int = 20) -> list:
    try:
        pool = await _pg()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT node_id, base_model, samples_used, duration_sec, backend, status, "
                "started_at, finished_at, error "
                "FROM engram.adapter_runs ORDER BY started_at DESC LIMIT $1",
                limit,
            )
        return [dict(r) for r in rows]
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

INDEX_HTML = Path(__file__).parent / "templates" / "index.html"


@app.get("/", response_class=HTMLResponse)
async def root() -> str:
    if INDEX_HTML.exists():
        return INDEX_HTML.read_text(encoding="utf-8")
    return "<h1>cogram dashboard</h1><p>UI not built. Try /api/stats</p>"


# ---------------------------------------------------------------------------
# Event-driven push (SSE) + HTTP webhook
# ---------------------------------------------------------------------------

@app.get("/api/events")
async def event_stream(request: Request) -> StreamingResponse:
    """Server-Sent Events stream. Subscribes to Redis pub/sub on cogram:events:*
    and forwards messages to the browser. ZERO server-side polling."""
    # Lazy import so dashboard works even if events module changes
    import sys
    sys.path.insert(0, "/app")
    try:
        from src import events as ev
    except ImportError:
        # When running outside the cogram-mcp container, fall back to a tiny inline impl
        ev = None

    async def gen():
        if ev is not None:
            async for msg in ev.stream("cogram:events:*"):
                if await request.is_disconnected():
                    break
                yield f"data: {json.dumps(msg, default=str)}\n\n"
        else:
            # Subscribe directly via redis client if `events` module is unavailable
            r = await _redis()
            if r is None:
                yield 'data: {"channel":"error","payload":{"msg":"redis unavailable"}}\n\n'
                return
            pubsub = r.pubsub()
            await pubsub.psubscribe("cogram:events:*")
            try:
                while not (await request.is_disconnected()):
                    message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=5.0)
                    if message and message.get("type") in ("pmessage", "message"):
                        try:
                            data = json.loads(message["data"])
                        except Exception:
                            data = {"raw": str(message.get("data"))}
                        yield f"data: {json.dumps(data, default=str)}\n\n"
                    else:
                        yield 'data: {"channel":"heartbeat"}\n\n'
            finally:
                await pubsub.punsubscribe("cogram:events:*")
                await pubsub.aclose()

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/webhook/{channel}")
async def webhook(channel: str, request: Request) -> dict:
    """Accept HTTP webhooks from any source (writers, external scripts).
    Forwards to Redis pub/sub so dashboard browsers see it via SSE.

    Example: curl -X POST http://localhost:7801/webhook/graph_change -H 'Content-Type: application/json' -d '{"reason":"manual"}'
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    r = await _redis()
    if r is None:
        return {"ok": False, "error": "redis unavailable"}
    msg = {
        "ts": datetime.now(timezone.utc).timestamp(),
        "channel": f"webhook:{channel}",
        "payload": body,
    }
    await r.publish(f"cogram:events:{channel}", json.dumps(msg, default=str))
    return {"ok": True, "channel": channel}

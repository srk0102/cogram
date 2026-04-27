"""Engram-style decision cache.

Backed by Postgres (cogram-postgres container, schema in postgres/init.sql)
when POSTGRES_DSN is set; falls back to a local SQLite file when not. Same
Policy API in both modes — production deploys use Postgres, local dev or
no-Postgres setups still work.

Modes (LLM_CACHE_MODE env var):
  record (default) — miss = call real API, store with confidence=1.0
  replay           — miss = call real API + store + WARN
  strict_replay    — miss = RAISE (test-only)
  off              — bypass cache entirely

Schema columns (Engram-compatible):
  id (text, primary key)         — sha256 hash of (namespace, payload)
  namespace (text)               — 'cogram'
  decision (enum)                — 'llm_response' | 'embedding' | 'narration' | ...
  confidence (float 0..1)        — current confidence
  meta (jsonb)                   — pickled response, base64-encoded
  created_at, updated_at (epoch ms)
  hit_count (int)
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import math
import os
import pickle
import sqlite3
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

# ---------------------------------------------------------------------------
# Config

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)
SQLITE_PATH = CACHE_DIR / "llm.db"

POSTGRES_DSN = os.environ.get("POSTGRES_DSN", "").strip()
NAMESPACE = os.environ.get("COGRAM_NAMESPACE", "cogram")
MODE = os.environ.get("LLM_CACHE_MODE", "record").lower()

DEFAULT_FLOOR = float(os.environ.get("LLM_CACHE_FLOOR", "0.2"))
DEFAULT_HALF_LIFE_DAYS = float(os.environ.get("LLM_CACHE_HALF_LIFE_DAYS", "30"))
_HALF_LIFE_SECONDS = DEFAULT_HALF_LIFE_DAYS * 86_400


def _now_ms() -> int:
    return int(time.time() * 1000)


def _hash(kind: str, payload: dict) -> str:
    blob = json.dumps({"kind": kind, "payload": payload}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def _decayed_confidence(stored: float, updated_at_ms: float) -> float:
    age_sec = max(0.0, time.time() - updated_at_ms / 1000.0)
    if age_sec == 0 or _HALF_LIFE_SECONDS == 0:
        return stored
    return stored * math.pow(0.5, age_sec / _HALF_LIFE_SECONDS)


# ---------------------------------------------------------------------------
# Postgres backend (preferred)
# ---------------------------------------------------------------------------

_pool = None


async def _get_pool():
    """Lazy asyncpg pool to engram.patterns."""
    global _pool
    if _pool is not None:
        return _pool
    if not POSTGRES_DSN:
        return None
    try:
        import asyncpg  # type: ignore
    except ImportError:
        return None
    _pool = await asyncpg.create_pool(POSTGRES_DSN, min_size=1, max_size=10)
    return _pool


async def _pg_get(key: str) -> Optional[tuple[Any, float]]:
    pool = await _get_pool()
    if pool is None:
        return None
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT meta, confidence, updated_at FROM engram.patterns "
            "WHERE id=$1 AND namespace=$2",
            key, NAMESPACE,
        )
    if row is None:
        return None
    blob_b64 = (row["meta"] or {}).get("response_b64") if isinstance(row["meta"], dict) else None
    if not blob_b64:
        return None
    try:
        response = pickle.loads(base64.b64decode(blob_b64))
    except Exception:
        return None

    current = _decayed_confidence(row["confidence"], float(row["updated_at"]))
    if current < DEFAULT_FLOOR:
        await _pg_evict(key)
        return None

    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE engram.patterns SET hit_count = hit_count + 1 WHERE id=$1",
            key,
        )
    return response, current


async def _pg_put(key: str, kind: str, response: Any, confidence: float = 1.0) -> None:
    pool = await _get_pool()
    if pool is None:
        return
    blob = base64.b64encode(pickle.dumps(response)).decode()
    now = _now_ms()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO engram.patterns
                (id, namespace, decision, confidence, created_at, updated_at, meta)
            VALUES ($1, $2, $3::engram.decision_kind, $4, $5, $5, $6::jsonb)
            ON CONFLICT (id) DO UPDATE
              SET confidence = EXCLUDED.confidence,
                  updated_at = EXCLUDED.updated_at,
                  meta       = EXCLUDED.meta
            """,
            key, NAMESPACE, _kind_to_enum(kind), confidence, now,
            json.dumps({"response_b64": blob}),
        )
        await conn.execute(
            "INSERT INTO engram.audit (pattern_id, namespace, decision, source) "
            "VALUES ($1, $2, $3::engram.decision_kind, 'miss')",
            key, NAMESPACE, _kind_to_enum(kind),
        )
    # Fire change event so dashboard updates without polling
    try:
        from cogram.utils import events as _ev
        await _ev.publish(_ev.CACHE_CHANGE, {"key": key[:12], "kind": kind, "source": "miss"})
    except Exception:
        pass


async def _pg_evict(key: str) -> None:
    pool = await _get_pool()
    if pool is None:
        return
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM engram.patterns WHERE id=$1", key)
        await conn.execute(
            "INSERT INTO engram.audit (pattern_id, namespace, decision, source) "
            "VALUES ($1, $2, 'llm_response'::engram.decision_kind, 'evict')",
            key, NAMESPACE,
        )


async def _pg_feedback(key: str, was_correct: bool, delta: float = 0.1) -> None:
    pool = await _get_pool()
    if pool is None:
        return
    adjustment = delta if was_correct else -2 * delta
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE engram.patterns
            SET confidence = LEAST(1.0, GREATEST(0.0, confidence + $2)),
                updated_at = $3,
                success_count = success_count + CASE WHEN $4 THEN 1 ELSE 0 END,
                failure_count = failure_count + CASE WHEN $4 THEN 0 ELSE 1 END
            WHERE id=$1
            """,
            key, adjustment, _now_ms(), was_correct,
        )


def _kind_to_enum(kind: str) -> str:
    """Map our cache kinds to the engram.decision_kind enum."""
    table = {
        "llm": "llm_response",
        "embed": "embedding",
        "narration": "narration",
        "dedup": "dedup",
        "extract": "extract",
    }
    return table.get(kind, "llm_response")


# ---------------------------------------------------------------------------
# SQLite fallback (local dev, no Postgres)
# ---------------------------------------------------------------------------

def _sqlite_conn() -> sqlite3.Connection:
    c = sqlite3.connect(SQLITE_PATH)
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS calls (
            key TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            response BLOB NOT NULL,
            confidence REAL NOT NULL DEFAULT 1.0,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            hits INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    return c


def _sqlite_get(key: str):
    with _sqlite_conn() as c:
        row = c.execute(
            "SELECT response, confidence, updated_at FROM calls WHERE key=?", (key,)
        ).fetchone()
    if row is None:
        return None
    response_blob, stored_conf, updated_at_sec = row
    current = _decayed_confidence(stored_conf, updated_at_sec * 1000.0)
    if current < DEFAULT_FLOOR:
        with _sqlite_conn() as c:
            c.execute("DELETE FROM calls WHERE key=?", (key,))
        return None
    with _sqlite_conn() as c:
        c.execute("UPDATE calls SET hits = hits + 1 WHERE key=?", (key,))
    return pickle.loads(response_blob), current


def _sqlite_put(key: str, kind: str, response: Any, confidence: float = 1.0) -> None:
    blob = pickle.dumps(response)
    now = time.time()
    with _sqlite_conn() as c:
        c.execute(
            "INSERT OR REPLACE INTO calls "
            "(key, kind, response, confidence, created_at, updated_at, hits) "
            "VALUES (?, ?, ?, ?, ?, ?, COALESCE((SELECT hits FROM calls WHERE key=?), 0))",
            (key, kind, blob, confidence, now, now, key),
        )


# ---------------------------------------------------------------------------
# Public sync helpers (used by graphiti's legacy patch path)
# ---------------------------------------------------------------------------

def _get_sync(key: str):
    """Sync read for graphiti monkey-patch path. Postgres if available, else SQLite."""
    if POSTGRES_DSN:
        # Run async in this thread
        try:
            return asyncio.run(_pg_get(key))
        except RuntimeError:
            # Already in an event loop — fall back to sqlite
            return _sqlite_get(key)
    return _sqlite_get(key)


def _put_sync(key: str, kind: str, response: Any, confidence: float = 1.0) -> None:
    if POSTGRES_DSN:
        try:
            asyncio.run(_pg_put(key, kind, response, confidence))
            return
        except RuntimeError:
            pass
    _sqlite_put(key, kind, response, confidence)


# ---------------------------------------------------------------------------
# Async helpers (preferred path inside the cogram-mcp server)
# ---------------------------------------------------------------------------

async def get(key: str):
    if POSTGRES_DSN:
        result = await _pg_get(key)
        if result is not None:
            return result
    # Fall back to sqlite for local dev compatibility
    return _sqlite_get(key)


async def put(key: str, kind: str, response: Any, confidence: float = 1.0) -> None:
    if POSTGRES_DSN:
        await _pg_put(key, kind, response, confidence)
    else:
        _sqlite_put(key, kind, response, confidence)


async def feedback(key: str, was_correct: bool, delta: float = 0.1) -> None:
    if POSTGRES_DSN:
        await _pg_feedback(key, was_correct, delta)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

async def stats() -> dict:
    if POSTGRES_DSN:
        pool = await _get_pool()
        if pool is None:
            return _sqlite_stats_sync()
        async with pool.acquire() as conn:
            total = await conn.fetchval(
                "SELECT COUNT(*) FROM engram.patterns WHERE namespace=$1", NAMESPACE
            )
            by_kind = await conn.fetch(
                "SELECT decision::text AS k, COUNT(*) AS n FROM engram.patterns "
                "WHERE namespace=$1 GROUP BY decision",
                NAMESPACE,
            )
            avg_conf = await conn.fetchval(
                "SELECT AVG(confidence) FROM engram.patterns WHERE namespace=$1", NAMESPACE
            )
            avg_hits = await conn.fetchval(
                "SELECT AVG(hit_count) FROM engram.patterns WHERE namespace=$1", NAMESPACE
            )
        return {
            "mode": MODE,
            "backend": "postgres",
            "namespace": NAMESPACE,
            "total": int(total or 0),
            "by_kind": {r["k"]: r["n"] for r in by_kind},
            "avg_confidence": round(float(avg_conf or 1.0), 3),
            "avg_hits_per_entry": round(float(avg_hits or 0.0), 2),
            "floor": DEFAULT_FLOOR,
            "half_life_days": DEFAULT_HALF_LIFE_DAYS,
        }
    return _sqlite_stats_sync()


def _sqlite_stats_sync() -> dict:
    with _sqlite_conn() as c:
        total = c.execute("SELECT COUNT(*) FROM calls").fetchone()[0]
        by_kind = dict(c.execute("SELECT kind, COUNT(*) FROM calls GROUP BY kind").fetchall())
        row = c.execute("SELECT AVG(confidence), AVG(hits) FROM calls").fetchone()
    return {
        "mode": MODE,
        "backend": "sqlite",
        "namespace": NAMESPACE,
        "total": total,
        "by_kind": by_kind,
        "avg_confidence": round(row[0] if row[0] is not None else 1.0, 3),
        "avg_hits_per_entry": round(row[1] if row[1] is not None else 0.0, 2),
        "floor": DEFAULT_FLOOR,
        "half_life_days": DEFAULT_HALF_LIFE_DAYS,
        "db": str(SQLITE_PATH),
    }


class CacheMiss(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Policy primitive (Engram TS API parity)
# ---------------------------------------------------------------------------

class Policy:
    """Engram-style policy. Wraps a brain with cached decisions + confidence."""

    def __init__(
        self,
        name: str,
        brain: Callable[..., Awaitable[Any]],
        cache_key_fn: Callable[..., str],
        confidence_floor: float = DEFAULT_FLOOR,
    ) -> None:
        self.name = name
        self.brain = brain
        self.cache_key_fn = cache_key_fn
        self.confidence_floor = confidence_floor

    def _full_key(self, *args, **kwargs) -> str:
        return _hash(self.name, {"k": self.cache_key_fn(*args, **kwargs)})

    async def __call__(self, *args, **kwargs) -> Any:
        if MODE == "off":
            return await self.brain(*args, **kwargs)

        key = self._full_key(*args, **kwargs)
        hit = await get(key)
        if hit is not None:
            return hit[0]

        if MODE == "strict_replay":
            raise CacheMiss(f"strict_replay miss on policy '{self.name}': {key[:12]}")
        if MODE == "replay":
            print(f"  [cache miss → live call: policy={self.name} {key[:12]}]")

        response = await self.brain(*args, **kwargs)
        await put(key, kind=self.name, response=response, confidence=1.0)
        return response

    async def feedback(self, was_correct: bool, *args, **kwargs) -> None:
        await feedback(self._full_key(*args, **kwargs), was_correct)


# ---------------------------------------------------------------------------
# Legacy graphiti monkey-patch (kept for backward-compat with existing pipeline)
# ---------------------------------------------------------------------------

def patch_clients(graphiti) -> None:
    if MODE == "off":
        return

    llm = graphiti.llm_client
    if not getattr(llm, "_cache_wrapped", False):
        original = llm.generate_response

        async def cached_generate(*args, **kwargs):
            payload = {"args": [str(a) for a in args], "kwargs": {k: str(v) for k, v in kwargs.items()}}
            key = _hash("llm", payload)
            if POSTGRES_DSN:
                hit = await _pg_get(key)
            else:
                hit = _sqlite_get(key)
            if hit is not None:
                return hit[0]
            if MODE == "strict_replay":
                raise CacheMiss(f"strict_replay miss on llm call: {key[:12]}")
            if MODE == "replay":
                print(f"  [cache miss → live LLM call: {key[:12]}]")
            resp = await original(*args, **kwargs)
            if POSTGRES_DSN:
                await _pg_put(key, "llm", resp)
            else:
                _sqlite_put(key, "llm", resp)
            return resp

        llm.generate_response = cached_generate
        llm._cache_wrapped = True

    emb = graphiti.embedder
    if not getattr(emb, "_cache_wrapped", False):
        original_create = emb.create

        async def cached_create(*args, **kwargs):
            input_data = kwargs.get("input_data") or (args[0] if args else None)
            payload = {"input": str(input_data), "model": getattr(emb.config, "embedding_model", "")}
            key = _hash("embed", payload)
            if POSTGRES_DSN:
                hit = await _pg_get(key)
            else:
                hit = _sqlite_get(key)
            if hit is not None:
                return hit[0]
            if MODE == "strict_replay":
                raise CacheMiss(f"strict_replay miss on embed: {key[:12]}")
            resp = await original_create(*args, **kwargs)
            if POSTGRES_DSN:
                await _pg_put(key, "embed", resp)
            else:
                _sqlite_put(key, "embed", resp)
            return resp

        emb.create = cached_create
        emb._cache_wrapped = True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cli() -> None:
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "stats":
        print(json.dumps(asyncio.run(stats()), indent=2))
    elif len(sys.argv) > 1 and sys.argv[1] == "clear":
        SQLITE_PATH.unlink(missing_ok=True)
        print(f"cleared {SQLITE_PATH}")
        if POSTGRES_DSN:
            print("postgres cache untouched (use SQL: DELETE FROM engram.patterns WHERE namespace='cogram')")
    else:
        print(json.dumps(asyncio.run(stats()), indent=2))


if __name__ == "__main__":
    cli()

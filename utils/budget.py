"""Daily spend tracking + safety cap.

Records each LLM call's cost into engram.daily_spend (when Postgres available)
and exposes a hard cap that the watcher honors. Designed to make it impossible
for a runaway watcher to exceed the user's $9 OpenAI budget by accident.
"""
from __future__ import annotations

import os
from datetime import date

# OpenAI gpt-4o-mini pricing (USD per token)
PRICE_IN = 0.15 / 1_000_000
PRICE_OUT = 0.60 / 1_000_000
PRICE_EMBED = 0.02 / 1_000_000

MAX_DAILY_INGEST_CALLS = int(os.environ.get("MAX_DAILY_INGEST_CALLS", "300"))
MAX_DAILY_USD = float(os.environ.get("MAX_DAILY_USD", "1.0"))   # extra ceiling

POSTGRES_DSN = os.environ.get("POSTGRES_DSN", "")

_pool = None


async def _pg():
    global _pool
    if _pool is not None:
        return _pool
    if not POSTGRES_DSN:
        return None
    try:
        import asyncpg  # type: ignore
    except ImportError:
        return None
    _pool = await asyncpg.create_pool(POSTGRES_DSN, min_size=1, max_size=3)
    return _pool


def estimate_cost(tokens_in: int, tokens_out: int, kind: str = "llm") -> float:
    if kind == "embed":
        return tokens_in * PRICE_EMBED
    return tokens_in * PRICE_IN + tokens_out * PRICE_OUT


async def record(tokens_in: int, tokens_out: int, kind: str = "llm") -> None:
    pool = await _pg()
    if pool is None:
        return
    cost = estimate_cost(tokens_in, tokens_out, kind)
    today = date.today()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO engram.daily_spend (day, calls, tokens_in, tokens_out, cost_usd)
            VALUES ($1, 1, $2, $3, $4)
            ON CONFLICT (day) DO UPDATE
              SET calls       = engram.daily_spend.calls + 1,
                  tokens_in   = engram.daily_spend.tokens_in + EXCLUDED.tokens_in,
                  tokens_out  = engram.daily_spend.tokens_out + EXCLUDED.tokens_out,
                  cost_usd    = engram.daily_spend.cost_usd + EXCLUDED.cost_usd
            """,
            today, tokens_in, tokens_out, cost,
        )
    # Fire change event so dashboard updates without polling
    try:
        from cogram.utils import events as _ev
        await _ev.publish(_ev.SPEND_CHANGE, {"kind": kind, "cost_usd": cost})
    except Exception:
        pass


async def today_summary() -> dict:
    pool = await _pg()
    if pool is None:
        return {"calls": 0, "cost_usd": 0.0, "backend": "none"}
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT calls, tokens_in, tokens_out, cost_usd "
            "FROM engram.daily_spend WHERE day=$1",
            date.today(),
        )
    if row is None:
        return {"calls": 0, "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0}
    return {
        "calls": int(row["calls"]),
        "tokens_in": int(row["tokens_in"]),
        "tokens_out": int(row["tokens_out"]),
        "cost_usd": float(row["cost_usd"]),
    }


async def is_over_cap() -> tuple[bool, str]:
    """Return (over_cap, reason)."""
    summary = await today_summary()
    if summary["calls"] >= MAX_DAILY_INGEST_CALLS:
        return True, f"daily call cap {MAX_DAILY_INGEST_CALLS} reached"
    if summary["cost_usd"] >= MAX_DAILY_USD:
        return True, f"daily $ cap ${MAX_DAILY_USD:.2f} reached"
    return False, ""

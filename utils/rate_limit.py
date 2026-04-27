"""Async rate limiter with two gates.

Every LLM/embedder call passes through TWO sliding windows:

  GLOBAL    — protects the upstream provider's per-minute quota
              (RATE_LIMIT_PER_MIN, default 150)
  PER-GROUP — protects against one noisy group_id starving everyone else
              (RATE_LIMIT_PER_GROUP_PER_MIN, default 50; 0 disables)

`acquire(group_id)` blocks while EITHER window is full. The default
`group_id="default"` keeps existing call sites working unchanged.

The per-group dict is purged opportunistically — every acquire trims
expired entries from the calling group, and groups whose latest entry
is older than WINDOW_SECONDS are dropped from the dict.

Patching: build_graphiti calls patch_clients(graphiti) which wraps the
LLM, embedder, and cross_encoder. Those wrappers don't yet thread a
group_id through (they fire from inside graphiti's extraction path
where group_id isn't easily reachable) so they always use 'default'.
The post-write maintenance modules pass their own group_id explicitly.
"""
import asyncio
import os
import time
from collections import defaultdict, deque
from typing import Deque, Dict


# ---------------------------------------------------------------------------
# Window config
# ---------------------------------------------------------------------------

_RATE = int(os.environ.get("RATE_LIMIT_PER_MIN", "150"))
MAX_PER_WINDOW = max(1, _RATE - 2)  # leave 2/min headroom

# 0 disables the per-group cap (purely global limiting). Default 50 — about
# 1/3 of the global cap so 3+ active groups can run in parallel without
# any one starving the others.
_GROUP_RATE = int(os.environ.get("RATE_LIMIT_PER_GROUP_PER_MIN", "50"))
MAX_PER_GROUP_WINDOW = 0 if _GROUP_RATE <= 0 else max(1, _GROUP_RATE - 2)

WINDOW_SECONDS = 60.0
MIN_INTERVAL = max(0.05, 60.0 / max(1, _RATE))

_lock = asyncio.Lock()
_history: Deque[float] = deque()
_per_group_history: Dict[str, Deque[float]] = defaultdict(deque)


def _purge_global(now: float) -> None:
    while _history and now - _history[0] > WINDOW_SECONDS:
        _history.popleft()


def _purge_group(group_id: str, now: float) -> None:
    q = _per_group_history.get(group_id)
    if not q:
        return
    while q and now - q[0] > WINDOW_SECONDS:
        q.popleft()
    if not q:
        _per_group_history.pop(group_id, None)


def _purge_idle_groups(now: float) -> None:
    """Drop dict entries for groups whose newest entry is older than the
    window — keeps the per-group history dict from growing unboundedly under
    workloads that create many short-lived group_ids."""
    stale = [gid for gid, q in _per_group_history.items() if not q or now - q[-1] > WINDOW_SECONDS]
    for gid in stale:
        _per_group_history.pop(gid, None)


async def acquire(group_id: str = "default") -> None:
    """Block until a call is safe per BOTH the global and per-group windows.

    `group_id` defaults to "default" so call sites that don't know their
    group still work — they share the "default" bucket, which is fine for
    most use. Pass a real group_id from per-episode call paths to get the
    fairness behavior."""
    async with _lock:
        while True:
            now = time.monotonic()
            _purge_global(now)
            _purge_group(group_id, now)

            # Periodic GC of dict entries for unrelated idle groups.
            if len(_per_group_history) > 32:
                _purge_idle_groups(now)

            # Spacing rule: at least MIN_INTERVAL since last GLOBAL call
            wait_spacing = 0.0
            if _history:
                wait_spacing = max(0.0, MIN_INTERVAL - (now - _history[-1]))

            # Global window-cap rule
            wait_window = 0.0
            if len(_history) >= MAX_PER_WINDOW:
                wait_window = WINDOW_SECONDS - (now - _history[0]) + 0.05

            # Per-group window-cap rule (0 disables)
            wait_group = 0.0
            if MAX_PER_GROUP_WINDOW > 0:
                gq = _per_group_history.get(group_id)
                if gq and len(gq) >= MAX_PER_GROUP_WINDOW:
                    wait_group = WINDOW_SECONDS - (now - gq[0]) + 0.05

            wait = max(wait_spacing, wait_window, wait_group)
            if wait <= 0:
                _history.append(now)
                _per_group_history[group_id].append(now)
                return
            await asyncio.sleep(wait)


def patch_clients(graphiti) -> None:
    """Wrap graphiti's llm_client and embedder so every call awaits the limiter.

    These wrappers can't thread a group_id (graphiti extraction doesn't
    surface group_id at the LLM-client boundary), so they use 'default'.
    Per-group fairness applies to the post-write maintenance modules that
    pass group_id through to acquire() explicitly."""
    llm = graphiti.llm_client
    if not getattr(llm, "_rate_limited", False):
        original_generate = llm.generate_response

        async def gated_generate(*args, **kwargs):
            await acquire()
            return await original_generate(*args, **kwargs)

        llm.generate_response = gated_generate
        llm._rate_limited = True

    emb = graphiti.embedder
    if not getattr(emb, "_rate_limited", False):
        original_create = emb.create

        async def gated_create(*args, **kwargs):
            await acquire()
            return await original_create(*args, **kwargs)

        emb.create = gated_create
        emb._rate_limited = True

    cross = getattr(graphiti, "cross_encoder", None)
    if cross is not None and not getattr(cross, "_rate_limited", False):
        rank = getattr(cross, "rank", None)
        if rank is not None:
            async def gated_rank(*args, **kwargs):
                await acquire()
                return await rank(*args, **kwargs)
            cross.rank = gated_rank
        cross._rate_limited = True

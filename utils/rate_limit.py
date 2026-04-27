"""Global async rate limiter for NIM's 40 req/min ceiling.

Patches graphiti's LLM and embedder clients so every outbound NIM call —
including the bursts graphiti makes internally per add_episode — gets paced.
"""
import asyncio
import time
from collections import deque

import os

# Reads RATE_LIMIT_PER_MIN from env. Default 150 (OpenAI tier 1 headroom).
# NIM free tier should set RATE_LIMIT_PER_MIN=38 in .env.
_RATE = int(os.environ.get("RATE_LIMIT_PER_MIN", "150"))
MAX_PER_WINDOW = max(1, _RATE - 2)  # leave 2/min headroom
WINDOW_SECONDS = 60.0
MIN_INTERVAL = max(0.05, 60.0 / max(1, _RATE))

_lock = asyncio.Lock()
_history: deque[float] = deque()


async def acquire() -> None:
    """Block until a NIM call is safe per the 40/min budget."""
    async with _lock:
        while True:
            now = time.monotonic()
            # Drop calls older than the window
            while _history and now - _history[0] > WINDOW_SECONDS:
                _history.popleft()

            # Spacing rule: at least MIN_INTERVAL since last call
            wait_spacing = 0.0
            if _history:
                wait_spacing = max(0.0, MIN_INTERVAL - (now - _history[-1]))

            # Window cap rule: if window full, wait until the oldest expires
            wait_window = 0.0
            if len(_history) >= MAX_PER_WINDOW:
                wait_window = WINDOW_SECONDS - (now - _history[0]) + 0.05

            wait = max(wait_spacing, wait_window)
            if wait <= 0:
                _history.append(now)
                return
            await asyncio.sleep(wait)


def patch_clients(graphiti) -> None:
    """Wrap graphiti's llm_client and embedder so every call awaits the limiter."""
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

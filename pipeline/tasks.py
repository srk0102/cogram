"""In-process registry for async cogram_post_write tasks.

When the MCP server runs in async pipeline mode (the default), `add_episode`
returns in ~3s and the post-write pipeline (annotation, narration, profile
distill, knot synthesis) runs in the background. Without observability hooks,
clients can't tell whether the background work finished, is still running, or
crashed.

This module provides a process-local registry so the three task-management MCP
tools (`list_episode_tasks`, `get_episode_task`, `cancel_episode_task`) can
introspect and control those background tasks.

Scope:
  - In-memory only. Restarting the MCP server clears the registry.
  - The registry keeps the most recent COGRAM_TASK_HISTORY (default 200) records
    so completed tasks remain inspectable for a while; older records are evicted
    in FIFO order.
  - Cross-process visibility (e.g. multiple MCP workers) is out of scope. If
    you scale horizontally, route a given group_id sticky to one worker.
"""
from __future__ import annotations

import asyncio
import os
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional


TASK_HISTORY_LIMIT = int(os.environ.get("COGRAM_TASK_HISTORY", "200"))


@dataclass
class TaskRecord:
    task_id: str
    episode_name: str
    group_id: str
    state: str  # 'running' | 'done' | 'failed' | 'cancelled'
    created_at: float
    finished_at: Optional[float] = None
    summary: Optional[dict] = None
    error: Optional[str] = None
    handle: Optional[asyncio.Task] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "episode_name": self.episode_name,
            "group_id": self.group_id,
            "state": self.state,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "elapsed_ms": (
                round((self.finished_at - self.created_at) * 1000, 1)
                if self.finished_at
                else round((time.time() - self.created_at) * 1000, 1)
            ),
            "summary": self.summary,
            "error": self.error,
        }


_records: "OrderedDict[str, TaskRecord]" = OrderedDict()
_lock = asyncio.Lock()


def _evict_if_needed() -> None:
    """Drop oldest finished records once the registry exceeds the cap.
    Running tasks are never evicted; only done/failed/cancelled."""
    if len(_records) <= TASK_HISTORY_LIMIT:
        return
    finished_ids = [tid for tid, rec in _records.items() if rec.state != "running"]
    excess = len(_records) - TASK_HISTORY_LIMIT
    for tid in finished_ids[:excess]:
        _records.pop(tid, None)


def new_task_id() -> str:
    """Allocate a task_id without yet registering a handle. Use this when the
    caller needs to know the id before the asyncio.Task is created (so the
    coroutine can update the record via mark_done/mark_failed)."""
    return uuid.uuid4().hex[:12]


def register(
    handle: asyncio.Task,
    episode_name: str,
    group_id: str,
    task_id: Optional[str] = None,
) -> str:
    """Record a newly spawned background pipeline task. Returns the task_id.
    If task_id is provided (e.g. from new_task_id()), the record is created
    under that id; otherwise a fresh id is allocated."""
    if task_id is None:
        task_id = new_task_id()
    rec = TaskRecord(
        task_id=task_id,
        episode_name=episode_name,
        group_id=group_id,
        state="running",
        created_at=time.time(),
        handle=handle,
    )
    _records[task_id] = rec
    _evict_if_needed()
    return task_id


def mark_done(task_id: str, summary: dict) -> None:
    rec = _records.get(task_id)
    if rec is None:
        return
    rec.state = "done"
    rec.finished_at = time.time()
    rec.summary = summary


def mark_failed(task_id: str, error: str) -> None:
    rec = _records.get(task_id)
    if rec is None:
        return
    rec.state = "failed"
    rec.finished_at = time.time()
    rec.error = error


def mark_cancelled(task_id: str) -> None:
    rec = _records.get(task_id)
    if rec is None:
        return
    rec.state = "cancelled"
    rec.finished_at = time.time()


def get(task_id: str) -> Optional[TaskRecord]:
    return _records.get(task_id)


def list_tasks(
    group_id: Optional[str] = None,
    state: Optional[str] = None,
    limit: int = 50,
) -> list[TaskRecord]:
    """Newest-first listing. Filters by group_id and/or state when provided."""
    items = list(_records.values())
    items.reverse()
    if group_id:
        items = [r for r in items if r.group_id == group_id]
    if state:
        items = [r for r in items if r.state == state]
    return items[:limit]


async def wait(task_id: str, timeout: float) -> TaskRecord:
    """Block until the task reaches a terminal state OR timeout elapses.
    Returns the (possibly still-running) TaskRecord either way; caller checks state.
    Raises KeyError if task_id isn't known."""
    rec = _records.get(task_id)
    if rec is None:
        raise KeyError(task_id)
    if rec.state != "running" or rec.handle is None:
        return rec
    try:
        await asyncio.wait_for(asyncio.shield(rec.handle), timeout=timeout)
    except asyncio.TimeoutError:
        pass
    return rec


def cancel(task_id: str) -> bool:
    """Request cancellation. Returns True if a running task was cancelled,
    False if the task was already terminal or unknown."""
    rec = _records.get(task_id)
    if rec is None:
        return False
    if rec.state != "running" or rec.handle is None:
        return False
    rec.handle.cancel()
    return True

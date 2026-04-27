"""COGRAM PHASE 5: Continuous watcher.

Tails ~/.claude/projects/<project>/*.jsonl files. When new turns appear,
converts them to per-turn episodes and ingests them automatically. The graph
grows as you have conversations — no `.\\run.bat` needed.

Architecture:
  - Poll every WATCH_POLL_SECONDS (default 15s)
  - Per-file byte offset tracked in cache/watcher_state.json
  - Debounce: wait WATCH_DEBOUNCE_SECONDS after last write before processing
    (so we don't ingest mid-turn)
  - Daily call cap: MAX_DAILY_INGEST_CALLS (default 300) prevents runaway cost
  - Cap resets at midnight UTC

Configure via .env or env vars:
  WATCH_GLOBS         — semicolon-separated globs (default: ~/.claude/projects/*/*.jsonl)
  WATCH_POLL_SECONDS  — how often to check for changes (default 15)
  WATCH_DEBOUNCE_SECONDS — quiet period before processing (default 30)
  MAX_DAILY_INGEST_CALLS — hard cap on per-day LLM-spending operations (default 300)

Run:
    .venv\\Scripts\\python.exe -m src.watch
"""
from __future__ import annotations

import asyncio
import glob
import json
import os
import re
import signal
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cogram.core.nodes import EpisodeType
from cogram.core.config import build_graphiti, Settings
from cogram.pipeline.import_transcripts import (
    _flatten_content,
    _parse_timestamp,
    _is_substantive,
    _pair_into_exchanges,
    _sanitize_group_id,
    _to_episode,
    Turn,
    EPISODES_DIR,
    MAX_EXCHANGE_CHARS,
)
from cogram.utils import budget
from cogram.utils.maintenance import drift_detection as drift as drift_mod
from cogram.utils.maintenance import node_narration as narration as node_narrator


STATE_PATH = Path(__file__).resolve().parent.parent / "cache" / "watcher_state.json"
DEFAULT_GLOBS = [
    str(Path.home() / ".claude" / "projects" / "*" / "*.jsonl"),
]
POLL = float(os.environ.get("WATCH_POLL_SECONDS", "15"))
DEBOUNCE = float(os.environ.get("WATCH_DEBOUNCE_SECONDS", "30"))
DAILY_CAP = int(os.environ.get("MAX_DAILY_INGEST_CALLS", "300"))


# ---- state ---------------------------------------------------------------


@dataclass
class FileState:
    last_byte: int = 0
    last_seq: int = 0
    last_mtime: float = 0.0


def _load_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return {"files": {}, "day": "", "calls_today": 0}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"files": {}, "day": "", "calls_today": 0}


def _save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


# ---- per-file processing -------------------------------------------------


def _read_new_turns(path: Path, last_byte: int) -> tuple[list[Turn], int]:
    """Read .jsonl from offset; return new turns + new offset."""
    if not path.exists():
        return [], last_byte
    size = path.stat().st_size
    if size <= last_byte:
        return [], last_byte
    turns: list[Turn] = []
    with path.open("rb") as f:
        f.seek(last_byte)
        chunk = f.read()
    new_offset = last_byte + len(chunk)
    text = chunk.decode("utf-8", errors="replace")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        msg = entry.get("message") or {}
        role = msg.get("role") or entry.get("type")
        if role not in {"user", "assistant"}:
            continue
        body, only_tool = _flatten_content(msg.get("content", ""))
        if body.startswith("[Request interrupted"):
            continue
        turns.append(
            Turn(
                role=role,
                text=body,
                timestamp=_parse_timestamp(entry),
                has_only_tool_blocks=only_tool,
            )
        )
    return turns, new_offset


# ---- main loop -----------------------------------------------------------


_stop = False


def _handle_signal(*_args) -> None:
    global _stop
    _stop = True
    print("\n[watcher] stop requested; finishing current pass...")


async def _process_file(graphiti, path: Path, file_state: FileState) -> int:
    """Return number of new exchanges ingested."""
    new_turns, new_offset = _read_new_turns(path, file_state.last_byte)
    if not new_turns:
        return 0
    project = _sanitize_group_id(path.parent.name)
    session_safe = _sanitize_group_id(path.stem)

    exchanges = _pair_into_exchanges(new_turns)
    if not exchanges:
        file_state.last_byte = new_offset
        return 0

    EPISODES_DIR.mkdir(exist_ok=True)
    written = 0
    for user, assistant in exchanges:
        file_state.last_seq += 1
        seq = file_state.last_seq
        result = _to_episode(user, assistant, group_id=project, seq=seq)
        if result is None:
            continue
        body, _ts = result
        fname = f"{project}__{session_safe}__{seq:04d}.md"
        dest = EPISODES_DIR / fname
        if dest.exists():
            continue
        dest.write_text(body, encoding="utf-8")

        # Ingest immediately
        ref_time = user.timestamp or assistant.timestamp or datetime.now(timezone.utc)
        try:
            await graphiti.add_episode(
                name=dest.stem,
                episode_body=body.split("---", 2)[-1].strip(),  # strip frontmatter for graphiti
                source=EpisodeType.text,
                source_description=path.name,
                reference_time=ref_time,
                previous_episode_uuids=[],
                group_id=project,
            )
            written += 1
            print(f"  [watch] ingested {dest.name}")
        except Exception as exc:
            print(f"  [watch] FAILED {dest.name}: {exc}")

    file_state.last_byte = new_offset
    return written


async def main() -> None:
    signal.signal(signal.SIGINT, _handle_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_signal)

    state = _load_state()
    if state.get("day") != _today_utc():
        state["day"] = _today_utc()
        state["calls_today"] = 0

    globs_raw = os.environ.get("WATCH_GLOBS")
    globs = globs_raw.split(";") if globs_raw else DEFAULT_GLOBS

    print(f"[watcher] watching {globs}")
    print(f"[watcher] poll={POLL}s debounce={DEBOUNCE}s daily_cap={DAILY_CAP}")
    print(f"[watcher] state at {STATE_PATH}")

    settings = Settings.from_env()
    graphiti = build_graphiti(settings)
    await graphiti.build_indices_and_constraints()

    try:
        while not _stop:
            # Daily cap reset
            if state.get("day") != _today_utc():
                state["day"] = _today_utc()
                state["calls_today"] = 0

            # Local in-process counter
            if state["calls_today"] >= DAILY_CAP:
                print(f"[watcher] daily cap {DAILY_CAP} reached (local); sleeping until midnight")
                await asyncio.sleep(60)
                continue

            # Postgres-backed budget cap (catches cross-process spend)
            try:
                over, reason = await budget.is_over_cap()
                if over:
                    print(f"[watcher] {reason}; sleeping 60s")
                    await asyncio.sleep(60)
                    continue
            except Exception:
                pass  # Postgres might not be reachable yet at startup

            now = time.time()
            for pattern in globs:
                for match in glob.glob(os.path.expanduser(os.path.expandvars(pattern))):
                    path = Path(match)
                    if not path.is_file():
                        continue
                    mtime = path.stat().st_mtime
                    # Debounce: skip files that were just written
                    if now - mtime < DEBOUNCE:
                        continue
                    rec = state["files"].get(str(path), {"last_byte": 0, "last_seq": 0, "last_mtime": 0.0})
                    if mtime <= rec.get("last_mtime", 0):
                        continue
                    fs = FileState(**{k: rec.get(k, FileState.__dataclass_fields__[k].default)
                                      for k in FileState.__dataclass_fields__})
                    written = await _process_file(graphiti, path, fs)
                    fs.last_mtime = mtime
                    state["files"][str(path)] = asdict(fs)
                    state["calls_today"] += written
                    if written:
                        _save_state(state)

                        # Trigger drift-gated narration for this group
                        try:
                            project = _sanitize_group_id(path.parent.name)
                            events = await node_narrator.narrate_group(
                                graphiti, group_id=project, settings=settings, limit=5,
                            )
                            for ev in events:
                                if ev.get("ok"):
                                    print(f"  [narr] {ev.get('name','?')} via {ev.get('source','?')}")
                        except Exception as exc:
                            print(f"  [narr] failed: {exc}")
                    if state["calls_today"] >= DAILY_CAP:
                        break

            await asyncio.sleep(POLL)
    finally:
        _save_state(state)
        await graphiti.close()
        print("[watcher] shut down cleanly")


def cli() -> None:
    """Sync entry point for the console script (cogram-watch)."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    cli()

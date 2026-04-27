"""
Convert Claude Code session transcripts (.jsonl) into PER-TURN episode files.

One .md per user/assistant exchange. Frontmatter carries group_id and timestamp
so the ingest step can stamp episodes correctly.

Filters out:
  - Tool-result-only user messages (graphiti has nothing to extract from raw tool output)
  - Pure tool-use assistant messages (no substantive text)
  - Empty or [REDACTED]-only content
  - Exchanges over 8k chars where the bulk is a file paste

Usage (PowerShell):
    .venv\\Scripts\\python.exe -m src.import_transcripts "$env:USERPROFILE\\.claude\\projects\\d--graphiti-with-meta\\*.jsonl"

    # Limit for cheap testing:
    .venv\\Scripts\\python.exe -m src.import_transcripts --limit 80 "$env:USERPROFILE\\.claude\\projects\\d--graphiti-with-meta\\*.jsonl"

    # Multiple projects:
    .venv\\Scripts\\python.exe -m src.import_transcripts --limit 100 \\
        "$env:USERPROFILE\\.claude\\projects\\d--graphiti-with-meta\\*.jsonl" \\
        "$env:USERPROFILE\\.claude\\projects\\D--AdHopper\\*.jsonl"
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

EPISODES_DIR = Path(__file__).resolve().parent.parent / "episodes"
MAX_EXCHANGE_CHARS = 8_000  # skip exchanges bigger than this (likely file pastes)
MIN_TEXT_CHARS = 30  # skip exchanges with almost no meaningful content


@dataclass
class Turn:
    role: str  # "user" or "assistant"
    text: str
    timestamp: datetime | None
    has_only_tool_blocks: bool


def _flatten_content(content) -> tuple[str, bool]:
    """Return (clean_text, only_tool_blocks). Strips tool_result, tool_use, thinking blocks."""
    if isinstance(content, str):
        return content, False
    if not isinstance(content, list):
        return str(content), False
    text_parts: list[str] = []
    saw_text = False
    saw_tool_only = True
    for block in content:
        if not isinstance(block, dict):
            text_parts.append(str(block))
            saw_text = True
            saw_tool_only = False
            continue
        kind = block.get("type")
        if kind == "text":
            t = block.get("text", "")
            if t.strip():
                text_parts.append(t)
                saw_text = True
                saw_tool_only = False
        elif kind == "tool_use":
            # Don't emit tool calls as content; mark this as tool-leaning
            pass
        elif kind == "tool_result":
            # User-side tool result — skip entirely
            pass
        elif kind == "thinking":
            # Reasoning trace — skip
            pass
        else:
            t = block.get("text") or block.get("content")
            if isinstance(t, str) and t.strip():
                text_parts.append(t)
                saw_text = True
                saw_tool_only = False
    return ("\n".join(text_parts).strip(), not saw_text and not saw_tool_only is False)


def _parse_timestamp(entry: dict) -> datetime | None:
    raw = entry.get("timestamp")
    if not raw:
        return None
    try:
        # Claude Code format: "2025-11-22T13:45:30.123Z"
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        return datetime.fromisoformat(raw)
    except (TypeError, ValueError):
        return None


def _parse_jsonl(path: Path) -> list[Turn]:
    turns: list[Turn] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
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

        text, only_tool = _flatten_content(msg.get("content", ""))
        if text.startswith("[Request interrupted"):
            continue

        turns.append(
            Turn(
                role=role,
                text=text,
                timestamp=_parse_timestamp(entry),
                has_only_tool_blocks=only_tool,
            )
        )
    return turns


def _is_substantive(turn: Turn) -> bool:
    """True if this turn carries enough text to be worth extracting from."""
    if not turn.text or len(turn.text.strip()) < MIN_TEXT_CHARS:
        return False
    # Pure-redacted content is noise
    stripped = turn.text.strip()
    if stripped == "[REDACTED]" or stripped.replace("[REDACTED]", "").strip() == "":
        return False
    return True


def _pair_into_exchanges(turns: list[Turn]) -> list[tuple[Turn, Turn]]:
    """Pair user → next-assistant. Skip user turns with only tool blocks (tool result follow-ups)."""
    exchanges: list[tuple[Turn, Turn]] = []
    i = 0
    while i < len(turns):
        if turns[i].role != "user" or not _is_substantive(turns[i]):
            i += 1
            continue
        # find next assistant turn
        j = i + 1
        assistant: Turn | None = None
        while j < len(turns):
            if turns[j].role == "assistant" and _is_substantive(turns[j]):
                assistant = turns[j]
                break
            j += 1
        if assistant is not None:
            exchanges.append((turns[i], assistant))
            i = j + 1
        else:
            i += 1
    return exchanges


def _sanitize_group_id(name: str) -> str:
    """FalkorDB and several other providers require alphanumeric / dash / underscore only."""
    out = re.sub(r"[^A-Za-z0-9_-]", "_", name)
    return out.strip("_") or "default"


def _to_episode(user: Turn, assistant: Turn, group_id: str, seq: int) -> tuple[str, str] | None:
    """Return (filename, body) for a single exchange, or None if filtered out."""
    body_text = f"## user\n\n{user.text.strip()}\n\n## assistant\n\n{assistant.text.strip()}\n"
    if len(body_text) > MAX_EXCHANGE_CHARS:
        return None

    # Use user's timestamp as the canonical reference_time for the exchange
    ts = user.timestamp or assistant.timestamp or datetime.now(timezone.utc)
    ts_str = ts.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    frontmatter = (
        "---\n"
        f"group_id: {group_id}\n"
        f"timestamp: {ts_str}\n"
        f"role: exchange\n"
        f"seq: {seq:04d}\n"
        "---\n\n"
    )
    return frontmatter + body_text, ts_str


def main(patterns: list[str], limit: int | None = None) -> None:
    EPISODES_DIR.mkdir(exist_ok=True)
    seen = 0
    written = 0

    for pattern in patterns:
        for match in glob.glob(os.path.expandvars(os.path.expanduser(pattern))):
            seen += 1
            src = Path(match)
            if not src.is_file():
                continue

            project = _sanitize_group_id(src.parent.name)
            session = src.stem  # uuid-like
            session_safe = _sanitize_group_id(session)

            turns = _parse_jsonl(src)
            exchanges = _pair_into_exchanges(turns)

            print(f"  {src.name}: {len(turns)} turns -> {len(exchanges)} exchanges")

            for seq, (user, assistant) in enumerate(exchanges, 1):
                if limit is not None and written >= limit:
                    print(f"\nReached --limit {limit}; stopping.")
                    print(f"Processed {seen} file(s); wrote {written} episode(s) to {EPISODES_DIR}")
                    return
                result = _to_episode(user, assistant, group_id=project, seq=seq)
                if result is None:
                    continue
                body, _ts = result
                fname = f"{project}__{session_safe}__{seq:04d}.md"
                dest = EPISODES_DIR / fname
                if dest.exists():
                    # idempotent: skip already-converted exchanges
                    continue
                dest.write_text(body, encoding="utf-8")
                written += 1

    print(f"\nProcessed {seen} file(s); wrote {written} new episode(s) to {EPISODES_DIR}")


def cli() -> None:
    """Console-script entry point (cogram-import)."""
    parser = argparse.ArgumentParser(description="Convert Claude Code transcripts to per-turn episodes.")
    parser.add_argument("patterns", nargs="+", help="Glob pattern(s) for .jsonl files")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after writing N exchanges (cheap testing). Default: no limit.",
    )
    args = parser.parse_args()
    main(args.patterns, limit=args.limit)


if __name__ == "__main__":
    cli()

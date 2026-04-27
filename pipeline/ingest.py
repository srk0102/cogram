import asyncio
import re
from datetime import datetime, timezone
from pathlib import Path

from graphiti_core.nodes import EpisodeType

from cogram.core.config import build_graphiti

EPISODES_DIR = Path(__file__).resolve().parent.parent / "episodes"
MAX_CHARS = 8_000  # per-turn episodes rarely exceed this; bigger ones get chunked


# ---- Frontmatter parsing ----------------------------------------------------

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    raw = m.group(1)
    body = text[m.end():]
    meta: dict[str, str] = {}
    for line in raw.splitlines():
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        meta[k.strip()] = v.strip()
    return meta, body


def _ref_time_from_meta(meta: dict[str, str], fallback_path: Path) -> datetime:
    raw = meta.get("timestamp")
    if raw:
        try:
            if raw.endswith("Z"):
                raw = raw[:-1] + "+00:00"
            return datetime.fromisoformat(raw)
        except ValueError:
            pass
    return datetime.fromtimestamp(fallback_path.stat().st_mtime, tz=timezone.utc)


def _group_id_from_meta(meta: dict[str, str], fallback_filename: str) -> str:
    raw = meta.get("group_id")
    if raw:
        return _sanitize(raw)
    # Fall back to first chunk of filename before "__"
    stem = fallback_filename.rsplit(".", 1)[0]
    head = stem.split("__", 1)[0]
    return _sanitize(head) or "default"


def _sanitize(name: str) -> str:
    out = re.sub(r"[^A-Za-z0-9_-]", "_", name)
    return out.strip("_")


# ---- Chunking (only for oversized episodes; per-turn shouldn't need this) ---


def _chunk(body: str, limit: int = MAX_CHARS) -> list[str]:
    if len(body) <= limit:
        return [body]
    chunks: list[str] = []
    paragraphs = body.split("\n\n")
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 > limit and current:
            chunks.append(current.strip())
            current = ""
        if len(para) > limit:
            for i in range(0, len(para), limit):
                chunks.append(para[i : i + limit].strip())
            continue
        current += para + "\n\n"
    if current.strip():
        chunks.append(current.strip())
    return chunks


# ---- Idempotent ingest ------------------------------------------------------


async def _episode_exists(graphiti, name: str) -> bool:
    async with graphiti.driver.session() as session:
        result = await session.run(
            "MATCH (e:Episodic {name: $name}) RETURN e LIMIT 1", name=name
        )
        record = await result.single()
        return record is not None


async def _ingest_with_backoff(
    graphiti,
    name: str,
    chunk: str,
    source_desc: str,
    ref_time: datetime,
    group_id: str,
) -> None:
    delays = [10, 30, 60, 120, 240]
    for attempt, wait in enumerate([0, *delays]):
        if wait:
            print(f"    rate-limited; sleeping {wait}s before retry {attempt}/{len(delays)}...")
            await asyncio.sleep(wait)
        try:
            await graphiti.add_episode(
                name=name,
                episode_body=chunk,
                source=EpisodeType.text,
                source_description=source_desc,
                reference_time=ref_time,
                previous_episode_uuids=[],
                group_id=group_id,
            )
            print(f"    done")
            return
        except Exception as exc:
            msg = str(exc)
            is_rate_limited = "429" in msg or "Too Many Requests" in msg or "rate" in msg.lower()
            if not is_rate_limited:
                print(f"    FAILED (non-retryable): {exc}")
                return
            if attempt == len(delays):
                print(f"    FAILED (after {attempt} retries): {exc}")
                return


async def main() -> None:
    graphiti = build_graphiti()
    await graphiti.build_indices_and_constraints()

    all_files = sorted(p for p in EPISODES_DIR.glob("*") if p.suffix.lower() in {".md", ".txt"})

    # Phase 1 only ingests:
    #   1. Files with YAML frontmatter (per-turn episodes from import_transcripts)
    #   2. Hand-curated sample_*.md files
    # Old chunked transcript_*.md files are skipped to keep the graph clean.
    files: list[Path] = []
    skipped: list[str] = []
    for p in all_files:
        head = p.read_text(encoding="utf-8")[:4]
        if head.startswith("---") or p.name.startswith("sample_"):
            files.append(p)
        else:
            skipped.append(p.name)

    if skipped:
        print(f"Skipping {len(skipped)} legacy file(s) without frontmatter: {skipped[0]}, ...")
        print("  (move them out of episodes/ if you want them gone)")

    if not files:
        print(f"No episodes found in {EPISODES_DIR}")
        return

    print(f"Ingesting {len(files)} episode file(s) from {EPISODES_DIR}")

    for path in files:
        raw = path.read_text(encoding="utf-8")
        meta, body = _parse_frontmatter(raw)
        ref_time = _ref_time_from_meta(meta, path)
        group_id = _group_id_from_meta(meta, path.name)

        chunks = _chunk(body)
        if len(chunks) == 1:
            names = [path.stem]
        else:
            names = [f"{path.stem}_part{i+1:03d}" for i in range(len(chunks))]
            print(f"Splitting {path.name} ({len(body)} chars) into {len(chunks)} chunks")

        for name, chunk in zip(names, chunks):
            if await _episode_exists(graphiti, name):
                continue  # silent skip — per-turn means many already-ingested files
            print(f"  ingesting {name} (group={group_id}, {len(chunk)} chars)...")
            await _ingest_with_backoff(
                graphiti,
                name=name,
                chunk=chunk,
                source_desc=path.name,
                ref_time=ref_time,
                group_id=group_id,
            )

    await graphiti.close()


if __name__ == "__main__":
    asyncio.run(main())

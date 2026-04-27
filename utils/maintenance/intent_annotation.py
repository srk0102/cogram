"""Edge intent annotation.

Two entrypoints:
  1. CLI back-fill (`python -m src.annotate`) — scans every un-annotated edge
     in the graph and writes intent_meta on each.
  2. Per-episode hook (`annotate_edges_for_episode`) — used by cogram_pipeline
     after a fresh add_episode call. Only annotates edges that reference the
     specific episode that just landed.
"""
import asyncio
import json
import re

from openai import AsyncOpenAI

from cogram.core.config import Settings, build_graphiti
from cogram.utils.rate_limit import acquire as _rate_acquire

ANNOTATION_PROMPT = """You are reading a single edge from a personal knowledge graph
that was extracted from the Director's own conversations. The edge connects two
entities and was created because something the Director said linked them.

Node A ({a_name}): {a_summary}
Node B ({b_name}): {b_summary}
Edge fact: {fact}
Source episode excerpts:
---
{episodes}
---

Answer in strict JSON, no prose, no markdown fences:
{{
  "why_connected": "<one sentence: the underlying reason the Director linked these two>",
  "director_vision": "<one sentence: the larger goal or outcome this link serves>",
  "cognitive_pattern": "<2-5 word label for the thinking style this reveals, e.g. 'cost-aware prototyping', 'first-principles validation'>"
}}"""

EDGES_QUERY_ALL = """
MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
WHERE r.intent_meta IS NULL
RETURN
  elementId(r) AS edge_id,
  a.name AS a_name,
  coalesce(a.summary, '') AS a_summary,
  b.name AS b_name,
  coalesce(b.summary, '') AS b_summary,
  coalesce(r.fact, r.name, '') AS fact,
  coalesce(r.episodes, []) AS episode_uuids
"""

EDGES_QUERY_BY_EPISODE = """
MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
WHERE r.intent_meta IS NULL
  AND $episode_uuid IN coalesce(r.episodes, [])
RETURN
  elementId(r) AS edge_id,
  a.name AS a_name,
  coalesce(a.summary, '') AS a_summary,
  b.name AS b_name,
  coalesce(b.summary, '') AS b_summary,
  coalesce(r.fact, r.name, '') AS fact,
  coalesce(r.episodes, []) AS episode_uuids
"""

EPISODES_QUERY = """
MATCH (e:Episodic) WHERE e.uuid IN $uuids
RETURN coalesce(e.content, '') AS content
"""

WRITE_QUERY = """
MATCH ()-[r]->() WHERE elementId(r) = $edge_id
SET r.intent_meta = $meta_json
"""


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object in response: {text[:200]}")
    return json.loads(text[start : end + 1])


async def _annotate_one(client: AsyncOpenAI, model: str, row: dict, episode_text: str) -> dict:
    prompt = ANNOTATION_PROMPT.format(
        a_name=row["a_name"],
        a_summary=row["a_summary"][:500],
        b_name=row["b_name"],
        b_summary=row["b_summary"][:500],
        fact=row["fact"][:500],
        episodes=episode_text[:3000] or "(no source episodes available)",
    )
    await _rate_acquire()
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        max_tokens=4096,
    )
    msg = resp.choices[0].message
    text = (msg.content or "") or (getattr(msg, "reasoning_content", None) or "")
    return _extract_json(text)


async def _fetch_episode_text(graphiti, episode_uuids: list[str]) -> str:
    if not episode_uuids:
        return ""
    async with graphiti.driver.session() as session:
        ep_rows = [
            r.data()
            async for r in await session.run(EPISODES_QUERY, uuids=episode_uuids)
        ]
    return "\n\n".join(r["content"] for r in ep_rows if r["content"])


async def _annotate_rows(
    graphiti,
    rows: list[dict],
    settings: Settings,
    verbose: bool = False,
) -> int:
    """Annotate a batch of edge rows. Returns count successfully annotated."""
    if not rows:
        return 0

    client = AsyncOpenAI(api_key=settings.api_key, base_url=settings.base_url)
    annotated = 0

    for i, row in enumerate(rows, 1):
        episode_text = await _fetch_episode_text(graphiti, row.get("episode_uuids") or [])

        try:
            meta = await _annotate_one(client, settings.annotator_llm_model, row, episode_text)
        except Exception as exc:
            if verbose:
                print(f"  [{i}/{len(rows)}] {row['a_name']} -> {row['b_name']}: FAILED ({exc})")
            continue

        async with graphiti.driver.session() as session:
            await session.run(
                WRITE_QUERY, edge_id=row["edge_id"], meta_json=json.dumps(meta)
            )
        annotated += 1
        if verbose:
            print(f"  [{i}/{len(rows)}] {row['a_name']} -> {row['b_name']}: {meta.get('cognitive_pattern')}")

        # Push to dashboard (best-effort)
        try:
            from cogram.utils import events as _ev
            await _ev.publish(_ev.INTENT_ANNOTATED, {
                "edge": f"{row['a_name']} -> {row['b_name']}",
                "pattern": meta.get("cognitive_pattern", ""),
            })
        except Exception:
            pass

    return annotated


async def annotate_edges_for_episode(
    graphiti,
    episode_uuid: str,
    settings: Settings,
) -> int:
    """Annotate every un-annotated edge that references this episode.

    Used by cogram_pipeline as a post-write hook. Returns count annotated.
    """
    async with graphiti.driver.session() as session:
        rows = [
            r.data()
            async for r in await session.run(
                EDGES_QUERY_BY_EPISODE, episode_uuid=episode_uuid
            )
        ]
    return await _annotate_rows(graphiti, rows, settings, verbose=False)


async def main() -> None:
    """CLI back-fill mode: annotate every un-annotated edge in the graph."""
    settings = Settings.from_env()
    graphiti = build_graphiti(settings)

    async with graphiti.driver.session() as session:
        rows = [r.data() async for r in await session.run(EDGES_QUERY_ALL)]

    if not rows:
        print("No un-annotated edges found.")
        await graphiti.close()
        return

    print(f"Annotating {len(rows)} edge(s) with {settings.annotator_llm_model}...")
    annotated = await _annotate_rows(graphiti, rows, settings, verbose=True)
    print(f"Done: {annotated}/{len(rows)} edges annotated.")
    await graphiti.close()


if __name__ == "__main__":
    asyncio.run(main())

"""COGRAM PROFILE — distill annotated edges into the Director profile.

Two outputs (both updated atomically):
  1. JSON file at director_profile.json (legacy / human-readable)
  2. Graph nodes in Neo4j:
       (:DirectorProfile {summary, generated_at, group_id})
       (:CognitivePattern {name, count, last_reinforced, group_id})
       (:DirectorProfile)-[:HAS_PATTERN]->(:CognitivePattern)
       (:CognitivePattern)-[:REINFORCED_BY]->(:Entity)
                          (one for each edge whose intent_meta uses this pattern)

The graph form lets retrieval do real Cypher traversal:
  MATCH (p:DirectorProfile)-[:HAS_PATTERN]->(pat:CognitivePattern)
        -[:REINFORCED_BY]->(e:Entity)
  ...with effective-confidence weighting via the decay function.
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from openai import AsyncOpenAI

from cogram.core.config import Settings, build_graphiti
from cogram.utils.rate_limit import acquire as _rate_acquire

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "director_profile.json"

COLLECT_QUERY = """
MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
WHERE r.intent_meta IS NOT NULL
RETURN a.name AS a, a.uuid AS a_uuid,
       b.name AS b, b.uuid AS b_uuid,
       coalesce(r.fact, r.name, '') AS fact,
       r.intent_meta AS meta,
       coalesce(a.group_id, b.group_id, 'default') AS group_id
"""

COMPRESSION_PROMPT = """You are given every annotated edge from the Director's
knowledge graph. Each edge has a "why_connected", "director_vision", and
"cognitive_pattern". Synthesize them into a single profile.

Edges (JSON list):
{edges}

Return strict JSON, no markdown:
{{
  "cognitive_patterns": ["<deduplicated, ranked by frequency>"],
  "recurring_visions": ["<2-6 distinct goals the Director repeatedly pursues>"],
  "working_style_summary": "<3-5 sentences in second person ('You ...') describing how the Director approaches problems, what they value, what they avoid>"
}}"""


# ---------------------------------------------------------------------------
# Graph upsert: persist :DirectorProfile + :CognitivePattern nodes
# ---------------------------------------------------------------------------

UPSERT_PROFILE_QUERY = """
MERGE (p:DirectorProfile {group_id: $group_id})
SET p.summary = $summary,
    p.generated_at = $now,
    p.visions = $visions,
    p.updated_at = $now
RETURN p.group_id AS group_id
"""

UPSERT_PATTERN_QUERY = """
MERGE (pat:CognitivePattern {name: $name, group_id: $group_id})
ON CREATE SET pat.count = $count,
              pat.first_seen = $now,
              pat.last_reinforced = $now,
              pat.confidence = $count
ON MATCH SET pat.count = $count,
             pat.last_reinforced = $now,
             pat.confidence = pat.confidence + 1
WITH pat
MATCH (p:DirectorProfile {group_id: $group_id})
MERGE (p)-[:HAS_PATTERN]->(pat)
RETURN pat.name AS name, pat.confidence AS confidence
"""

LINK_PATTERN_TO_EDGES_QUERY = """
MATCH (a:Entity {uuid: $a_uuid})-[r:RELATES_TO]->(b:Entity {uuid: $b_uuid})
WHERE r.intent_meta IS NOT NULL
WITH a, b, r
MATCH (pat:CognitivePattern {name: $pattern_name, group_id: $group_id})
MERGE (pat)-[:REINFORCED_BY]->(a)
MERGE (pat)-[:REINFORCED_BY]->(b)
SET r.cognitive_pattern_name = $pattern_name
"""


async def upsert_profile_to_graph(
    graphiti,
    profile: dict,
    edges_data: list[dict],
    group_id: str,
) -> dict:
    """Persist the profile + patterns + REINFORCED_BY links into Neo4j.
    Returns counts of what was upserted."""
    now = time.time()
    counts = {"profile": 0, "patterns": 0, "links": 0}

    async with graphiti.driver.session() as session:
        # Upsert :DirectorProfile node
        await session.run(
            UPSERT_PROFILE_QUERY,
            group_id=group_id,
            summary=profile.get("working_style_summary", ""),
            visions=profile.get("recurring_visions", []),
            now=now,
        )
        counts["profile"] = 1

        # Count occurrences per pattern from the edges
        pattern_counts: dict[str, int] = {}
        for e in edges_data:
            p = (e.get("cognitive_pattern") or "").strip()
            if p:
                pattern_counts[p] = pattern_counts.get(p, 0) + 1

        # Upsert each :CognitivePattern node + link to profile
        for name, cnt in pattern_counts.items():
            await session.run(
                UPSERT_PATTERN_QUERY,
                name=name,
                group_id=group_id,
                count=cnt,
                now=now,
            )
            counts["patterns"] += 1

        # Link each pattern to the entities it reinforces
        for e in edges_data:
            p = (e.get("cognitive_pattern") or "").strip()
            if not p:
                continue
            await session.run(
                LINK_PATTERN_TO_EDGES_QUERY,
                a_uuid=e["a_uuid"],
                b_uuid=e["b_uuid"],
                pattern_name=p,
                group_id=group_id,
            )
            counts["links"] += 1

    return counts


# ---------------------------------------------------------------------------
# Main: build profile, persist both JSON and graph form
# ---------------------------------------------------------------------------

async def main() -> None:
    settings = Settings.from_env()
    graphiti = build_graphiti(settings)

    async with graphiti.driver.session() as session:
        rows = [r.data() async for r in await session.run(COLLECT_QUERY)]

    if not rows:
        print("No annotated edges. Run src.annotate first.")
        await graphiti.close()
        return

    edges_data: list[dict] = []
    group_ids: dict[str, int] = {}
    for row in rows:
        try:
            meta = json.loads(row["meta"]) if isinstance(row["meta"], str) else row["meta"]
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(meta, dict):
            continue
        edges_data.append({
            "a": row["a"],
            "a_uuid": row["a_uuid"],
            "b": row["b"],
            "b_uuid": row["b_uuid"],
            "fact": row["fact"],
            "group_id": row["group_id"],
            **meta,
        })
        gid = row["group_id"]
        group_ids[gid] = group_ids.get(gid, 0) + 1

    # Pick the dominant group_id for the global profile
    primary_group = max(group_ids.items(), key=lambda kv: kv[1])[0] if group_ids else "default"

    print(f"Compressing {len(edges_data)} annotated edges (group={primary_group})...")

    client = AsyncOpenAI(api_key=settings.api_key, base_url=settings.base_url)
    await _rate_acquire()
    resp = await client.chat.completions.create(
        model=settings.annotator_llm_model,
        messages=[{"role": "user", "content": COMPRESSION_PROMPT.format(
            edges=json.dumps([
                {k: v for k, v in e.items() if k not in ("a_uuid", "b_uuid", "group_id")}
                for e in edges_data
            ], indent=2),
        )}],
        temperature=0.5,
        max_tokens=8192,
    )
    msg = resp.choices[0].message
    text = ((msg.content or "") or (getattr(msg, "reasoning_content", None) or "")).strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    profile = json.loads(text[start : end + 1])

    # 1. Write JSON for backward compat
    OUTPUT_PATH.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    print(f"  Wrote {OUTPUT_PATH}")

    # 2. Persist as graph nodes
    counts = await upsert_profile_to_graph(graphiti, profile, edges_data, primary_group)
    print(f"  Upserted to graph: {counts['profile']} profile, {counts['patterns']} patterns, {counts['links']} pattern→entity links")

    await graphiti.close()
    print("\nProfile JSON:")
    print(json.dumps(profile, indent=2))


if __name__ == "__main__":
    asyncio.run(main())

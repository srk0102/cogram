"""Pattern-name deduplication.

The annotator (gpt-4o-mini) emits slight variations of the same cognitive
pattern across edges — "integration-focused development",
"integration-focused innovation", "modular integration approach",
"systematic integration" — when these are really one underlying idea.

Without dedup, the cognitive_pattern list bloats from O(real patterns) to
O(annotation calls). This module collapses near-duplicates by embedding the
names and merging clusters where cosine similarity > MERGE_THRESHOLD.

Strategy:
  1. Embed every CognitivePattern.name in a group via the graphiti embedder.
  2. Greedy clustering: for each pattern, find the highest-confidence
     pattern within MERGE_THRESHOLD cosine; if found, merge into it.
  3. Merge:
     - confidence_winner.confidence += loser.confidence
     - confidence_winner.count       += loser.count
     - rewire HAS_PATTERN edges from loser → winner
     - rewire REINFORCED_BY edges from loser → winner
     - rewire RELATES_TO.cognitive_pattern_name from loser.name → winner.name
     - delete loser CognitivePattern node

Public entrypoints:
  await dedup_patterns(graphiti, group_id, settings, threshold=0.88) -> dict
  python -m src.pattern_dedup --group <group_id>   (CLI)
"""
from __future__ import annotations

import argparse
import asyncio
import math
from typing import Any

from cogram.core.config import Settings, build_graphiti


MERGE_THRESHOLD = 0.88  # cosine similarity above which two pattern names are merged


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


LIST_PATTERNS_QUERY = """
MATCH (pat:CognitivePattern {group_id: $g})
RETURN pat.name AS name,
       coalesce(pat.confidence, 0) AS confidence,
       coalesce(pat.count, 0) AS count
ORDER BY confidence DESC, count DESC
"""

MERGE_PATTERN_QUERY = """
// Find the two pattern nodes
MATCH (winner:CognitivePattern {name: $winner_name, group_id: $g})
MATCH (loser:CognitivePattern  {name: $loser_name,  group_id: $g})

// Bump winner stats with loser's contributions
SET winner.confidence = coalesce(winner.confidence, 0) + coalesce(loser.confidence, 0),
    winner.count      = coalesce(winner.count, 0)      + coalesce(loser.count, 0),
    winner.aliases    = coalesce(winner.aliases, []) + [$loser_name]

// Move HAS_PATTERN edges from loser → winner (idempotent via MERGE)
WITH winner, loser
OPTIONAL MATCH (p:DirectorProfile)-[hp:HAS_PATTERN]->(loser)
FOREACH (_ IN CASE WHEN hp IS NULL THEN [] ELSE [1] END |
  MERGE (p)-[:HAS_PATTERN]->(winner)
  DELETE hp
)

// Move REINFORCED_BY edges from loser → winner
WITH winner, loser
OPTIONAL MATCH (loser)-[rb:REINFORCED_BY]->(e:Entity)
FOREACH (_ IN CASE WHEN rb IS NULL THEN [] ELSE [1] END |
  MERGE (winner)-[:REINFORCED_BY]->(e)
  DELETE rb
)

// Update RELATES_TO.cognitive_pattern_name pointers from loser.name → winner.name
WITH winner, loser
OPTIONAL MATCH ()-[r:RELATES_TO]->()
WHERE r.cognitive_pattern_name = $loser_name
SET r.cognitive_pattern_name = $winner_name

// Delete the loser node
WITH winner, loser
DETACH DELETE loser

RETURN winner.name AS winner, winner.confidence AS new_confidence, winner.count AS new_count
"""


async def dedup_patterns(
    graphiti,
    group_id: str,
    settings: Settings,
    threshold: float = MERGE_THRESHOLD,
) -> dict:
    """Cluster near-duplicate cognitive patterns and merge them. Returns summary dict.

    Idempotent: running on an already-deduped group is a no-op.
    """
    async with graphiti.driver.session() as session:
        rows = [r.data() async for r in await session.run(LIST_PATTERNS_QUERY, g=group_id)]

    if len(rows) <= 1:
        return {"group_id": group_id, "before": len(rows), "merged": 0, "after": len(rows), "merges": []}

    # Embed every pattern name
    names = [r["name"] for r in rows]
    embeddings = await graphiti.embedder.create_batch(names)

    # Greedy clustering: high-confidence patterns absorb similar low-confidence ones
    # Iterate over candidates ranked by confidence DESC. For each candidate, scan
    # already-claimed winners; if cosine > threshold, this candidate is a loser.
    winners: list[int] = []   # indexes of winners (will keep their names)
    merges: list[dict] = []   # list of {winner, loser, sim}
    claimed: set[int] = set()  # losers already merged

    for i, _row in enumerate(rows):
        if i in claimed:
            continue
        merged_into_existing = False
        for w in winners:
            sim = _cosine(embeddings[i], embeddings[w])
            if sim >= threshold:
                merges.append({
                    "winner": rows[w]["name"],
                    "loser": rows[i]["name"],
                    "sim": round(sim, 3),
                })
                claimed.add(i)
                merged_into_existing = True
                break
        if not merged_into_existing:
            winners.append(i)

    # Apply the merges sequentially (each merge is its own Cypher transaction)
    applied: list[dict] = []
    async with graphiti.driver.session() as session:
        for m in merges:
            try:
                rec = await (await session.run(
                    MERGE_PATTERN_QUERY,
                    winner_name=m["winner"],
                    loser_name=m["loser"],
                    g=group_id,
                )).single()
                if rec:
                    applied.append({
                        **m,
                        "new_confidence": rec["new_confidence"],
                        "new_count": rec["new_count"],
                    })
            except Exception as exc:
                applied.append({**m, "error": str(exc)})

    # Invalidate the cached profile JSON so callers see deduped state
    try:
        import os
        import redis.asyncio as _r
        client = _r.from_url(os.environ.get("REDIS_URL", "redis://redis:6379/0"), decode_responses=True)
        async for key in client.scan_iter(match=f"cogram:cache:profile:{group_id}:*"):
            await client.delete(key)
        await client.aclose()
    except Exception:
        pass

    # Emit profile_change event for downstream consumers
    try:
        from cogram.utils import events as _ev
        await _ev.publish(_ev.PROFILE_CHANGE, {
            "action": "pattern_dedup",
            "group_id": group_id,
            "merged": len(applied),
            "before": len(rows),
            "after": len(rows) - len(applied),
        })
    except Exception:
        pass

    return {
        "group_id": group_id,
        "threshold": threshold,
        "before": len(rows),
        "merged": len(applied),
        "after": len(rows) - len(applied),
        "merges": applied,
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description="Dedup cognitive patterns by embedding similarity.")
    parser.add_argument("--group", default="default", help="group_id to dedup (default: 'default')")
    parser.add_argument("--threshold", type=float, default=MERGE_THRESHOLD)
    args = parser.parse_args()

    settings = Settings.from_env()
    g = build_graphiti(settings)
    result = await dedup_patterns(g, args.group, settings, threshold=args.threshold)
    import json as _json
    print(_json.dumps(result, indent=2))
    await g.close()


if __name__ == "__main__":
    asyncio.run(main())

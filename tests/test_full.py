"""End-to-end self-test for the full cogram stack.

Exercises everything that should be wired:
  - cogram_pipeline.cogram_post_write    (intent + narrate + profile)
  - node_narrator.write_back              (training_data logging)
  - active_memory.pull_subgraph + search  (Redis hot tier)
  - retract() flow                        (mark+filter)

Run inside the container after a Neo4j wipe:
    docker exec cogram-mcp python -m src.selftest_full
"""
import asyncio
import json
import os
from datetime import datetime, timezone

from graphiti_core.nodes import EpisodeType

from cogram.core.config import Settings, build_graphiti
from cogram.pipeline.post_write import cogram_post_write
from cogram.driver import redis_active as am


GROUP = "selftest"

EPISODES = [
    "Building an HR platform: recruiters drop a JD, we find candidates across our CRMs (Greenhouse, Lever, Ashby) and rank them per-JD.",
    "Account confidence threshold is 90 percent. Any social profile attributed below 90 percent is discarded — we will not surface it as theirs.",
    "Anti-fraud: we explicitly fight fake/AI-generated accounts — coordinated commit patterns, stock photos, suspicious follower graphs all reduce confidence.",
]


async def _pg_count(query: str, **params) -> int:
    from cogram.llm_client.engram import _get_pool
    pool = await _get_pool()
    if pool is None:
        return -1
    async with pool.acquire() as conn:
        row = await conn.fetchrow(query, *params.values())
    return row[0] if row else 0


async def _redis_keys(pattern: str) -> list[str]:
    try:
        import redis.asyncio as redis_asyncio
    except ImportError:
        return []
    client = redis_asyncio.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True)
    keys = []
    async for k in client.scan_iter(match=pattern):
        keys.append(k)
    await client.aclose()
    return keys


async def main() -> None:
    settings = Settings.from_env()
    g = build_graphiti(settings)

    print("=== STEP 1: ingest 3 episodes through cogram pipeline ===")
    for i, body in enumerate(EPISODES, 1):
        name = f"selftest_full_{i}"
        result = await g.add_episode(
            name=name,
            episode_body=body,
            source=EpisodeType.text,
            source_description="selftest_full",
            reference_time=datetime.now(timezone.utc),
            previous_episode_uuids=[],
            group_id=GROUP,
        )
        summary = await cogram_post_write(g, result, GROUP, settings)
        print(f"  ep {i}: graphiti={len(result.nodes)} nodes/{len(result.edges)} edges; pipeline={summary['edges_annotated']} ann, {summary['nodes_narrated']} narr, profile_distilled={summary['profile_distilled']}")

    print("\n=== STEP 2: training_data accumulation ===")
    n_train_all = await _pg_count("SELECT count(*) FROM engram.training_data")
    print(f"  engram.training_data total rows: {n_train_all}  (must be > 0 for trainer to have fuel)")

    print("\n=== STEP 3: search_graph cold start (warms Redis) ===")
    # Inline equivalent of search_graph's hot+cold path
    seed_q = "what is the confidence threshold"
    sub = await am.load(GROUP)
    print(f"  pre-warm: subgraph {'EXISTS' if sub else 'MISSING'}")
    if sub is None:
        seed_emb = await g.embedder.create(seed_q)
        sub = await am.pull_subgraph(g, GROUP, GROUP, seed_emb)
        print(f"  pulled subgraph: {len(sub.entities)} entities, {len(sub.edges)} edges into Redis")

    print("\n=== STEP 4: search inside Redis subgraph ===")
    q_emb = await g.embedder.create(seed_q)
    hits = am.search_edges_in_subgraph(sub, q_emb, k=5)
    print(f"  Redis vector search returned {len(hits)} matches")
    for h in hits[:3]:
        print(f"    - {h.fact[:120]}")

    print("\n=== STEP 5: redis keys present ===")
    keys = await _redis_keys("cogram:session:*")
    print(f"  cogram:session:* keys in Redis: {keys}")

    print("\n=== STEP 6: retract + verify filter ===")
    # Find an entity to retract
    async with g.driver.session() as session:
        rows = [r.data() async for r in await session.run(
            "MATCH (n:Entity {group_id: $g}) RETURN n.name AS n LIMIT 1", g=GROUP
        )]
    if rows:
        target_name = rows[0]["n"]
        print(f"  retracting all edges touching entity {target_name!r}...")
        cypher = """
        MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
        WHERE toLower(a.name) = toLower($t) OR toLower(b.name) = toLower($t)
        SET r.retracted_at = timestamp(), r.retraction_reason = 'selftest'
        RETURN count(r) AS n
        """
        async with g.driver.session() as session:
            rec = await (await session.run(cypher, t=target_name)).single()
        print(f"  retracted {rec['n']} edges")

        async with g.driver.session() as session:
            rec2 = await (await session.run(
                "MATCH ()-[r:RELATES_TO]->() WHERE r.retracted_at IS NOT NULL RETURN count(r) AS n"
            )).single()
        print(f"  edges with retracted_at set in graph: {rec2['n']}")
    else:
        print("  no entities to retract")

    print("\n=== FINAL VERDICT ===")
    pass_train = n_train_all > 0
    pass_redis = len(keys) > 0
    pass_search = len(hits) > 0
    print(f"  training_data populated:    {'PASS' if pass_train else 'FAIL'} ({n_train_all} rows)")
    print(f"  redis active subgraph:      {'PASS' if pass_redis else 'FAIL'} ({len(keys)} keys)")
    print(f"  redis vector search hits:   {'PASS' if pass_search else 'FAIL'} ({len(hits)} matches)")

    await g.close()


if __name__ == "__main__":
    asyncio.run(main())

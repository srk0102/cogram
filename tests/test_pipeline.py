"""Self-test for the cogram pipeline.

Runs the full add_episode + cogram_post_write chain twice with fixed content,
then prints the resulting Neo4j state. Intended to be invoked inside the
cogram-mcp container after a Neo4j wipe:

    docker exec cogram-mcp python -m src.selftest_pipeline

Pass criteria:
  - intent_meta count > 0
  - vllm_narrative count > 0
  - :DirectorProfile count == 1
  - :CognitivePattern count > 0
"""
import asyncio
import json
from datetime import datetime, timezone

from graphiti_core.nodes import EpisodeType

from cogram.core.config import Settings, build_graphiti
from cogram.pipeline.post_write import cogram_post_write


GROUP = "selftest"

EPISODES = [
    "Building an HR platform: recruiters drop a JD, we find candidates across our CRMs (Greenhouse, Lever, Ashby) and rank them per-JD.",
    "Account confidence threshold is 90 percent. Any social profile attributed below 90 percent is discarded — we will not surface it as theirs.",
]


async def main() -> None:
    settings = Settings.from_env()
    g = build_graphiti(settings)

    for i, body in enumerate(EPISODES, 1):
        name = f"selftest_{i}"
        print(f"\n[{i}/{len(EPISODES)}] add_episode: {name}")
        result = await g.add_episode(
            name=name,
            episode_body=body,
            source=EpisodeType.text,
            source_description="selftest",
            reference_time=datetime.now(timezone.utc),
            previous_episode_uuids=[],
            group_id=GROUP,
        )
        print(f"  graphiti extracted: {len(result.nodes)} nodes, {len(result.edges)} edges")

        summary = await cogram_post_write(g, result, GROUP, settings)
        print(f"  cogram pipeline:   {json.dumps(summary, indent=2)}")

    # Final state report
    cypher_checks = [
        ("Episodic count", "MATCH (e:Episodic) RETURN count(e) AS n"),
        ("Entity count", "MATCH (n:Entity) RETURN count(n) AS n"),
        ("RELATES_TO edges", "MATCH ()-[r:RELATES_TO]->() RETURN count(r) AS n"),
        ("intent_meta on edges", "MATCH ()-[r:RELATES_TO]->() WHERE r.intent_meta IS NOT NULL RETURN count(r) AS n"),
        ("vllm_narrative on nodes", "MATCH (n:Entity) WHERE n.vllm_narrative IS NOT NULL RETURN count(n) AS n"),
        (":DirectorProfile", "MATCH (p:DirectorProfile) RETURN count(p) AS n"),
        (":CognitivePattern", "MATCH (c:CognitivePattern) RETURN count(c) AS n"),
        ("REINFORCED_BY edges", "MATCH ()-[r:REINFORCED_BY]->() RETURN count(r) AS n"),
        ("HAS_PATTERN edges", "MATCH ()-[r:HAS_PATTERN]->() RETURN count(r) AS n"),
    ]
    print("\n=== final graph state ===")
    async with g.driver.session() as session:
        for label, q in cypher_checks:
            rec = await (await session.run(q)).single()
            print(f"  {label:32s} {rec['n']}")

    await g.close()


if __name__ == "__main__":
    asyncio.run(main())

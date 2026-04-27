"""Cogram quickstart — standalone Python (no MCP needed).

Prereqs:
  docker compose up -d        # spins Neo4j + Postgres + Redis
  pip install -e .            # installs the cogram package

Then:
  export OPENAI_API_KEY=sk-...
  python examples/quickstart.py

What this demonstrates:
  1. Initialize cogram (uses local Neo4j by default)
  2. Add three episodes describing a developer's preferences
  3. Search the graph for what the developer values
  4. Pull the DirectorProfile (after the cogram pipeline distills it)
"""
import asyncio
import os
from datetime import datetime, timezone

from cogram import Cogram


EPISODES = [
    "I prefer Postgres over MongoDB for the user service — strong consistency matters more than flexibility here.",
    "We chose async over sync for backend services because asyncio's maturity makes it easier to scale to thousands of concurrent connections.",
    "Tests are non-negotiable for backend code. I get frustrated when PRs land without tests.",
]


async def main() -> None:
    # Enable the cogram pipeline (intent annotation, narration, profile distill, knot synthesis)
    os.environ["COGRAM_FULL_PIPELINE"] = "true"

    cogram = Cogram(
        uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        user=os.environ.get("NEO4J_USER", "neo4j"),
        password=os.environ.get("NEO4J_PASSWORD", "password"),
    )

    print("=== Adding 3 episodes ===")
    for i, body in enumerate(EPISODES, 1):
        result = await cogram.add_episode(
            name=f"quickstart_ep_{i}",
            episode_body=body,
            source_description="quickstart",
            reference_time=datetime.now(timezone.utc),
            group_id="quickstart",
        )
        print(f"  ep {i}: extracted {len(result.nodes)} entities, {len(result.edges)} edges")

    # Pipeline runs in background; give it a few seconds to populate intent_meta
    print("\n=== Waiting 30s for cogram pipeline to populate intent + narratives ===")
    await asyncio.sleep(30)

    print("\n=== Searching the graph ===")
    results = await cogram.search("what does the developer value in databases")
    for edge in results[:5]:
        print(f"  - {edge.fact}")
        intent = (edge.attributes or {}).get("intent_meta")
        if intent:
            print(f"      WHY: {intent.get('why_connected', '')[:120]}")
            print(f"      VISION: {intent.get('director_vision', '')[:120]}")

    await cogram.close()


if __name__ == "__main__":
    asyncio.run(main())

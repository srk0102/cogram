# Cogram

**Intent-aware memory for LLM agents.** A fork of [Graphiti](https://github.com/getzep/graphiti) with the intent layer, Engram cache, Redis active subgraph, and MCP server baked in.

```
mem0       → flat facts in a vector DB
graphiti   → temporal knowledge graph
cogram     → graphiti + WHY each connection exists + DirectorProfile + cached
             synthesis, ready for LLM agents to consume directly
```

> *"Memory with intent. Cached forever after first call."*

---

## What's different about cogram

Most memory products store **what** the user said. Cogram stores **why they said it**, **what goal it serves**, and **how they think** — pre-synthesized for direct LLM consumption.

| | mem0 | graphiti (alone) | cogram |
|---|---|---|---|
| Stores facts | ✅ | ✅ | ✅ |
| Temporal model | ❌ | ✅ | ✅ (inherited) |
| Per-edge **why_connected** + **director_vision** | ❌ | ❌ | ✅ |
| Per-entity narrative + stance + open questions | ❌ | ❌ | ✅ |
| DirectorProfile + cognitive patterns | ❌ | ❌ | ✅ |
| Pre-synthesized knot narratives (LLM-ready paragraphs) | ❌ | ❌ | ✅ |
| Engram cache (cost approaches zero on warm reads) | ❌ | ❌ | ✅ |
| MCP server (Claude/Cursor/Windsurf/etc.) | partial | ✅ | ✅ |

The intent layer is the moat. Every edge stores not just the fact ("Director chose Postgres over Mongo") but the **reasoning** ("strong consistency matters more than flexibility"), the **goal** ("reliable backend storage"), and the **thinking style** ("data-driven validation"). When an LLM agent queries cogram, it gets compressed, intent-aware context — not raw fragments.

---

## Install

```bash
git clone https://github.com/srk0102/cogram.git
cd cogram
cp .env.example .env       # paste OPENAI_API_KEY
docker compose up -d
```

That's it. Six containers come up: Neo4j (cold tier), Postgres (Engram warm cache), Redis (hot subgraph + events), cogram-mcp (port 7800 — MCP server), cogram-dashboard (port 7801), cogram-trainer (dormant until enough samples accumulate).

Connect from Claude Desktop by editing `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "cogram": {
      "command": "npx",
      "args": ["-y", "mcp-remote", "http://localhost:7800/mcp/"]
    }
  }
}
```

Restart Claude Desktop. Cogram tools appear.

---

## Quickstart (5 lines, no MCP)

```python
import asyncio
from cogram import Graphiti

async def demo():
    g = Graphiti(uri="bolt://localhost:7687", user="neo4j", password="password")
    await g.add_episode(
        name="ep1",
        episode_body="Director chose Postgres over Mongo for the user service — strong consistency matters more than flexibility here.",
        source_description="demo",
    )
    print(await g.search("what does the director value in databases"))

asyncio.run(demo())
```

When `COGRAM_FULL_PIPELINE=true` is set, the cogram pipeline fires after every `add_episode`:
- intent_meta annotated on every new edge (3 LLM calls per ~5 new edges)
- vllm_narrative synthesized on every new entity (capped at 5/episode)
- DirectorProfile re-distilled every 5 episodes
- Knot narratives synthesized for hub nodes (Gemma 3 4B local + gpt-4o-mini fallback)
- Redis active subgraph invalidated for the group

After cache warmup, repeat queries cost effectively zero.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       LLM Agent                             │
│              (Claude / GPT / local / etc.)                  │
└────────────────────────────┬────────────────────────────────┘
                             │ MCP (stdio + HTTP/SSE)
                ┌────────────▼────────────┐
                │      cogram MCP         │
                │   (12 tools exposed)    │
                └────────────┬────────────┘
                             │
                ┌────────────▼────────────────────────────────┐
                │  Read path:                                 │
                │    1. Engram cache hit?     (Postgres ~1ms) │
                │    2. Redis active subgraph (<1ms)          │
                │    3. Neo4j Cypher          (~50ms)         │
                │    4. On miss, Gemma 3 4B synthesizes       │
                │       (or gpt-4o-mini fallback)             │
                └────────────┬────────────────────────────────┘
                             │
                ┌────────────▼────────────────────────────────┐
                │  Write path (per add_episode):              │
                │    1. graphiti extracts entities + edges    │
                │    2. intent annotator → why on each edge   │
                │    3. node narrator → vllm_narrative        │
                │    4. profile distill (every 5 episodes)    │
                │    5. knot detection + Gemma synthesis      │
                │    6. Redis cache invalidation + events     │
                └─────────────────────────────────────────────┘
```

Three storage tiers (substrate-agnostic):

- **Cold (Neo4j by default)** — graph source of truth. Pluggable: also supports FalkorDB, Kuzu, Neptune via graphiti's driver abstraction.
- **Warm (Postgres / Engram)** — LLM decision cache. Every LLM call is hashed and stored; cache hits return zero-cost.
- **Hot (Redis)** — active subgraph cache + event pub/sub for live dashboard updates.

Five LLM call types:

1. **Extraction** (graphiti's, gpt-4o-mini) — entities + edges from raw text
2. **Annotation** (gpt-4o-mini) — intent_meta on each new edge
3. **Narration** (gpt-4o-mini) — vllm_narrative on hub entities
4. **Distillation** (gpt-4o-mini, every 5 episodes) — DirectorProfile + cognitive patterns
5. **Knot synthesis** (Gemma 3 4B local, gpt-4o-mini fallback) — pre-compressed paragraph for LLM consumption

---

## MCP tools exposed

| Tool | Use |
|---|---|
| `add_episode(content, group_id)` | Write an episode; pipeline fires async |
| `record_fact(subject, predicate, object)` | Convenience wrapper for atomic SPO facts |
| `search_graph(query, group_id)` | Vector + profile-aware retrieval; Redis-cached |
| `find_connections(entity_name)` | All edges touching an entity |
| `recent_episodes(entity_name, n)` | Recent episodes mentioning an entity |
| `get_episode(uuid)` | Full content of one episode |
| `get_director_profile(group_id)` | DirectorProfile + top cognitive patterns + per-pattern WHY examples |
| `get_unified_profile()` | Cross-group merged profile (single Cypher) |
| `get_knot(entity_name, format)` | Pre-synthesized knot narrative + raw subgraph for hub entities |
| `list_cognitive_patterns(group_id)` | Distinct cognitive patterns + their edge counts |
| `confidence(entity_name)` | Decayed effective confidence + label |
| `retract(target, reason)` | Mark fact as wrong; cascades through profile + caches |
| `dedup_patterns(group_id)` | Embed + merge near-duplicate cognitive_pattern names |

---

## Cost characteristics

| Pattern | Per write | Per read | Notes |
|---|---|---|---|
| First episode in group (cold) | ~10–25s, ~$0.005 | — | Pipeline fires fully |
| Episode N+1 with cache warmup | ~3s, ~$0.001 | — | Annotation/narration cache hits |
| Repeat search on same group | — | <1ms (Redis) | Active subgraph hit |
| `get_director_profile` after first call | — | <5ms | Redis JSON cache, single Cypher round-trip |
| Knot resynthesis (delta-gated) | ~3s, ~free with local Gemma | — | Rate-capped at 5/hr/group |

Five cost-bound knot detection parameters (env-tunable):
- `COGRAM_HARD_DEGREE_FLOOR=5`
- `COGRAM_MIN_KNOT_SCORE=6.0`
- `COGRAM_MAX_KNOTS_PER_GROUP=25`
- `COGRAM_RESYNTHESIS_DELTA=3.0`
- `COGRAM_RESYNTHESIS_RATE_CAP_PER_HOUR=5`

These make worst-case daily cost provably bounded — knots can't proliferate, re-synthesis can't thrash.

---

## License

Apache 2.0. See [LICENSE](LICENSE).

This is a fork of [Graphiti](https://github.com/getzep/graphiti) by Zep AI Research, Inc. (also Apache 2.0). Per Apache §4(d), any redistribution must preserve the [NOTICE](NOTICE) file. See [ATTRIBUTION.md](ATTRIBUTION.md) for the human-readable rules — short version: keep the NOTICE, add one credit line to your README, you're done.

---

## Status

Beta. The architecture is real and tested end-to-end:

- ✅ All 12 MCP tools work
- ✅ Pipeline fires with measured perf: graphiti extract = 11s, full cogram chain = 30s sync / ~3s async
- ✅ Engram cache + Redis active subgraph wired
- ✅ Knot detection + Gemma synthesis with fallback
- ✅ Self-test verified (`docker exec cogram-mcp python -m tests.test_v2`)

Still rough:
- Examples directory needs expansion
- Documentation site (just markdown for now)
- Pip-publishable wheel (works as `pip install -e .` today)
- Hosted SaaS (later — currently self-host only)

Built solo. Issues/PRs welcome at [github.com/srk0102/cogram](https://github.com/srk0102/cogram).

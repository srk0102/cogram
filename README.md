# Cogram

**Intent-aware memory for LLM agents.** A fork of [Graphiti](https://github.com/getzep/graphiti) that captures *why* every connection exists, not just *that* it exists.

```
mem0       →  flat facts in a vector DB
graphiti   →  temporal knowledge graph
cogram     →  graphiti + WHY each fact exists + a model of HOW the user thinks
              + cached forever (cost approaches zero on warm reads)
```

---

## Why

Most LLM memory products store **what** the user said. When a different agent (Claude in your terminal vs Claude in your browser vs a custom GPT) reads the same memory, each one invents its own reasoning around bare facts. The agents drift apart.

Cogram stores the **why** alongside the what. Every fact carries the user's reasoning, the larger goal it serves, and the thinking pattern it reveals. Any agent reading cogram converges on the same interpretation — they're forced into the same lane because they all see the same `why_connected` and `director_vision`.

That's the moat: **canonical multi-surface context**, not just memory.

---

## Concrete example: same scenario, graphiti vs cogram

A user tells Claude: *"I rejected server-side LinkedIn scraping because of legal issues. We use a Chrome extension during the end-user's logged-in session instead."*

**Graphiti alone stores:**
```
(User) -[REJECTED]-> (server-side LinkedIn scraping)
       fact: "User rejected server-side LinkedIn scraping"
```
A future agent reading this thinks: *"Maybe the user will accept it now if I phrase it differently."* → wrong route.

**Cogram stores the same edge with intent:**
```json
{
  "fact": "User rejected server-side LinkedIn scraping",
  "intent_meta": {
    "why_connected": "Server-side scraping conflicts with LinkedIn ToS, creating legal risk",
    "director_vision": "Build a legally compliant AI recruitment platform",
    "cognitive_pattern": "legal risk mitigation"
  }
}
```
A future agent in any interface reads this and reasons: *"User's vision is legal compliance. So scraping Indeed via residential proxies would be rejected by the same logic, even though we never specifically discussed Indeed."* → right route, every time, across every surface.

---

## Install — three commands, no clone needed

```bash
mkdir cogram && cd cogram
curl -O https://raw.githubusercontent.com/srk0102/cogram/master/docker-compose.yml
curl -O https://raw.githubusercontent.com/srk0102/cogram/master/.env.example
mv .env.example .env       # edit, paste your OPENAI_API_KEY
docker compose pull && docker compose up -d
```

After ~60-90 seconds, five containers are running:

| Service | Port | Image | Role |
|---|---|---|---|
| `cogram-mcp` | 7800 | `ghcr.io/srk0102/cogram-mcp:latest` | MCP server (stdio + HTTP/SSE) |
| `cogram-dashboard` | 7801 | `ghcr.io/srk0102/cogram-dashboard:latest` | Live force-graph viz |
| `cogram-neo4j` | 7474 / 7687 | `neo4j:5.26` | Graph (cold tier) |
| `cogram-postgres` | 5432 | `postgres:16-alpine` | Engram cache (warm tier) |
| `cogram-redis` | 6379 | `redis:7-alpine` | Active subgraph + events (hot tier) |

Visit http://localhost:7801 to see the graph viz. Cogram is reachable at http://localhost:7800/mcp/.

### Connect Claude Desktop

Edit `%APPDATA%/Claude/claude_desktop_config.json` (Windows) or `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS):

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

Restart Claude Desktop. Cogram tools appear: `add_episode`, `record_fact`, `search_graph`, `get_director_profile`, and 9 more.

---

## What cogram adds on top of graphiti

| | Inherited from graphiti | Added by cogram |
|---|---|---|
| Temporal knowledge graph | ✅ | – |
| Multi-DB drivers (Neo4j, FalkorDB, Kuzu, Neptune) | ✅ | – |
| Hybrid BM25 + vector + graph search | ✅ | – |
| **Per-edge `why_connected` + `director_vision` + `cognitive_pattern`** | ❌ | ✅ |
| **Per-entity `vllm_narrative` (perspective + stance + open questions)** | ❌ | ✅ |
| **DirectorProfile distillation + cognitive pattern aggregation** | ❌ | ✅ |
| **Engram cache (Postgres) — repeat LLM calls cost zero** | ❌ | ✅ |
| **Redis active subgraph (hot tier, <1ms reads)** | ❌ | ✅ |
| **Knot synthesis with local Gemma 3n** (free hub narratives) | ❌ | ✅ |
| **Drift detection + contradiction classifier + confidence decay** | ❌ | ✅ |
| **Knot-aware retraction with cascade** (mark wrong → invalidate caches + decrement profile) | ❌ | ✅ |
| **MCP server (13 tools)** for any LLM agent | partial | ✅ |

The forked graphiti code lives directly inside `cogram/` (no separate `graphiti-core` install). Cogram's additions live in their natural locations: `cogram/utils/maintenance/{intent_annotation,node_narration,profile_distillation,knot_synthesis,drift_detection,pattern_dedup}.py`, `cogram/llm_client/engram.py`, `cogram/driver/redis_active.py`, plus two new top-level subdirs `cogram/pipeline/` and `cogram/server/`.

---

## MCP tools (13)

| Tool | Purpose |
|---|---|
| `add_episode(content, group_id)` | Write a fact; cogram pipeline fires async |
| `record_fact(subject, predicate, object)` | Atomic SPO fact for foundational principles |
| `search_graph(query, group_id)` | Profile-aware retrieval, Redis-cached |
| `find_connections(entity_name)` | All edges touching an entity |
| `recent_episodes(entity_name, n)` | Recent episodes mentioning an entity |
| `get_episode(uuid)` | Full content of one episode |
| `get_director_profile(group_id)` | DirectorProfile + cognitive patterns + per-pattern WHY examples |
| `get_unified_profile()` | Cross-group merged profile |
| `get_knot(entity_name, format)` | Pre-synthesized knot narrative + raw subgraph |
| `list_cognitive_patterns(group_id)` | Distinct cognitive patterns + edge counts |
| `confidence(entity_name)` | Decayed effective confidence + label |
| `retract(target, reason)` | Mark fact wrong; cascades through profile + caches |
| `dedup_patterns(group_id)` | Embed + merge near-duplicate cognitive patterns |

Full architecture in [docs/architecture.md](docs/architecture.md).

---

## Daily commands

```bash
docker compose up -d               # start everything
docker compose down                # stop (volumes preserved)
docker compose down -v             # stop + wipe all data (destructive)
docker compose logs -f cogram-mcp  # tail server logs
docker compose pull                # update to latest images from ghcr.io
```

For development on the source code, `git clone https://github.com/srk0102/cogram.git` and run `docker compose up -d` from inside — the compose file builds locally from your changes instead of pulling from ghcr.io.

---

## License + attribution

Apache 2.0. See [LICENSE](LICENSE).

This is a fork of [Graphiti](https://github.com/getzep/graphiti) by Zep AI Research, Inc. (Apache 2.0). Per Apache §4(d), redistributions must preserve the [NOTICE](NOTICE) file. Forks must additionally credit Cogram (this repo) and Graphiti (the upstream) in their README and any user-facing surface — see [ATTRIBUTION.md](ATTRIBUTION.md) for the plain-language rules.

---

## Status

**Beta.** Verified end-to-end:
- Async pipeline fires on every `add_episode` (~3s MCP latency, ~15s background)
- Engram (Postgres) + Redis active subgraph wired and active
- Knot detection + Gemma local synthesis with gpt-4o-mini fallback
- All 13 MCP tools functional
- Public Docker images on ghcr.io, anonymous pull works

**Known limitations:**
- LLM annotator can confuse "context" with "user intent" on edges that mention competitors. Mitigation: use `record_fact` for foundational principles to lock the SPO structure.
- Trainer container (T2 LoRA per-node adapters) is opt-in via `docker compose --profile training up`, deferred until you have ≥50 samples per node.

**Roadmap (v0.2):**
- `edge_kind` field in intent_meta to distinguish principle / action / context / competitor edges
- Inline cogram value-add deeper into graphiti's hot-path functions
- T2 LoRA training activation
- Hosted SaaS option

Issues / PRs welcome at [github.com/srk0102/cogram](https://github.com/srk0102/cogram).

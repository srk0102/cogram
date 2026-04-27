# Cogram

**Intent-aware memory for LLM agents — a fork of [Graphiti](https://github.com/getzep/graphiti) that captures *why* every connection exists, not just *that* it exists.**

Cogram = forked Graphiti's temporal knowledge graph + an intent layer (every edge knows *why* it exists and *what goal it serves*) + a director-profile distillation (a model of *how the user thinks*) + Engram-style cache (cost approaches zero on warm reads) + MCP server. **One Docker stack, one command, ready for Claude / Cursor / any MCP client.**

> *"Memory with intent. Cached forever after first call."*

---

## The 60-second pitch

Most LLM memory products store **what** the user said. Cogram stores **why they said it**, **what goal it serves**, and **how they think** — pre-synthesized for direct LLM consumption.

The result: when Claude in your browser, Claude in your terminal, Cursor, and a custom GPT-4 agent all hit cogram, **they all reason the same way about your decisions** because they all see the same `why_connected` + `director_vision` for every fact.

That's the moat: **canonical multi-surface context**, not just memory.

---

## A concrete example: same scenario, graphiti vs cogram

You tell Claude: *"I rejected server-side LinkedIn scraping because of legal issues. We use a Chrome extension during the user's logged-in session instead."*

### What stock Graphiti stores

```
(Siva) -[REJECTED]-> (server-side LinkedIn scraping)
       fact: "Siva rejected server-side LinkedIn scraping"
```

That's it. A fact, an edge, no reasoning.

**Future Claude (in some other interface) reading this:**
> "OK, noted. Siva rejected server-side scraping. Maybe they'll accept it if I phrase it differently. Or maybe scraping Indeed is fine because it's not LinkedIn."

→ Wrong route. Same mistake from a different agent.

### What Cogram stores

```
(Siva) -[REJECTED]-> (server-side LinkedIn scraping)
       fact: "Siva rejected server-side LinkedIn scraping"
       intent_meta: {
         "why_connected": "Server-side scraping conflicts with LinkedIn ToS,
                           creating legal risk for the platform",
         "director_vision": "Build a legally compliant AI HR recruitment
                             platform that respects every source's terms",
         "cognitive_pattern": "legal risk mitigation"
       }
```

**Future Claude (in any interface) reading this:**
> "Director's vision: legal compliance. Pattern: legal risk mitigation.
> So when asked about scraping Indeed, the same principle applies — that's
> rejected by the same logic, even though we never specifically discussed it."

→ Right route. Forecloses the wrong implementations across surfaces.

**This is the difference.** Cogram lets a future agent reason from your *underlying principles*, not just retrieve your stored rules. Local Claude, browser Claude, Cursor agents, custom GPT-4 — they all converge on the same answer because they all see the same intent.

---

## What Cogram improves over Graphiti, exactly

Cogram is a fork of Graphiti — it inherits everything Graphiti does (bi-temporal model, Neo4j/FalkorDB/Kuzu drivers, multi-LLM clients, hybrid BM25+vector+graph search). The forked code lives directly inside `cogram/` (no separate `graphiti-core` install).

The cogram-specific additions:

| # | Improvement | Where it lives in the fork | What it does |
|---|---|---|---|
| 1 | **Per-edge intent annotation** | `cogram/utils/maintenance/intent_annotation.py` | After every `add_episode`, an LLM call adds `why_connected` / `director_vision` / `cognitive_pattern` to each new edge. The "WHY" layer. |
| 2 | **Per-entity narration** | `cogram/utils/maintenance/node_narration.py` | Hub entities get a `vllm_narrative` — a paragraph speaking from the entity's perspective with the user's stance + open questions. LLMs drop this directly into context. |
| 3 | **Director profile distillation** | `cogram/utils/maintenance/profile_distillation.py` | Every 5 episodes, all annotated edges are compressed into a `:DirectorProfile` node + a ranked list of `:CognitivePattern` nodes. Profile-aware retrieval traverses this. |
| 4 | **Knot synthesis with local Gemma** | `cogram/utils/maintenance/knot_synthesis.py` | Hub nodes (degree ≥ 5, knot_score ≥ 6.0) get a pre-synthesized narrative paragraph via Gemma 3n e4b running locally in Ollama. Free of OpenAI API cost. |
| 5 | **Engram cache (Postgres)** | `cogram/llm_client/engram.py` | Every LLM call is hashed + cached in `engram.patterns`. Repeat queries cost zero. Cost approaches zero as graphs mature. |
| 6 | **Redis active subgraph (hot tier)** | `cogram/driver/redis_active.py` | Per-group_id subgraph cached in Redis. <1ms in-memory vector search vs ~50ms Neo4j Cypher. |
| 7 | **Drift detection + contradiction classifier** | `cogram/utils/maintenance/drift_detection.py` | Cosine drift on edge/node/profile centroids + gpt-4o-mini contradiction classifier. Triggers re-narration only when meaningful change happens. |
| 8 | **Confidence decay** | `cogram/utils/confidence.py` | 30-day exponential half-life on every node's confidence. Old facts decay; reinforced facts strengthen. |
| 9 | **Knot-aware retraction** | `cogram/server/mcp.py` (`retract` tool) | Mark a fact as wrong; cascades through pattern confidence + Redis cache invalidation + audit log. Required for legally-defensible auditing. |
| 10 | **MCP server (HTTP + stdio)** | `cogram/server/mcp.py` | 13 MCP tools exposing add/search/retract/profile/knot operations. Connects to Claude Desktop, Cursor, any MCP client. |

If you delete any of those, you're left with something between graphiti and cogram. They're additive on top of the fork.

---

## Install

Two paths depending on whether you want to modify cogram or just run it.

### Option A — Just run it (Docker, no git)

If you only want to use cogram and don't need the source code, grab the compose file and `.env.example` directly:

```bash
mkdir cogram && cd cogram
curl -O https://raw.githubusercontent.com/srk0102/cogram/master/docker-compose.yml
curl -O https://raw.githubusercontent.com/srk0102/cogram/master/.env.example
mv .env.example .env             # then edit .env, paste your OPENAI_API_KEY
docker compose up -d
```

Six containers come up; cogram is reachable on `http://localhost:7800/mcp/`. No Python/Node/build tools needed locally — Docker pulls everything.

> Until cogram images are published to Docker Hub (planned, v0.2), `docker compose` will build images locally on first run, which takes ~3-5 minutes. After Docker Hub publish, first-run becomes a quick image pull.

### Option B — Clone the source (for developers / contributors)

```bash
git clone https://github.com/srk0102/cogram.git
cd cogram
cp .env.example .env             # paste OPENAI_API_KEY
docker compose up -d
```

This gives you the full source tree to modify, plus the same six containers running.

### What runs

- `cogram-mcp` (port 7800) — the MCP server (Streamable HTTP + stdio)
- `cogram-dashboard` (port 7801) — live force-graph viz + metrics
- `cogram-neo4j` (ports 7474/7687) — graph (cold tier)
- `cogram-postgres` — Engram cache (warm tier)
- `cogram-redis` — active subgraph + events (hot tier)
- `cogram-trainer` — opt-in via `docker compose --profile training up`, deferred until ≥50 samples per node

### Daily commands

```bash
docker compose up -d               # start everything
docker compose down                # stop (volumes preserved)
docker compose down -v             # stop + wipe all data (destructive!)
docker compose logs -f cogram-mcp  # tail server logs
docker compose ps                  # see what's running
docker compose pull                # pull latest images (after Docker Hub publish)
```

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

Restart Claude Desktop. Cogram tools appear.

For Gemma local synthesis (optional, but free vs OpenAI cost):
```bash
ollama pull gemma3n:e4b   # 7.5 GB, runs on CPU or GPU
```

---

## What you'll see after writing 3 episodes

Open the dashboard at `http://localhost:7801`. Live counters tick up:

```
episodes:        3
entities:       11
edges:           7
intent_meta:    7/7  (100%)
narrated:        4
knots:           1   ← Gemma synthesized "Siva" as a hub
profile:         1   ← DirectorProfile distilled
patterns:        5
cached LLM calls: ~30  ← Engram, repeats now free
```

Your OpenAI cost for 3 episodes: ~$0.01–0.02. Everything cached forward.

---

## MCP tools exposed

| Tool | Use |
|---|---|
| `add_episode(content, group_id)` | Write an episode; pipeline fires async |
| `record_fact(subject, predicate, object)` | Atomic SPO fact for foundational principles |
| `search_graph(query, group_id)` | Profile-aware retrieval; Redis-cached |
| `find_connections(entity_name)` | All edges touching an entity |
| `recent_episodes(entity_name, n)` | Recent episodes mentioning an entity |
| `get_episode(uuid)` | Full content of one episode |
| `get_director_profile(group_id, top_patterns, examples_per_pattern)` | Profile + cognitive patterns + per-pattern WHY examples |
| `get_unified_profile()` | Cross-group merged profile (single Cypher) |
| `get_knot(entity_name, format)` | Pre-synthesized knot narrative (from Gemma) + raw subgraph |
| `list_cognitive_patterns(group_id)` | Distinct cognitive patterns + edge counts |
| `confidence(entity_name)` | Decayed effective confidence + label |
| `retract(target, reason)` | Mark wrong; cascades through profile + caches |
| `dedup_patterns(group_id)` | Embed + merge near-duplicate cognitive patterns |

---

## The honest pitch

Most "AI memory" products are just facts in a vector DB + LLM extraction. Cogram is one architectural step beyond:

- **Substrate**: Graphiti's bi-temporal knowledge graph (we forked it, kept everything, modified it)
- **+ Intent layer**: every edge knows *why* it exists from the user's perspective
- **+ Director model**: a distilled profile of *how the user thinks* across all their decisions
- **+ Free synthesis**: Gemma local for hub narratives at $0 marginal cost
- **+ Cached forever**: Engram pattern means warm reads are free
- **+ Multi-surface canonical**: every LLM agent talking to cogram gets the same `why` + `vision`, so they all reason consistently

**The thing graphiti gets right:** the temporal model + driver abstraction + multi-LLM support is solid. We took it intact.

**The thing cogram adds:** the reasoning layer. *Why* each fact exists. *What goal* it serves. *Pattern* it reveals. So agents downstream don't have to guess your principles — they read them.

---

## Attribution

This is a fork of [Graphiti](https://github.com/getzep/graphiti) by Zep AI Research, Inc. (Apache 2.0). All graphiti code lives in `cogram/` with explicit modifications for the intent layer + cogram subdirs (`pipeline/`, `server/`).

Per Apache §4(d), redistributions must preserve the [NOTICE](NOTICE) file. Forks must additionally credit *"Cogram by Siva Rama Krishna"* and *"Built on Graphiti by Zep AI Research"* in their README and user-facing surfaces. See [ATTRIBUTION.md](ATTRIBUTION.md) for the full plain-language rules.

License: Apache 2.0. See [LICENSE](LICENSE).

---

## Status

**Beta.** Architecture verified end-to-end:
- ✅ Async pipeline fires after every `add_episode` (~3s MCP latency, ~15s background)
- ✅ Engram cache + Redis hot tier wired and active
- ✅ Knot detection + Gemma synthesis working with gpt-4o-mini fallback
- ✅ All 13 MCP tools functional
- ✅ Dashboard with live force-graph viz

**Known weakness — annotator drift:** the LLM annotator can confuse "context" with "user intent" on edges that mention competitors or third parties. Mitigation: use `record_fact` for foundational principles (locks subject/predicate/object structurally) and the annotator's drift on prose edges becomes supporting context, not load-bearing truth.

**Roadmap (v0.2):**
- `edge_kind` field in intent_meta to distinguish principle/action/context/competitor edges
- Inline cogram value-add deeper into graphiti's hot-path functions (currently sibling files in graphiti's natural subdirs)
- T2 LoRA training (cogram-trainer wired but dormant; activates when ≥50 samples per hub node)
- Multi-tenant auth + Stripe billing for SaaS deployment

Built solo. Issues / PRs welcome at [github.com/srk0102/cogram](https://github.com/srk0102/cogram).

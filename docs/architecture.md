# Cogram Architecture

This document explains how cogram is organized and what each piece does.

## The fork relationship

Cogram is a **fork** of [Graphiti](https://github.com/getzep/graphiti) by Zep AI Research, not a wrapper around it. Graphiti's code lives directly inside the `cogram/` package — driver/, embedder/, llm_client/, prompts/, search/, utils/, models/, cross_encoder/, namespaces/, migrations/, telemetry/. When you run cogram, you are running a modified graphiti.

Cogram extends graphiti in three ways:

1. **New value-add files placed alongside graphiti's existing files** in their natural locations — no parallel sibling layer.
2. **A few of graphiti's hot-path modules are modified** — most notably the main `Graphiti.add_episode()` (now `Cogram.add_episode()`) which fires the cogram post-write pipeline as a background task.
3. **Two new top-level subdirs** — `pipeline/` (orchestration) and `server/` (MCP) — for genuinely new concerns that have no graphiti counterpart.

## Package layout

```
cogram/
├── cogram.py                        # main `Cogram` class (Graphiti alias preserved)
├── nodes.py, edges.py               # Pydantic models for graph elements
├── clients.py                       # `CogramClients` bundle (was GraphitiClients)
├── helpers.py, errors.py, decorators.py, tracer.py, graph_queries.py
├── config.py                        # Cogram Settings + build_graphiti factory
├── budget.py, rate_limit.py         # Cross-cutting cost controls
├── sanitize.py, import_transcripts.py  # Episode preprocessing
│
├── driver/                          # Graph DB drivers (Neo4j, FalkorDB, Kuzu, Neptune)
│   └── redis_active.py              # Cogram: Redis hot-tier subgraph cache
│
├── embedder/                        # Embedding clients (OpenAI, Gemini, Voyage, Azure)
│
├── llm_client/                      # LLM clients with structured-output support
│   ├── client.py                    # Modified: wraps `_generate_response_with_retry` with Engram cache
│   └── engram.py                    # Cogram: Postgres-backed decision cache (the "Engram" pattern)
│
├── prompts/                         # Prompt templates + Pydantic response models
│
├── search/                          # Hybrid search (BM25 + semantic + graph traversal)
│
├── utils/
│   ├── confidence.py                # Cogram: 30-day half-life decay math
│   ├── events.py                    # Cogram: Redis pub/sub for live dashboard
│   └── maintenance/                 # Per-write graph operations
│       ├── edge_operations.py       # Graphiti's
│       ├── node_operations.py       # Graphiti's
│       ├── community_operations.py  # Graphiti's
│       ├── intent_annotation.py     # Cogram: per-edge `why_connected` / `director_vision` / `cognitive_pattern`
│       ├── node_narration.py        # Cogram: per-entity `vllm_narrative`
│       ├── profile_distillation.py  # Cogram: DirectorProfile + CognitivePattern nodes
│       ├── pattern_dedup.py         # Cogram: embedding-based pattern merging
│       ├── drift_detection.py       # Cogram: cosine drift + contradiction classifier
│       └── knot_synthesis.py        # Cogram: hub detection + Gemma 3 4B narrative synthesis
│
├── models/                          # Graphiti's DB query builders
├── cross_encoder/                   # Reranker (e.g., OpenAIRerankerClient)
├── namespaces/                      # NodeNamespace, EdgeNamespace
├── migrations/                      # Schema migrations
├── telemetry/                       # OpenTelemetry tracing (PostHog disabled by default in cogram)
│
├── pipeline/                        # NEW: cogram orchestration
│   ├── post_write.py                # `cogram_post_write()` — runs after every add_episode
│   ├── ingest.py                    # batch CLI ingestion
│   ├── watch.py                     # JSONL watcher daemon
│   └── runner.py                    # full-pipeline CLI entry
│
└── server/                          # NEW: cogram serves over MCP
    └── mcp.py                       # FastMCP HTTP/SSE + 12 MCP tools
```

## The three storage tiers

| Tier | Where | Latency | Job |
|---|---|---|---|
| **Cold** | Neo4j (or FalkorDB / Kuzu / Neptune) | ~50ms | Source of truth. Persistent graph. |
| **Warm** | Postgres (`engram.patterns`, `engram.audit`) | ~1ms | LLM decision cache + audit log + training data |
| **Hot** | Redis | <1ms | Active subgraph cache + event pub/sub |

The Engram cache (in `llm_client/engram.py` and via `Policy` wrapping) makes repeat LLM calls free after first computation. This is the cost-sustainability story.

## The five LLM call types

Every LLM call cogram makes falls into one of these categories. All are cached via Engram.

| # | Call | Module | Default Model | When fired |
|---|---|---|---|---|
| 1 | **Extraction** | `utils/maintenance/edge_operations.py` (graphiti's) | gpt-4o-mini | every `add_episode` |
| 2 | **Annotation** | `utils/maintenance/intent_annotation.py` | gpt-4o-mini | per new edge from add_episode |
| 3 | **Narration** | `utils/maintenance/node_narration.py` | gpt-4o-mini | per qualifying entity (cap 5/episode) |
| 4 | **Distillation** | `utils/maintenance/profile_distillation.py` | gpt-4o-mini | every 5 episodes |
| 5 | **Knot synthesis** | `utils/maintenance/knot_synthesis.py` | Gemma 3 4B local (gpt-4o-mini fallback) | per qualifying knot (rate-capped) |
| 6 | **Contradiction classifier** | `utils/maintenance/drift_detection.py` | gpt-4o-mini | per turn (heavily cached) |

## Add-episode flow

```
add_episode(text)
  └→ graphiti extracts entities + edges via gpt-4o-mini
  └→ resolves nodes (dedup against existing)
  └→ resolves edges (dedup, invalidations)
  └→ writes Episodic + Entity + RELATES_TO + MENTIONS to Neo4j
  └→ returns AddEpisodeResults
       │
       ├─ COGRAM_FULL_PIPELINE=true:
       │     └→ asyncio.create_task(cogram_post_write(...))
       │           ├─ intent_annotation: annotate every new edge
       │           ├─ node_narration: narrate hub entities
       │           ├─ profile_distillation: redistill DirectorProfile (every 5th)
       │           ├─ knot_synthesis: detect + synthesize hub narratives
       │           ├─ Redis active_memory.flush(group_id)
       │           └─ events.publish(GRAPH_CHANGE)
       │
       └─ Returns to caller in ~3s (background pipeline takes another ~15s)
```

## Search flow

```
search_graph(query, group_id)
  └→ Check Redis active subgraph for group_id
  │     └→ HIT: vector search inside cached subgraph (<1ms)
  │     └→ MISS: pull from Neo4j → cache in Redis → search
  └→ Cold tier: graphiti vector search (always runs as fallback)
  └→ If subjective query: traverse :DirectorProfile → :CognitivePattern → :Entity
  └→ Filter retracted edges
  └→ Merge results (dedup by edge uuid)
  └→ Return
```

## Cost-bound parameters

Five env-tunable thresholds bound the cogram pipeline's worst-case cost:

```
COGRAM_HARD_DEGREE_FLOOR=5                  # nodes below this never become knots
COGRAM_MIN_KNOT_SCORE=6.0                   # weighted (degree + annotated*1.5 + patterns*3)
COGRAM_MAX_KNOTS_PER_GROUP=25               # hard cap on knots per group
COGRAM_RESYNTHESIS_DELTA=3.0                # only re-synthesize when score moved this much
COGRAM_RESYNTHESIS_RATE_CAP_PER_HOUR=5      # per-group fire rate cap
```

These make worst-case daily cost provably bounded.

## Rate limiting (dual-gate, v0.2+)

Every LLM and embedder call passes through two sliding-window gates in
[utils/rate_limit.py](../utils/rate_limit.py):

| Gate | Default | Purpose |
|---|---|---|
| **Global** | `RATE_LIMIT_PER_MIN=150` | Stays under upstream provider's per-minute quota (OpenAI tier 1, NIM, etc.) |
| **Per-group** | `RATE_LIMIT_PER_GROUP_PER_MIN=50` | Prevents one busy `group_id` from monopolizing the global pool |

`acquire(group_id)` blocks while EITHER gate is full. The default
`group_id="default"` keeps zero-arg callers working — they share one
"default" bucket, which is fine for single-tenant deployments.

Per-group fairness applies to the post-write maintenance modules
(intent annotation, narration, profile distillation) since they have
the group_id in scope. Graphiti's extraction call and the contradiction
classifier currently share the "default" bucket because their wrappers
don't surface group_id.

To disable per-group fairness (rely on global cap only), set
`RATE_LIMIT_PER_GROUP_PER_MIN=0`.

The per-group history dict opportunistically purges idle groups
(no acquire in the last 60s) when it grows past 32 entries, so workloads
with many short-lived group_ids don't leak memory.

## MCP tool surface (v0.2.1, 14 tools)

Full reference: [docs/agent_playbook.md](agent_playbook.md). The agent's job tree maps cleanly onto a session-start ritual:

| Step | Tool | Role |
|---|---|---|
| 1 | `list_groups(query?)` | **First call.** Discover what contexts exist before searching. |
| 2 | `get_director_profile(group_id)` or `get_unified_profile()` | User's reasoning model — pick a group or merge across all. |
| 3 | `get_entity_view(name, mode)` | **Primary "what is X" tool.** Modes: `narrative` (default), `edges`, `episodes`, `all`. |
| 4 | `edges_by_pattern(pattern)` | Cross-decision routing — surfaces every prior decision under one cognitive pattern. |
| 5 | `search_graph(query, group_id)` | General semantic lookup. Profile-aware on subjective queries. |
| W | `add_episode` / `record_fact` | Write tools. Return a `task_id` you can wait on. |
| W+ | `get_episode_task(task_id, wait_seconds=N)` | Block until annotations settle before reading the graph. |

Five tools were trimmed from the v0.1 surface (19 → 14) in v0.2.1:

- **`find_connections`, `get_node_narrative`, `recent_episodes`** — merged into one `get_entity_view(name, mode)`. All three answered the same question shape (*"tell me about entity X"*) from different angles, which made tool selection harder than the work warranted. Now one tool, one entity-name input, mode picks the angle.
- **`get_knot`** — knot synthesis runs unreliably under current rate-cap + timeout interactions; agents calling `get_knot` first got an empty hint ~95% of the time. Use `get_entity_view(name)` instead. The synthesizer still runs in the background pipeline; it just isn't exposed as an agent tool yet.
- **`dedup_patterns`** — operator-grade maintenance, not agent-grade. Triggering bulk embedding calls during a chat session can merge pattern names across in-flight conversations. Now CLI-only: `python -m cogram.utils.maintenance.pattern_dedup`.
- **`confidence`** — decay-weighted scores are internal infrastructure. Agents need *"is this still believed"*, which `search_graph` already filters via retraction.

## Backward compatibility

- `Graphiti = Cogram` aliased at module level — `from cogram import Graphiti` keeps working
- `GraphitiClients = CogramClients` aliased — same
- The `Cogram` class is API-compatible with the upstream `Graphiti` class

## What's NOT in cogram

- T2 LoRA training (infrastructure exists in `containers/cogram-trainer/`, deferred until enough samples accumulate per node)
- Multi-tenant auth + Stripe billing (deferred until OSS adoption justifies it)
- AWS deployment (defer until paying customers)
- REST API for non-MCP clients (planned, not yet shipped)
- Hosted SaaS

## Reference

- Upstream graphiti: <https://github.com/getzep/graphiti>
- Cogram repo: <https://github.com/srk0102/cogram>
- License: Apache 2.0 (both)
- Attribution: see [NOTICE](../NOTICE) and [ATTRIBUTION.md](../ATTRIBUTION.md)

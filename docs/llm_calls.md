# LLM calls in cogram

Cogram fires five distinct LLM call types per episode write, plus one
contradiction classifier on drift. Each is routed to one of three
configurable tiers (Large / Small / Embedder), so you can mix providers
to optimize cost and quality.

## The 5 LLM call types + 1 classifier

| # | Call | Tier | When fired | Default model | Cost / call (gpt-4o-mini) |
|---|---|---|---|---|---|
| 1 | Graphiti entity/edge extraction | **large** | Once per `add_episode`, synchronous | `gpt-4o-mini` (env `LARGE_LLM_MODEL`) | ~$0.0008 |
| 2 | Intent annotation (`edge_kind`, `why_connected`, `director_vision`, `cognitive_pattern`) | **small** | Once per new edge, in async post-write pipeline | `gpt-4o-mini` (env `SMALL_LLM_MODEL`) | ~$0.00006 |
| 3 | Node narration (`vllm_narrative`) | **small** | Once per qualifying hub node (degree ≥ 5), capped at `COGRAM_NARRATE_CAP=5` per episode | `gpt-4o-mini` (env `SMALL_LLM_MODEL`) | ~$0.0001 |
| 4 | Profile distillation | **small** | Per group_id, throttled (`COGRAM_PROFILE_EVERY_N=5` episodes) | `gpt-4o-mini` (env `SMALL_LLM_MODEL`) | ~$0.0005 |
| 5 | Knot synthesis | **gemma** | Per qualifying knot node, rate-capped 5/hr/group | `gemma3n:e4b` (env `GEMMA_MODEL`); fallback `gpt-4o-mini` | $0 (local) / ~$0.0001 (fallback) |
| C | Contradiction classifier (drift signal) | **small** | Once per episode that hits the cosine-drift gate | `gpt-4o-mini` (env `SMALL_LLM_MODEL`), cached by turn text | ~$0.00005 |

Plus embedding calls (entities, episodes, edges, narratives) on the
**embedder** tier (`EMBEDDER_MODEL`, default `text-embedding-3-small`).

## Recommended provider matrix

For minimum cost while keeping quality where it matters:

```env
# Heavy extraction → OpenAI gpt-4o for accuracy
LARGE_LLM_MODEL=gpt-4o
LARGE_LLM_API_KEY=sk-openai-...

# Cheap pipeline calls → DeepSeek or local Qwen (one-shot, structured)
SMALL_LLM_MODEL=deepseek-chat
SMALL_LLM_API_KEY=sk-deepseek-...
SMALL_LLM_BASE_URL=https://api.deepseek.com/v1

# Embeddings → OpenAI (best $/recall ratio)
EMBEDDER_MODEL=text-embedding-3-small
EMBEDDER_API_KEY=sk-openai-...

# Knots → local Gemma via Ollama (free, runs once per hub)
GEMMA_BASE_URL=http://host.docker.internal:11434/v1
GEMMA_MODEL=gemma3n:e4b
```

Setting `LARGE_LLM_*` and `SMALL_LLM_*` to different providers cuts the
post-write pipeline cost by ~10× while preserving extraction quality.

## What gets cached

Engram cache (Postgres `engram.patterns` table) wraps the small-tier
calls so a re-issued prompt within ~24h returns the cached result —
no LLM call at all. Cached calls don't count against `RATE_LIMIT_PER_MIN`
and are returned in <5ms.

The contradiction classifier specifically caches by normalized turn
text so the same conversational turn never re-pays the call cost.

## How tier selection happens

Each call site reads its tier's fields from the `Settings` dataclass
(see [cogram/core/config.py](../core/config.py)):

| Call | Reads from |
|---|---|
| Graphiti extraction | `large_llm_model`, `large_llm_api_key`, `large_llm_base_url` (via `build_graphiti`) |
| Intent annotation | `small_llm_model`, `small_llm_api_key`, `small_llm_base_url` |
| Node narration | same — small tier |
| Profile distillation | same — small tier |
| Contradiction classifier | same — small tier |
| Knot synthesis | `GEMMA_MODEL`, `GEMMA_BASE_URL` (separate path) |
| Embeddings | `embedder_model`, `embedder_api_key`, `embedder_base_url` |

Backward-compat: legacy env vars (`GRAPHITI_LLM_MODEL`,
`ANNOTATOR_LLM_MODEL`, `EMBEDDING_MODEL`, `OPENAI_API_KEY`,
`LLM_BASE_URL`) are still read as fallbacks — existing v0.1 `.env`
files keep working unchanged.

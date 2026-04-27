# The annotator drift problem

> **Status (v0.2):** mitigated by the `edge_kind` taxonomy described below.
> **Status (v0.1):** known limitation, mitigated only by manual `record_fact` use.

## What the flaw is

Cogram's intent annotator (`utils/maintenance/intent_annotation.py`) reads each
new edge from Neo4j, looks at the source episode, and asks gpt-4o-mini for:

```json
{
  "why_connected":     "...",
  "director_vision":   "...",
  "cognitive_pattern": "..."
}
```

Those three fields drive the Director profile: every edge with a non-empty
`cognitive_pattern` becomes a `:CognitivePattern` reinforcement, every
`director_vision` becomes a recurring goal, and the LLM compresses every
`why_connected` into the working-style summary.

The flaw: **the annotator can't tell the difference between a fact the
Director believes and a fact the Director was merely citing.** Two examples
that produced bad profile output before v0.2:

1. *"Mem0 uses Qdrant for embeddings."* — a competitor comparison.
   The annotator wrote `cognitive_pattern: "vector-store-driven memory"` and
   the Director profile started claiming Director values vector stores.
   Director was actually arguing the opposite (cogram uses graph-shaped
   storage on purpose).

2. *"Graphiti stores edges in Neo4j."* — a neutral background fact about the
   upstream project cogram is forked from.
   The annotator wrote `director_vision: "centralize knowledge in graph
   form"` as if Director had endorsed Neo4j; really, Director was just
   stating a fact about a third-party tool.

In both cases the annotator confused *context* (background information about
something the Director was reasoning about) with *intent* (the Director's own
beliefs and actions). With enough such edges, the Director profile drifts
toward a profile of the third-party tools the Director happens to be reading
about, not the Director's own thinking.

## Why the model can't tell on its own

The annotation prompt only sees:
- two entity names
- two short summaries
- the edge fact (one sentence)
- ~3000 chars of episode excerpt

For an episode that says "I'm choosing graph-shaped memory because mem0 uses
Qdrant and that's not what I want", the LLM sees the entity pair (`mem0`,
`Qdrant`) plus a fact like "mem0 uses Qdrant" and has every reason to assume
the Director endorses the relationship. The contrastive intent of the
surrounding sentence is easily lost.

## v0.2 fix — `edge_kind` taxonomy

Each annotated edge now carries a fifth field:

```json
{
  "edge_kind":         "principle | action | context | competitor | unknown",
  "why_connected":     "...",
  "director_vision":   "...",
  "cognitive_pattern": "..."
}
```

Definitions:

| edge_kind  | meaning                                                                                                                  | example                                                              |
|------------|--------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------|
| principle  | a belief, value, preference, rule, or stance the Director holds                                                          | *"I always cache LLM outputs because of cost."*                      |
| action     | something the Director did, decided, built, deployed, or chose                                                           | *"Deployed cogram-mcp to ghcr.io."*                                  |
| context    | a neutral background fact about a third-party tool, product, or concept the Director is reasoning about (not endorsing) | *"Graphiti stores edges in Neo4j."*                                  |
| competitor | information about a competing product or alternative approach the Director is comparing AGAINST (not endorsing)         | *"Mem0 uses Qdrant for embeddings."*                                 |
| unknown    | genuinely unclear from fact + episode                                                                                    | *"Anthropic published a paper."* (could be principle, could be context) |

### Where the field flows

1. **Annotation** — `intent_annotation.py` asks the LLM for `edge_kind` along
   with the existing three fields. The prompt now includes explicit
   heuristics that push the LLM toward `context` / `competitor` when the
   edge mentions third-party tool names or comparison phrases ("unlike X",
   "vs Y", "competitor", "alternative").

2. **Storage** — written into `r.intent_meta` (a JSON blob on the edge), so
   no schema migration. Existing edges without `edge_kind` are treated as
   `unknown`.

3. **Profile distillation** — `pipeline/post_write.py:_collect_annotated_edges_for_group`
   filters edges by `edge_kind`. Only `principle`, `action`, and `unknown`
   contribute to the Director profile and `:CognitivePattern` graph. Edges
   classified as `context` or `competitor` are stored normally and
   searchable, but they don't reinforce the Director's self-model.

4. **Search & traversal** — unchanged for v0.2. Context/competitor edges
   are still returned by `search_graph` and `find_connections` so agents can
   reason about competitors; they're only excluded from the profile layer.

The eligible-kinds set is centralized as
`PROFILE_ELIGIBLE_KINDS = {"principle", "action", "unknown"}` in
`utils/maintenance/profile_distillation.py` so future tuning (e.g. weighting
instead of filtering) only changes one spot.

### Defensive normalization

LLMs sometimes return synonyms (`belief`, `decision`, `external`,
`alternative`). `intent_annotation._normalize_edge_kind` maps the common
ones to the canonical five and falls back to `unknown` for anything
unrecognized so a misbehaving model never breaks downstream consumers.

## What's still imperfect

- **Sarcasm & negation**: *"Of course Mem0 'solves' memory."* — depending on
  episode context, the annotator may still misclassify; the fix would
  require longer context windows or a second pass.
- **Self-citation**: when the Director quotes themselves in third person
  ("the user said X"), the annotator sometimes flips edge_kind=context. The
  fix would be a Director-name registry the prompt can reference; deferred.
- **Backfill**: this version doesn't rewrite existing edges. Edges
  annotated before v0.2 stay at `edge_kind=unknown`. A backfill pass via
  `python -m cogram.utils.maintenance.intent_annotation` will re-run them,
  but legacy intent_meta on already-annotated edges is preserved
  (re-annotation only touches edges where `intent_meta IS NULL`). To force
  a re-pass, clear `r.intent_meta` for the targeted edges first.

## Roadmap

- **v0.2 (this branch):** `edge_kind` field + filtered profile distillation.
- **v0.3 (next):** opt-in backfill MCP tool that re-annotates edges and
  diffs `cognitive_pattern` distribution before/after, so users can audit
  the impact on their existing profile.
- **v0.3:** weighted distillation — instead of filtering context/competitor
  outright, weight `principle=1.0`, `action=0.7`, `unknown=0.3`,
  `context/competitor=0.0`. Lets neutral edges contribute proportionally.
- **v0.4:** entity-level `is_director_owned` flag so the annotator knows
  which entity names are the Director's products (cogram, engram, …) vs.
  external tools (graphiti, mem0, neo4j, …). Drops failure rate on
  ambiguous edges sharply.

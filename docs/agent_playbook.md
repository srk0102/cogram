# Cogram Agent Playbook

A one-page reference for any LLM agent (Claude, GPT, custom) connecting to a
cogram MCP server. Read this once at session start and you'll route correctly
every time.

Cogram exposes **14 tools**. They cluster into four rituals:

---

## 1. Session-start ritual — *"what does this user have?"*

Run these in order at the start of any session, before answering questions
that depend on user history.

```
list_groups()
  → discover what contexts (group_ids) exist
  → returns each group_id with episode/entity/pattern/knot counts

get_unified_profile()        OR   get_director_profile(group_id)
  → cross-group view             → one group's working-style summary,
  → for "who is this user        → recurring visions, top cognitive patterns
     across all their work"      → with concrete reinforcing-edge examples
```

That's the bootstrap. Two calls, ~1 second, and the agent has a model of how
this user thinks and what contexts they work in.

---

## 2. Decision-time ritual — *"what would the user do here?"*

When the agent is about to make a constrained choice (approve a PR, pick a
tool, accept or reject a suggestion), surface the relevant prior decisions:

```
list_cognitive_patterns(query="<topic>")
  → discover pattern names like "legal risk mitigation", "cost-aware prototyping"

edges_by_pattern(pattern="<name>")
  → every prior decision under that thinking style
  → with why_connected + director_vision intact
```

This is the **cross-decision routing primitive**. It's what turns *"the user
rejected X once"* into *"the user has a recurring rule about Y; here are 7
prior decisions reinforcing it."* Lead with this in any agent flow that has
to be coherent across surfaces.

---

## 3. Lookup ritual — *"tell me about X"*

Three layers, pick the cheapest that answers the question:

```
get_entity_view(entity_name)                          [mode="narrative", default]
  → 2-4 sentence compressed view, stance, open questions, cognitive_pattern_label
  → confidence + decay so you know if it's stale
  → THE primary "what is X" tool

get_entity_view(entity_name, mode="edges")
  → raw edge dump with intent_meta on every edge
  → use when narrative feels stale or you need provenance

get_entity_view(entity_name, mode="episodes")
  → recent episode excerpts that mention this entity
  → use to find episode uuids for retract or get_episode

get_entity_view(entity_name, mode="all")
  → all three at once when you're not sure which you need
```

For fuzzy semantic search (no specific entity in mind):

```
search_graph(query, group_id)
  → semantic search across all edges
  → profile-aware on subjective queries ("would I...", "how do I feel...")
  → Redis hot-tier accelerated; <1ms after first warm
```

---

## 4. Write ritual — *"remember this"*

Two write tools, same async pipeline:

```
add_episode(content, group_id)
  → for narrative prose
  → returns task_id immediately (~3s); pipeline runs ~15s in background

record_fact(subject, predicate, object, group_id)
  → for clean SPO statements
  → constructs "{subject} {predicate} {object}" and writes
  → locks edge structure better than free prose

# Both return cogram.task_id. Block on it BEFORE reading the graph:
get_episode_task(task_id, wait_seconds=20)
  → state ∈ running|done|failed|cancelled
  → wait_seconds=0 peeks; wait_seconds=N blocks up to N seconds
  → without this, you'll do stale reads
```

If a task is misbehaving:

```
list_episode_tasks(group_id)
  → newest-first listing of in-flight + recent tasks

cancel_episode_task(task_id)
  → cooperative cancel; written annotations stay, only future steps skip
```

---

## 5. Retract ritual — *"that was wrong"*

When you (or the user) catches an annotator hallucination or a stale fact:

```
get_entity_view(entity_name, mode="edges")
  → verify what's there before retracting

retract(target=<name|edge_uuid|episode_uuid>, reason="<why>")
  → marks edges as no-longer-believed
  → kept on disk for audit; filtered from search_graph
  → BUMPS per-entity cache_version so narratives re-generate on next read
```

**Critical gotcha:** `retract` does NOT accept the friendly `episode_name`
slug (`mcp_<timestamp>`) that `add_episode` returns. It accepts:
- (a) Entity name (substring), OR
- (b) Edge uuid, OR
- (c) **Episodic uuid** — get this via `get_entity_view(name, mode="episodes")`
      or `get_episode(uuid)`.

---

## Anti-patterns (mistakes prior agents made)

1. **Going straight to entity-keyed tools without `list_groups` first.** If
   you don't know what `group_id`s exist, you can't pass a meaningful one
   to `search_graph` or `get_director_profile`. List groups first.

2. **Calling `retract(target=episode_name)`** with the `mcp_<ts>` slug. That's
   a friendly name, not a uuid. Use `get_episode(uuid)` to find the real one.

3. **Reading the graph immediately after `add_episode`/`record_fact`.** The
   pipeline runs async. Without `get_episode_task(task_id, wait_seconds=20)`
   you'll see un-annotated edges and miss the intent_meta layer that makes
   cogram useful.

4. **Treating `find_connections` as the primary "what is X" tool.** It used
   to be (in v0.1). It's now `get_entity_view` (default mode=narrative).
   Edge dumps are an audit/fallback, not the headline.

5. **Writing the same canonical entity (you, your products) under multiple
   `group_id`s.** Groups isolate contexts on purpose; the entity gets
   fragmented into multiple UUIDs. For cross-context entities, pick one
   canonical group_id and stick with it. Or use `get_unified_profile` to
   merge views across groups at read time.

---

## The 14 tools at a glance

| Phase | Tool | Role |
|---|---|---|
| Discovery | `list_groups` | First call — what contexts exist |
| Discovery | `list_cognitive_patterns` | Discover thinking styles |
| Profile | `get_director_profile(group_id)` | One group's reasoning model |
| Profile | `get_unified_profile()` | Merged across all groups |
| Lookup | `get_entity_view(name, mode)` | What is X — narrative / edges / episodes |
| Lookup | `search_graph(query, group_id)` | Fuzzy semantic search |
| Lookup | `get_episode(uuid)` | Full episode body by uuid |
| Routing | `edges_by_pattern(pattern)` | Cross-decision routing primitive |
| Write | `add_episode(content)` | Write prose |
| Write | `record_fact(s, p, o)` | Write SPO statement |
| Write | `retract(target)` | Undo a fact |
| Tasks | `list_episode_tasks` | List background pipeline tasks |
| Tasks | `get_episode_task(task_id, wait_seconds)` | Wait/peek |
| Tasks | `cancel_episode_task(task_id)` | Abort a runaway task |

That's the whole surface. Memorize the rituals, not the tool list — the
tools fall into place once you know which question you're answering.

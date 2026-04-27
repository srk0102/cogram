"""COGRAM MCP server.

Exposes the cogram graph (entities + edges + intent_meta + vllm_narrative + confidence)
as MCP tools. Supports two transports:
  - stdio (default; for desktop MCP clients launching this as a child process)
  - HTTP/SSE on :7800/mcp (for remote clients, container-to-container, or browsers)

Run:
  python -m src.mcp_server                    # stdio
  python -m src.mcp_server --http             # HTTP/SSE on 0.0.0.0:7800
  python -m src.mcp_server --http --port 7800 # explicit
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from cogram.core.config import build_graphiti, Settings
from cogram.utils.confidence import effective_confidence, label_for, normalized_confidence

PROFILE_PATH = Path(__file__).resolve().parent.parent / "director_profile.json"

# When PROFILE_INJECTION=false the server behaves like stock graphiti — no
# Director profile, no Cypher traversal on subjective queries. Useful for A/B
# benchmarking: run two MCP instances side-by-side, identical graph, identical
# tools, only the retrieval layer differs.
PROFILE_INJECTION = os.environ.get("COGRAM_PROFILE_INJECTION", "true").lower() != "false"
SERVER_NAME = os.environ.get("COGRAM_SERVER_NAME", "cogram")

mcp = FastMCP(SERVER_NAME)
_graphiti = None


def _g():
    global _graphiti
    if _graphiti is None:
        _graphiti = build_graphiti()
    return _graphiti


_settings_cache: Settings | None = None


def _settings() -> Settings:
    global _settings_cache
    if _settings_cache is None:
        _settings_cache = Settings.from_env()
    return _settings_cache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_json(value: Any):
    if not value:
        return None
    if isinstance(value, dict):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None


def _parse_meta(attributes):
    if not attributes:
        return None
    raw = attributes.get("intent_meta") if isinstance(attributes, dict) else None
    return _parse_json(raw)


# ---------------------------------------------------------------------------
# Existing tools (kept from prior version)
# ---------------------------------------------------------------------------

_PROFILE_CACHE_TTL_SECONDS = int(os.environ.get("COGRAM_PROFILE_CACHE_TTL", "300"))


def _profile_cache_key(group_id: str, top_patterns: int, examples_per_pattern: int) -> str:
    return f"cogram:cache:profile:{group_id}:top{top_patterns}:ex{examples_per_pattern}"


@mcp.tool()
async def get_knot(entity_name: str, format: str = "both") -> str:
    """Return the pre-synthesized 'knot' narrative + raw subgraph for a hub entity.

    Cogram pre-synthesizes one narrative paragraph per qualifying hub node (degree
    >= 5, knot_score >= threshold) using Gemma 3 4B local (or gpt-4o-mini fallback)
    and caches BOTH forms in Redis:
      - narrative: a 3-5 sentence paragraph an LLM agent can drop into context
      - subgraph: raw entities + edges + intent_meta as JSON

    `format` selects which to return: 'narrative' | 'subgraph' | 'both'.

    If the entity isn't a knot yet (low degree or under threshold), returns
    a hint to use search_graph or find_connections instead.
    """
    g = _g()
    cypher = """
    MATCH (n:Entity)
    WHERE toLower(n.name) CONTAINS toLower($q)
    RETURN n.uuid AS uuid, n.name AS name LIMIT 1
    """
    async with g.driver.session() as session:
        rec = await (await session.run(cypher, q=entity_name)).single()
    if rec is None:
        return json.dumps({"ok": False, "error": f"No entity matches {entity_name!r}."}, indent=2)

    try:
        from cogram.utils.maintenance.knot_synthesis import get_knot as _get_knot
        knot = await _get_knot(rec["uuid"])
    except Exception as exc:
        return json.dumps({"ok": False, "error": f"knots module error: {exc}"}, indent=2)

    if not knot or (not knot.get("narrative") and not knot.get("subgraph")):
        return json.dumps({
            "ok": False,
            "entity": rec["name"],
            "uuid": rec["uuid"],
            "hint": "This entity isn't a synthesized knot yet (likely low degree or below score threshold). Try search_graph or find_connections.",
        }, indent=2)

    out = {"ok": True, "entity": rec["name"], "uuid": rec["uuid"], "meta": knot.get("meta")}
    if format in ("narrative", "both"):
        out["narrative"] = knot.get("narrative")
    if format in ("subgraph", "both"):
        out["subgraph"] = knot.get("subgraph")
    return json.dumps(out, indent=2, default=str)


@mcp.tool()
async def get_director_profile(
    group_id: str = "default",
    top_patterns: int = 10,
    examples_per_pattern: int = 2,
) -> str:
    """Return the compressed Director profile: cognitive patterns, recurring
    visions, and a working-style summary distilled from every annotated edge in
    the graph.

    Each cognitive pattern is returned with `examples` — concrete (entity → entity)
    edges with their `why_connected` and `director_vision` annotations that
    actually reinforced this pattern. This is the **why** behind each label.

    By default returns the top 10 patterns by confidence; pass top_patterns=N to
    expand. Result is cached in Redis for 5 minutes; cache is invalidated on
    profile_change events (new ingestion, retraction). Single Cypher query
    (no 1+N round-trips even with many patterns).
    """
    # ---- Redis cache check (assembled JSON) ----
    cache_key = _profile_cache_key(group_id, top_patterns, examples_per_pattern)
    try:
        import redis.asyncio as _r
        _redis = _r.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True)
        cached = await _redis.get(cache_key)
        if cached:
            await _redis.aclose()
            return cached
    except Exception:
        _redis = None

    g = _g()

    # ---- Single Cypher: profile + top-N patterns + examples per pattern, all in one round-trip ----
    single_cypher = """
    MATCH (p:DirectorProfile {group_id: $g})
    OPTIONAL MATCH (p)-[:HAS_PATTERN]->(pat:CognitivePattern {group_id: $g})
    WITH p, pat
    ORDER BY coalesce(pat.confidence, 0) DESC
    LIMIT $top_patterns
    WITH p, collect(pat) AS top_pats

    UNWIND CASE WHEN size(top_pats) = 0 THEN [null] ELSE top_pats END AS pat
    OPTIONAL MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
        WHERE pat IS NOT NULL
          AND r.intent_meta IS NOT NULL
          AND r.retracted_at IS NULL
          AND r.cognitive_pattern_name = pat.name
          AND coalesce(a.group_id, b.group_id, 'default') = $g
    WITH p, pat,
         collect({
             a: a.name,
             b: b.name,
             fact: coalesce(r.fact, ''),
             intent_meta: r.intent_meta
         })[0..$cap] AS examples
    WITH p,
         collect(CASE WHEN pat IS NULL THEN null ELSE {
             name: pat.name,
             count: pat.count,
             confidence: pat.confidence,
             examples: examples
         } END) AS patterns

    RETURN
        p.summary AS summary,
        coalesce(p.visions, []) AS visions,
        p.generated_at AS generated_at,
        p.updated_at AS updated_at,
        [pat IN patterns WHERE pat IS NOT NULL] AS patterns
    """

    async with g.driver.session() as session:
        rec = await (await session.run(
            single_cypher, g=group_id, top_patterns=top_patterns, cap=examples_per_pattern
        )).single()

    if rec is None or rec["summary"] is None:
        if PROFILE_PATH.exists():
            return PROFILE_PATH.read_text(encoding="utf-8")
        return json.dumps({
            "error": f"No DirectorProfile yet for group_id={group_id!r}. Either ingest some episodes (the cogram pipeline distills it automatically) or run `python -m src.profile`.",
            "group_id": group_id,
        }, indent=2)

    # Total pattern count for pagination metadata
    total_cypher = "MATCH (:DirectorProfile {group_id: $g})-[:HAS_PATTERN]->(pat:CognitivePattern {group_id: $g}) RETURN count(pat) AS total"
    async with g.driver.session() as session:
        total_rec = await (await session.run(total_cypher, g=group_id)).single()
        total_patterns = (total_rec["total"] if total_rec else 0) or 0

    patterns_out: list[dict] = []
    for pat in rec["patterns"] or []:
        examples_list = []
        for ex in (pat.get("examples") or []):
            if not ex.get("a"):
                continue
            meta = _parse_json(ex.get("intent_meta")) or {}
            examples_list.append({
                "between": f"{ex['a']} → {ex['b']}",
                "fact": (ex.get("fact") or "")[:200],
                "why_connected": meta.get("why_connected", ""),
                "director_vision": meta.get("director_vision", ""),
            })
        patterns_out.append({
            "name": pat.get("name"),
            "confidence": pat.get("confidence"),
            "count": pat.get("count"),
            "examples": examples_list,
        })

    body = {
        "group_id": group_id,
        "working_style_summary": rec["summary"],
        "recurring_visions": rec["visions"],
        "cognitive_patterns": patterns_out,
        "pagination": {
            "returned": len(patterns_out),
            "total": total_patterns,
            "top_patterns": top_patterns,
            "more_available": total_patterns > top_patterns,
            "next_call_hint": f"call get_director_profile(group_id={group_id!r}, top_patterns={total_patterns}) to fetch all" if total_patterns > top_patterns else None,
        },
        "generated_at": rec["generated_at"],
        "updated_at": rec["updated_at"],
    }
    payload = json.dumps(body, indent=2)

    # ---- Cache assembled JSON (TTL'd; invalidated on profile_change) ----
    if _redis is not None:
        try:
            await _redis.set(cache_key, payload, ex=_PROFILE_CACHE_TTL_SECONDS)
            await _redis.aclose()
        except Exception:
            pass

    return payload


import math as _math
import re as _re
import time as _time

_FIRST_PERSON_RE = _re.compile(
    r"\b(I|my|me|we|our|would|should|prefer|feel|think|approve|reject|refuse|"
    r"choose|believe|value|hate|love|stance|opinion)\b",
    _re.IGNORECASE,
)

# 30-day half-life for confidence decay
_HALF_LIFE_SECONDS = 30 * 86_400


def _decay(stored: float, last_reinforced_ts: float, now: float | None = None) -> float:
    """Same math as src/confidence.py — duplicated here so MCP server stays self-contained."""
    if now is None:
        now = _time.time()
    age = max(0.0, now - last_reinforced_ts)
    if _HALF_LIFE_SECONDS == 0 or age == 0:
        return stored
    return stored * _math.pow(0.5, age / _HALF_LIFE_SECONDS)


# ---------------------------------------------------------------------------
# Profile-aware retrieval via Cypher path traversal
# ---------------------------------------------------------------------------

# Pull the Director profile + top patterns by EFFECTIVE confidence
PROFILE_TRAVERSAL_QUERY = """
MATCH (p:DirectorProfile)
OPTIONAL MATCH (p)-[:HAS_PATTERN]->(pat:CognitivePattern)
WITH p, pat
ORDER BY pat.confidence DESC
LIMIT 8
RETURN
    p.summary AS summary,
    p.visions AS visions,
    collect({
        name: pat.name,
        confidence: pat.confidence,
        last_reinforced: pat.last_reinforced
    }) AS patterns
"""

# Find edges connected to a pattern (via the entities it reinforces)
EDGES_FROM_PATTERN_QUERY = """
MATCH (pat:CognitivePattern {name: $pattern})-[:REINFORCED_BY]->(n:Entity)
MATCH (n)-[r:RELATES_TO]-(other:Entity)
WHERE r.intent_meta IS NOT NULL
RETURN
    n.name AS source_name,
    other.name AS target_name,
    coalesce(r.fact, '') AS fact,
    r.intent_meta AS intent_meta
LIMIT 5
"""


async def _traverse_profile(g) -> dict | None:
    """Pull profile + top patterns + edges that REINFORCED each pattern.
    Returns None if no :DirectorProfile node exists yet."""
    async with g.driver.session() as session:
        result = await session.run(PROFILE_TRAVERSAL_QUERY)
        row = await result.single()

    if row is None or row["summary"] is None:
        return None

    now = _time.time()
    patterns_with_decay = []
    for p in (row["patterns"] or []):
        if not p.get("name"):
            continue
        stored = float(p.get("confidence", 0) or 0)
        last_ts = float(p.get("last_reinforced", 0) or 0)
        eff = _decay(stored, last_ts, now) if last_ts else stored
        patterns_with_decay.append({
            "name": p["name"],
            "stored_confidence": round(stored, 2),
            "effective_confidence": round(eff, 3),
        })

    # Sort by effective confidence (after decay), highest first
    patterns_with_decay.sort(key=lambda x: -x["effective_confidence"])

    # For each top pattern, pull a few example edges that reinforce it
    top_patterns = patterns_with_decay[:5]
    pattern_evidence = []
    async with g.driver.session() as session:
        for p in top_patterns:
            edge_rows = [
                r.data() async for r in await session.run(
                    EDGES_FROM_PATTERN_QUERY, pattern=p["name"]
                )
            ]
            evidence = []
            for er in edge_rows:
                meta = _parse_json(er.get("intent_meta"))
                evidence.append({
                    "source": er["source_name"],
                    "target": er["target_name"],
                    "fact": er["fact"],
                    "why": (meta or {}).get("why_connected", "") if meta else "",
                })
            pattern_evidence.append({**p, "examples": evidence[:3]})

    return {
        "summary": row["summary"],
        "visions": row["visions"] or [],
        "top_patterns": pattern_evidence,
    }


@mcp.tool()
async def search_graph(query: str, limit: int = 10, group_id: str = "default") -> str:
    """Semantic search across the knowledge graph.

    Profile-aware: if the query is subjective ('would I...', 'how do I feel...'),
    Cypher-traverses (:DirectorProfile)-[:HAS_PATTERN]->(:CognitivePattern)-
    [:REINFORCED_BY]->(:Entity) to surface the user's stance + reinforcing
    edges with confidence-decay weighting.

    Hot-tier accelerated: per-group_id active subgraph is cached in Redis. The
    first call pulls the relevant subgraph from Neo4j into Redis (~50ms one-time);
    subsequent searches within the same group_id check the in-memory subgraph
    first (<1ms hits). Cold tier (Neo4j vector search) always runs as fallback.
    """
    g = _g()
    response: dict = {}
    is_subjective = bool(_FIRST_PERSON_RE.search(query)) and PROFILE_INJECTION

    if is_subjective:
        profile_data = await _traverse_profile(g)
        if profile_data is not None:
            response["director_profile"] = {
                **profile_data,
                "_source": "cypher_traversal",
                "_note": "Injected via :DirectorProfile graph traversal because the query is subjective",
            }

    # ---- Hot tier: Redis active subgraph ---------------------------------
    redis_hits: list[dict] = []
    cache_status = "disabled"
    try:
        from cogram.driver import redis_active as _am
        sub = await _am.load(group_id)
        if sub is None:
            # Cold start: embed query as seed and pull a focused subgraph
            try:
                seed_embedding = await g.embedder.create(query)
            except Exception:
                seed_embedding = None
            sub = await _am.pull_subgraph(g, group_id, group_id, seed_embedding)
            cache_status = "warmed"
        else:
            cache_status = "hit"

        # Vector search within the cached subgraph
        if sub.entities or sub.edges:
            try:
                q_emb = await g.embedder.create(query)
                cached_edges = _am.search_edges_in_subgraph(sub, q_emb, k=limit)
                for ce in cached_edges:
                    redis_hits.append({
                        "edge_uuid": ce.uuid,
                        "source_node": ce.source_uuid,
                        "target_node": ce.target_uuid,
                        "fact": ce.fact,
                        "intent_meta": ce.intent_meta,
                        "_tier": "hot",
                    })
                if redis_hits:
                    sub.hit_count += 1
                else:
                    sub.miss_count += 1
                await _am._save(sub)
            except Exception:
                cache_status += "+search_error"
    except Exception as cache_exc:
        cache_status = f"error: {cache_exc}"

    # ---- Cold tier: graphiti vector search (always runs) -----------------
    try:
        edges = await g.search(query, num_results=limit * 2)  # over-fetch since we'll filter retractions
    except Exception:
        edges = []
    cold_results: list[dict] = []

    # Pull retracted edge uuids in one shot so we can filter
    retracted: set[str] = set()
    try:
        async with g.driver.session() as ses:
            rows = [r.data() async for r in await ses.run(
                "MATCH ()-[r:RELATES_TO]->() WHERE r.retracted_at IS NOT NULL RETURN r.uuid AS uuid"
            )]
        retracted = {r["uuid"] for r in rows if r.get("uuid")}
    except Exception:
        pass

    for e in edges:
        edge_uuid = getattr(e, "uuid", None)
        if edge_uuid and edge_uuid in retracted:
            continue
        cold_results.append({
            "source_node": e.source_node_uuid,
            "target_node": e.target_node_uuid,
            "fact": e.fact,
            "name": getattr(e, "name", None),
            "intent_meta": _parse_meta(getattr(e, "attributes", {})),
            "_tier": "cold",
        })
        if len(cold_results) >= limit:
            break

    # Also drop retracted edges from the hot-tier results
    redis_hits = [
        h for h in redis_hits
        if h.get("source_node") and h.get("target_node")
        and (not h.get("edge_uuid") or h["edge_uuid"] not in retracted)
    ]

    # Merge: prefer hot-tier results, dedup by (source, target, fact prefix)
    seen: set[tuple[str, str, str]] = set()
    merged: list[dict] = []
    for entry in redis_hits + cold_results:
        key = (entry.get("source_node") or "", entry.get("target_node") or "", (entry.get("fact") or "")[:80])
        if key in seen:
            continue
        seen.add(key)
        merged.append(entry)

    response["edges"] = merged[:limit]
    response["_cache"] = {
        "tier": cache_status,
        "hot_hits": len(redis_hits),
        "cold_hits": len(cold_results),
        "group_id": group_id,
    }

    if not response["edges"] and not response.get("director_profile"):
        return "No matching edges and no profile available."
    return json.dumps(response, indent=2, default=str)


@mcp.tool()
async def find_connections(entity_name: str, limit: int = 25) -> str:
    """Return all edges connected to an entity by name (case-insensitive
    substring match). Useful for 'show me everything connected to Graphiti'
    or 'what does the Director say about Kimi K2'."""
    g = _g()
    cypher = """
    MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
    WHERE toLower(a.name) CONTAINS toLower($q) OR toLower(b.name) CONTAINS toLower($q)
    RETURN a.name AS a, b.name AS b,
           coalesce(r.fact, r.name, '') AS fact,
           r.intent_meta AS intent_meta
    LIMIT $limit
    """
    async with g.driver.session() as session:
        rows = [r.data() async for r in await session.run(cypher, q=entity_name, limit=limit)]
    out = []
    for r in rows:
        out.append({
            "a": r["a"], "b": r["b"], "fact": r["fact"],
            "intent_meta": _parse_json(r.get("intent_meta")),
        })
    return json.dumps(out, indent=2, default=str)


@mcp.tool()
async def list_groups(query: str = "", limit: int = 20, offset: int = 0) -> str:
    """List group_ids in the graph with episode/entity/pattern/knot counts.

    Args:
      query:  optional substring filter on group_id (case-insensitive).
              e.g. query='cogram' matches 'cogram-knowledge' and 'cogram-test'.
              Empty string returns all (default).
      limit:  max number of groups to return (default 20, max 200).
      offset: skip the first N matches for pagination (default 0).

    Use at session start to discover what projects / contexts the user has
    memory for, then route subsequent queries to the right group_id. The
    query parameter is the cheap way to find a specific group when the user
    has many — don't pull the entire list if you only want the cogram-* ones.

    Returns paginated results with `total` (matched groups) and `returned`
    (groups in this page). If `total > offset + returned`, call again with
    a higher offset to get the next page.
    """
    g = _g()
    limit = max(1, min(int(limit or 20), 200))
    offset = max(0, int(offset or 0))
    needle = (query or "").strip().lower()

    eps_q = "MATCH (e:Episodic) RETURN coalesce(e.group_id, 'default') AS gid, count(e) AS n"
    ent_q = "MATCH (n:Entity) RETURN coalesce(n.group_id, 'default') AS gid, count(n) AS n"
    pat_q = "MATCH (p:CognitivePattern) RETURN coalesce(p.group_id, 'default') AS gid, count(p) AS n"
    knot_q = "MATCH (k:Entity) WHERE k.knot_narrative IS NOT NULL RETURN coalesce(k.group_id, 'default') AS gid, count(k) AS n"
    profile_q = "MATCH (p:DirectorProfile) RETURN coalesce(p.group_id, 'default') AS gid"

    async with g.driver.session() as session:
        eps   = {r["gid"]: r["n"] for r in [r.data() async for r in await session.run(eps_q)]}
        ents  = {r["gid"]: r["n"] for r in [r.data() async for r in await session.run(ent_q)]}
        pats  = {r["gid"]: r["n"] for r in [r.data() async for r in await session.run(pat_q)]}
        knots = {r["gid"]: r["n"] for r in [r.data() async for r in await session.run(knot_q)]}
        profiles = {r["gid"] for r in [r.data() async for r in await session.run(profile_q)]}

    all_gids = set(eps) | set(ents) | set(pats) | set(knots) | profiles

    # Substring filter (skip when needle is empty)
    if needle:
        all_gids = {g for g in all_gids if needle in g.lower()}

    rows_all = sorted(
        [
            {
                "group_id": gid,
                "episodes": eps.get(gid, 0),
                "entities": ents.get(gid, 0),
                "patterns": pats.get(gid, 0),
                "knots": knots.get(gid, 0),
                "has_director_profile": gid in profiles,
            }
            for gid in all_gids
        ],
        key=lambda r: (-r["episodes"], -r["entities"]),
    )

    total = len(rows_all)
    page = rows_all[offset : offset + limit]

    response: dict = {"groups": page, "total": total, "returned": len(page)}
    if query:
        response["query"] = query
    if total > offset + len(page):
        response["next_offset"] = offset + len(page)
        response["hint"] = f"call list_groups(query={query!r}, limit={limit}, offset={offset + len(page)}) for the next page"
    if total == 0:
        response["note"] = (
            "No matching groups." if needle else "No data yet. Use add_episode to start a memory."
        )
    return json.dumps(response, indent=2, default=str)


@mcp.tool()
async def list_cognitive_patterns(query: str = "", limit: int = 20, offset: int = 0) -> str:
    """List distinct cognitive_pattern labels with edge counts.

    Args:
      query:  optional substring filter on pattern name (case-insensitive).
              e.g. query='legal' matches 'legal risk mitigation' and
              'legal compliance focus'. Empty returns all (default).
      limit:  max patterns returned (default 20, max 200).
      offset: skip first N matches for pagination.

    Patterns are sorted by edge_count DESC (most-reinforced first), so
    limit=10 with empty query returns the user's top 10 thinking styles.
    """
    g = _g()
    limit = max(1, min(int(limit or 20), 200))
    offset = max(0, int(offset or 0))
    needle = (query or "").strip().lower()

    cypher = """
    MATCH ()-[r:RELATES_TO]->()
    WHERE r.intent_meta IS NOT NULL
    RETURN r.intent_meta AS m
    """
    async with g.driver.session() as session:
        rows = [r.data() async for r in await session.run(cypher)]
    counts: dict[str, int] = {}
    for row in rows:
        meta = _parse_json(row["m"])
        if meta and (p := meta.get("cognitive_pattern")):
            counts[p] = counts.get(p, 0) + 1

    items = [{"pattern": k, "edge_count": v} for k, v in counts.items()]
    if needle:
        items = [it for it in items if needle in it["pattern"].lower()]
    items.sort(key=lambda it: -it["edge_count"])

    total = len(items)
    page = items[offset : offset + limit]
    response: dict = {"patterns": page, "total": total, "returned": len(page)}
    if query:
        response["query"] = query
    if total > offset + len(page):
        response["next_offset"] = offset + len(page)
        response["hint"] = f"call list_cognitive_patterns(query={query!r}, limit={limit}, offset={offset + len(page)}) for the next page"
    if total == 0:
        response["note"] = (
            "No matching patterns." if needle else "No annotated edges yet."
        )
    return json.dumps(response, indent=2, default=str)


@mcp.tool()
async def edges_by_pattern(pattern: str, limit: int = 25) -> str:
    """Return edges whose cognitive_pattern matches (case-insensitive substring)."""
    g = _g()
    cypher = """
    MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
    WHERE r.intent_meta IS NOT NULL
    RETURN a.name AS a, b.name AS b,
           coalesce(r.fact, '') AS fact,
           r.intent_meta AS m
    """
    async with g.driver.session() as session:
        rows = [r.data() async for r in await session.run(cypher)]
    needle = pattern.lower()
    out = []
    for row in rows:
        meta = _parse_json(row["m"]) or {}
        if needle in (meta.get("cognitive_pattern") or "").lower():
            out.append({"a": row["a"], "b": row["b"], "fact": row["fact"], "intent_meta": meta})
            if len(out) >= limit:
                break
    return json.dumps(out, indent=2, default=str) if out else "No matching edges."


# ---------------------------------------------------------------------------
# NEW tools (Phase 7)
# ---------------------------------------------------------------------------

@mcp.tool()
async def get_node_narrative(entity_name: str) -> str:
    """Return the vLLM-generated narrative for a node by name (case-insensitive
    substring match). The narrative is the entity's perspective: stance,
    open questions, cognitive pattern label. If no narrative has been generated
    yet for this node, returns a placeholder."""
    g = _g()
    cypher = """
    MATCH (n:Entity)
    WHERE toLower(n.name) CONTAINS toLower($q)
    RETURN
        n.uuid AS uuid,
        n.name AS name,
        coalesce(n.summary, '') AS summary,
        n.vllm_narrative AS narrative,
        coalesce(n.vllm_confidence, 0.0) AS confidence,
        coalesce(n.vllm_last_reinforced, 0.0) AS last_reinforced
    LIMIT 5
    """
    async with g.driver.session() as session:
        rows = [r.data() async for r in await session.run(cypher, q=entity_name)]
    if not rows:
        return f"No entity matches '{entity_name}'."
    out = []
    for r in rows:
        narr = _parse_json(r["narrative"])
        out.append({
            "uuid": r["uuid"],
            "name": r["name"],
            "summary": r["summary"],
            "narrative": narr if narr else "(no narrative generated yet — request via re-narration)",
            "confidence_stored": r["confidence"],
            "confidence_decayed": round(effective_confidence(
                r["confidence"], r["last_reinforced"]
            ), 3) if r["last_reinforced"] else 0.0,
            "confidence_label": label_for(effective_confidence(r["confidence"], r["last_reinforced"])) if r["last_reinforced"] else "stale",
        })
    return json.dumps(out, indent=2, default=str)


@mcp.tool()
async def recent_episodes(entity_name: str, n: int = 5) -> str:
    """Return the n most recent episodic events that mention this entity.
    Useful for 'what was I just talking about regarding X'."""
    g = _g()
    cypher = """
    MATCH (e:Episodic)-[:MENTIONS]->(n:Entity)
    WHERE toLower(n.name) CONTAINS toLower($q)
    RETURN
        e.uuid AS uuid,
        coalesce(e.name, '') AS name,
        coalesce(e.content, '') AS content,
        coalesce(e.valid_at, e.created_at) AS ts
    ORDER BY coalesce(e.valid_at, e.created_at) DESC
    LIMIT $n
    """
    async with g.driver.session() as session:
        rows = [r.data() async for r in await session.run(cypher, q=entity_name, n=n)]
    if not rows:
        return f"No episodes mention '{entity_name}'."
    return json.dumps([{
        "uuid": r["uuid"],
        "name": r["name"],
        "content_excerpt": (r["content"] or "")[:600],
        "timestamp": r["ts"],
    } for r in rows], indent=2, default=str)


@mcp.tool()
async def get_episode(uuid: str) -> str:
    """Return the full content of a single episode by uuid. Useful when an edge
    cites an episode and the agent wants the raw context."""
    g = _g()
    cypher = """
    MATCH (e:Episodic {uuid: $uuid})
    RETURN
        e.uuid AS uuid,
        coalesce(e.name, '') AS name,
        coalesce(e.content, '') AS content,
        coalesce(e.source_description, '') AS source_description,
        coalesce(e.group_id, '') AS group_id,
        coalesce(e.valid_at, e.created_at) AS ts
    """
    async with g.driver.session() as session:
        result = await session.run(cypher, uuid=uuid)
        row = await result.single()
    if row is None:
        return f"No episode with uuid {uuid}"
    return json.dumps(dict(row), indent=2, default=str)


_PIPELINE_MODE = os.environ.get("COGRAM_PIPELINE_MODE", "async").lower()  # 'async' | 'sync'


async def _run_pipeline_background(
    g, result, group_id: str, settings, episode_name: str, task_id: str | None = None
) -> None:
    """Background pipeline execution. Errors are logged to stdout + Redis events;
    they never block the MCP response. If a task_id is provided, the task
    registry is updated with the terminal state (done/failed/cancelled) so the
    task-management MCP tools can report status."""
    from cogram.pipeline import tasks as _tasks
    try:
        from cogram.pipeline.post_write import cogram_post_write
        summary = await cogram_post_write(graphiti=g, episode_result=result, group_id=group_id, settings=settings)
        if task_id:
            _tasks.mark_done(task_id, summary)
        # Emit the pipeline-completion event so dashboard/clients can react
        try:
            from cogram.utils import events as _ev
            await _ev.publish("cogram:events:pipeline_done", {
                "task_id": task_id,
                "episode_name": episode_name,
                "group_id": group_id,
                "summary": summary,
            })
        except Exception:
            pass
    except asyncio.CancelledError:
        if task_id:
            _tasks.mark_cancelled(task_id)
        raise
    except Exception as exc:
        if task_id:
            _tasks.mark_failed(task_id, str(exc))
        print(f"[cogram] background pipeline failed for {episode_name}: {exc}")
        try:
            from cogram.utils import events as _ev
            await _ev.publish("cogram:events:pipeline_done", {
                "task_id": task_id,
                "episode_name": episode_name,
                "group_id": group_id,
                "error": str(exc),
            })
        except Exception:
            pass


@mcp.tool()
async def add_episode(content: str, source_description: str = "claude-mcp", group_id: str = "default") -> str:
    """Write a new episode (a piece of text) to the graph. Graphiti will extract
    entities + relationships + intent annotations from it.

    Use this to actively record facts during a conversation:
      - 'Candidate X has 5 years of React experience'
      - 'Director's stance: never approve unjustified deps'
      - any chunk of text Claude wants to commit to long-term memory

    Pipeline modes (controlled by COGRAM_PIPELINE_MODE env var):
      - 'async' (default): graphiti.add_episode runs and returns. The cogram
        post-write pipeline (intent annotation, narration, profile distill,
        Redis cache invalidation) runs as a background task. MCP latency is
        ~2-3s regardless of how many entities the episode produces.
      - 'sync': pipeline runs inline before returning. Latency 15-30s but
        the response includes the full pipeline summary.

    Returns a summary of what was extracted (entity count, edge count). In sync
    mode the response also includes the full pipeline summary; in async mode
    the pipeline result is published to the 'cogram:events:pipeline_done'
    Redis channel."""
    from datetime import datetime, timezone
    from cogram.core.nodes import EpisodeType
    g = _g()
    name = f"mcp_{int(time.time()*1000)}"
    try:
        result = await g.add_episode(
            name=name,
            episode_body=content,
            source=EpisodeType.text,
            source_description=source_description,
            reference_time=datetime.now(timezone.utc),
            previous_episode_uuids=[],
            group_id=group_id,
        )
        nodes = getattr(result, "nodes", [])
        edges = getattr(result, "edges", [])

        response: dict = {
            "ok": True,
            "episode_name": name,
            "extracted": {"entities": len(nodes), "edges": len(edges)},
            "group_id": group_id,
        }

        if os.environ.get("COGRAM_FULL_PIPELINE", "false").lower() == "true":
            if _PIPELINE_MODE == "sync":
                # Inline pipeline (legacy/debug)
                try:
                    from cogram.pipeline.post_write import cogram_post_write
                    response["cogram"] = await cogram_post_write(
                        graphiti=g, episode_result=result, group_id=group_id, settings=_settings(),
                    )
                except Exception as pipe_exc:
                    response["cogram"] = {"error": f"pipeline failed: {pipe_exc}"}
            else:
                # Fire-and-forget: pipeline runs as a background task. We
                # allocate the task_id up front so the coroutine can update
                # its own status via the task registry, then register the
                # asyncio.Task handle so the management tools can cancel it.
                from cogram.pipeline import tasks as _tasks
                task_id = _tasks.new_task_id()
                bg = asyncio.create_task(
                    _run_pipeline_background(g, result, group_id, _settings(), name, task_id)
                )
                _tasks.register(bg, episode_name=name, group_id=group_id, task_id=task_id)
                response["cogram"] = {
                    "mode": "async",
                    "status": "pipeline_running_in_background",
                    "task_id": task_id,
                    "note": (
                        "intent_meta/narrative/profile will populate within ~15s. "
                        "Poll get_add_memory_task_status(task_id) or block on "
                        "wait_for_add_memory_task(task_id). Subscribe to "
                        "cogram:events:pipeline_done for the same signal via Redis."
                    ),
                }

        return json.dumps(response, indent=2)
    except Exception as exc:
        return json.dumps({"ok": False, "error": str(exc)})


@mcp.tool()
async def record_fact(subject: str, predicate: str, object: str, group_id: str = "default") -> str:
    """Convenience wrapper around add_episode for stating a simple subject-predicate-object fact.
    e.g. record_fact('Siva', 'building', 'AI HR platform')."""
    return await add_episode(
        content=f"{subject} {predicate} {object}",
        source_description="claude-fact",
        group_id=group_id,
    )


# ---------------------------------------------------------------------------
# Async pipeline task management
#
# When add_episode runs in async mode (the default), the cogram post-write
# pipeline (annotation / narration / profile / knot synthesis) runs in the
# background and the task_id is returned in the add_episode response.
# These four tools let an MCP client introspect and control those tasks
# without subscribing to the Redis events channel.
# ---------------------------------------------------------------------------

@mcp.tool()
async def list_add_memory_tasks(
    group_id: str = "",
    state: str = "",
    limit: int = 50,
) -> str:
    """List in-flight and recently completed background pipeline tasks spawned
    by add_episode (async mode). Newest first.

    Filters:
      group_id  — restrict to one group; empty = all groups
      state     — 'running' | 'done' | 'failed' | 'cancelled' | '' (all)
      limit     — max records to return (default 50)

    The registry holds the most recent ~200 records process-wide; older finished
    tasks are evicted in FIFO order. Restarting the MCP server clears it."""
    from cogram.pipeline import tasks as _tasks
    items = _tasks.list_tasks(
        group_id=group_id or None,
        state=state or None,
        limit=limit,
    )
    return json.dumps(
        {
            "count": len(items),
            "tasks": [r.to_dict() for r in items],
        },
        indent=2,
        default=str,
    )


@mcp.tool()
async def get_add_memory_task_status(task_id: str) -> str:
    """Return the current state + summary of a background pipeline task.

    States:
      running   — pipeline still working
      done      — finished; full summary available (edges_annotated,
                  nodes_narrated, profile_distilled, knot_synthesis, ...)
      failed    — exception; see `error` field
      cancelled — cancelled via cancel_add_memory_task

    Returns ok=False if the task_id is unknown (registry was cleared, or
    record was evicted past COGRAM_TASK_HISTORY)."""
    from cogram.pipeline import tasks as _tasks
    rec = _tasks.get(task_id)
    if rec is None:
        return json.dumps(
            {"ok": False, "error": f"unknown task_id {task_id!r}"},
            indent=2,
        )
    return json.dumps({"ok": True, **rec.to_dict()}, indent=2, default=str)


@mcp.tool()
async def wait_for_add_memory_task(task_id: str, timeout_seconds: float = 30.0) -> str:
    """Block until the named pipeline task finishes (done / failed / cancelled),
    or `timeout_seconds` elapses. Returns the same shape as
    get_add_memory_task_status; the `state` field tells you whether the task
    finished or the wait timed out.

    Typical use: an agent calls add_episode and then immediately asks the graph
    a question that depends on the post-write annotations. Block on the task_id
    first to avoid a stale read."""
    from cogram.pipeline import tasks as _tasks
    try:
        rec = await _tasks.wait(task_id, timeout=timeout_seconds)
    except KeyError:
        return json.dumps(
            {"ok": False, "error": f"unknown task_id {task_id!r}"},
            indent=2,
        )
    return json.dumps({"ok": True, **rec.to_dict()}, indent=2, default=str)


@mcp.tool()
async def cancel_add_memory_task(task_id: str) -> str:
    """Request cancellation of a still-running pipeline task. Returns
    ok=True if a running task was cancelled, ok=False if the task was
    already terminal or unknown.

    Cancellation is cooperative: the task hits its next `await` and aborts
    with asyncio.CancelledError. Any annotations / narrations already
    written to Neo4j stay; only future steps are skipped."""
    from cogram.pipeline import tasks as _tasks
    cancelled = _tasks.cancel(task_id)
    return json.dumps(
        {
            "ok": cancelled,
            "task_id": task_id,
            "note": (
                "Task cancellation requested." if cancelled
                else "No running task with that id (already terminal or unknown)."
            ),
        },
        indent=2,
    )


@mcp.tool()
async def get_unified_profile(top_patterns: int = 15, examples_per_pattern: int = 1) -> str:
    """Unified Director profile across ALL group_ids in the graph. Merges every
    :DirectorProfile into one synthesized view: combined working-style summaries,
    deduplicated visions, and cognitive patterns aggregated across groups
    (confidence/count summed when the same pattern name appears in multiple groups).

    Use this when the user has multiple group_ids (e.g. 'hr_platform',
    'default', 'engram') and you want a single 'who is this person across all
    their work' answer.

    Single Cypher round-trip (no per-group fan-out)."""
    cache_key = f"cogram:cache:profile:UNIFIED:top{top_patterns}:ex{examples_per_pattern}"
    try:
        import redis.asyncio as _r
        _redis = _r.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True)
        cached = await _redis.get(cache_key)
        if cached:
            await _redis.aclose()
            return cached
    except Exception:
        _redis = None

    g = _g()
    cypher = """
    // 1. All profiles
    MATCH (p:DirectorProfile)
    WITH collect({group_id: p.group_id, summary: p.summary, visions: coalesce(p.visions, [])}) AS profiles

    // 2. All patterns (sum confidence/count when same name across groups)
    MATCH (pat:CognitivePattern)
    WITH profiles, pat.name AS name, sum(coalesce(pat.confidence, 0)) AS total_conf,
         sum(coalesce(pat.count, 0)) AS total_count, collect(DISTINCT pat.group_id) AS groups
    ORDER BY total_conf DESC, total_count DESC
    LIMIT $top_patterns
    WITH profiles, collect({name: name, confidence: total_conf, count: total_count, groups: groups}) AS top_pats

    // 3. For each top pattern, sample reinforcing edges
    UNWIND CASE WHEN size(top_pats) = 0 THEN [null] ELSE top_pats END AS pat
    OPTIONAL MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
        WHERE pat IS NOT NULL
          AND r.intent_meta IS NOT NULL
          AND r.retracted_at IS NULL
          AND r.cognitive_pattern_name = pat.name
    WITH profiles, pat,
         collect({a: a.name, b: b.name, fact: coalesce(r.fact, ''), intent_meta: r.intent_meta})[0..$cap] AS examples
    WITH profiles,
         collect(CASE WHEN pat IS NULL THEN null ELSE {
             name: pat.name,
             confidence: pat.confidence,
             count: pat.count,
             groups: pat.groups,
             examples: examples
         } END) AS patterns

    RETURN profiles, [pat IN patterns WHERE pat IS NOT NULL] AS patterns
    """

    async with g.driver.session() as session:
        rec = await (await session.run(cypher, top_patterns=top_patterns, cap=examples_per_pattern)).single()

    if rec is None or not rec["profiles"]:
        return json.dumps({
            "error": "No DirectorProfile nodes anywhere in the graph yet. Ingest some episodes first.",
        }, indent=2)

    # Merge per-group summaries into one combined view
    summaries = [p["summary"] for p in rec["profiles"] if p.get("summary")]
    all_visions: list[str] = []
    seen_visions: set[str] = set()
    for p in rec["profiles"]:
        for v in (p.get("visions") or []):
            key = (v or "").strip().lower()[:140]
            if key and key not in seen_visions:
                seen_visions.add(key)
                all_visions.append(v)

    patterns_out = []
    for pat in rec["patterns"] or []:
        examples_list = []
        for ex in (pat.get("examples") or []):
            if not ex.get("a"):
                continue
            meta = _parse_json(ex.get("intent_meta")) or {}
            examples_list.append({
                "between": f"{ex['a']} → {ex['b']}",
                "fact": (ex.get("fact") or "")[:200],
                "why_connected": meta.get("why_connected", ""),
                "director_vision": meta.get("director_vision", ""),
            })
        patterns_out.append({
            "name": pat.get("name"),
            "confidence": pat.get("confidence"),
            "count": pat.get("count"),
            "appears_in_groups": pat.get("groups") or [],
            "examples": examples_list,
        })

    body = {
        "groups_merged": [p.get("group_id") for p in rec["profiles"]],
        "combined_working_style_summaries": summaries,
        "recurring_visions_deduped": all_visions,
        "cognitive_patterns": patterns_out,
        "pagination": {"returned": len(patterns_out), "top_patterns": top_patterns},
    }
    payload = json.dumps(body, indent=2)

    if _redis is not None:
        try:
            await _redis.set(cache_key, payload, ex=_PROFILE_CACHE_TTL_SECONDS)
            await _redis.aclose()
        except Exception:
            pass

    return payload


@mcp.tool()
async def dedup_patterns(group_id: str = "default", threshold: float = 0.88) -> str:
    """Collapse near-duplicate cognitive_pattern names within a group.

    The annotator emits slight variations ('integration-focused development',
    'systematic integration', 'modular integration approach') that are really
    one underlying idea. This tool embeds every pattern name and merges any
    two whose cosine similarity is above `threshold` (default 0.88), folding
    confidence + count + REINFORCED_BY edges + RELATES_TO.cognitive_pattern_name
    pointers into the winner. Idempotent.

    Returns a summary of merges performed."""
    g = _g()
    try:
        from cogram.utils.maintenance.pattern_dedup import dedup_patterns as _dedup
        result = await _dedup(g, group_id, _settings(), threshold=threshold)
        return json.dumps(result, indent=2, default=str)
    except Exception as exc:
        return json.dumps({"ok": False, "error": str(exc)}, indent=2)


@mcp.tool()
async def retract(target: str, reason: str = "") -> str:
    """Mark a fact in the graph as retracted (i.e. wrong / no longer believed).

    `target` is either:
      - an Episodic uuid (retracts every edge that came from that episode)
      - an Entity name (retracts every RELATES_TO edge touching that entity)
      - an edge uuid (retracts only that edge)

    Retracted edges are kept on disk for audit, but are filtered out of
    `search_graph` results. Use this when the user says 'that's wrong',
    'I never said that', or corrects a specific fact.
    """
    g = _g()
    now = time.time()
    cypher_by_episode = """
    MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
    WHERE $t IN coalesce(r.episodes, [])
    SET r.retracted_at = $now, r.retraction_reason = $reason
    RETURN count(r) AS n
    """
    cypher_by_entity = """
    MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
    WHERE toLower(a.name) = toLower($t) OR toLower(b.name) = toLower($t)
    SET r.retracted_at = $now, r.retraction_reason = $reason
    RETURN count(r) AS n
    """
    cypher_by_edge = """
    MATCH ()-[r:RELATES_TO]->()
    WHERE r.uuid = $t
    SET r.retracted_at = $now, r.retraction_reason = $reason
    RETURN count(r) AS n
    """
    counts: dict[str, int] = {}
    affected_patterns: list[str] = []
    affected_groups: set[str] = set()
    async with g.driver.session() as session:
        for label, cypher in [
            ("by_episode", cypher_by_episode),
            ("by_entity", cypher_by_entity),
            ("by_edge_uuid", cypher_by_edge),
        ]:
            rec = await (await session.run(cypher, t=target, now=now, reason=reason)).single()
            counts[label] = (rec["n"] if rec else 0)

        # Identify the patterns that were reinforced by edges we just retracted,
        # and decrement their confidence by the count of newly retracted edges.
        impact = await (await session.run(
            """
            MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
            WHERE r.retracted_at = $now
            WITH r.cognitive_pattern_name AS pname,
                 coalesce(a.group_id, b.group_id, 'default') AS gid,
                 count(*) AS retracted_count
            WHERE pname IS NOT NULL
            MATCH (pat:CognitivePattern {name: pname, group_id: gid})
            SET pat.confidence = CASE
                    WHEN pat.confidence - retracted_count <= 0 THEN 0
                    ELSE pat.confidence - retracted_count
                END,
                pat.last_retracted_at = $now
            RETURN collect(DISTINCT pname) AS patterns,
                   collect(DISTINCT gid) AS groups
            """,
            now=now,
        )).single()
        if impact:
            affected_patterns = list(impact["patterns"] or [])
            affected_groups = set(impact["groups"] or [])

    total = sum(counts.values())
    if total == 0:
        return json.dumps({
            "ok": False,
            "error": f"No edges matched target {target!r}. Pass an episode uuid, entity name, or edge uuid.",
        }, indent=2)

    # Invalidate caches: Redis active_memory subgraphs + Redis profile JSON cache
    try:
        import redis.asyncio as _r
        client = _r.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True)
        async for key in client.scan_iter(match="cogram:session:*:active"):
            await client.delete(key)
        async for key in client.scan_iter(match="cogram:cache:profile:*"):
            await client.delete(key)
        await client.aclose()
    except Exception:
        pass

    # Emit events for downstream cache invalidation
    try:
        from cogram.utils import events as _ev
        await _ev.publish(_ev.GRAPH_CHANGE, {
            "action": "retract",
            "target": target,
            "edges": total,
            "affected_patterns": affected_patterns,
            "affected_groups": list(affected_groups),
        })
        if affected_patterns:
            await _ev.publish(_ev.PROFILE_CHANGE, {
                "action": "retract_propagated",
                "patterns": affected_patterns,
                "groups": list(affected_groups),
            })
    except Exception:
        pass

    return json.dumps({
        "ok": True,
        "target": target,
        "reason": reason,
        "retracted_edges": total,
        "match_breakdown": counts,
        "patterns_decremented": affected_patterns,
        "groups_affected": list(affected_groups),
        "note": "Retracted edges are kept on disk for audit but filtered from search_graph and profile distillation. Pattern confidence has been decremented for any patterns the retracted edges reinforced.",
    }, indent=2)


@mcp.tool()
async def confidence(entity_name: str) -> str:
    """Return the current effective confidence (decayed) and human-readable label
    for a node. Useful for 'how deeply rooted is the Director's belief about X'."""
    g = _g()
    cypher = """
    MATCH (n:Entity)
    WHERE toLower(n.name) CONTAINS toLower($q)
    RETURN
        n.uuid AS uuid,
        n.name AS name,
        coalesce(n.vllm_confidence, 0.0) AS conf,
        coalesce(n.vllm_last_reinforced, 0.0) AS ts
    LIMIT 5
    """
    async with g.driver.session() as session:
        rows = [r.data() async for r in await session.run(cypher, q=entity_name)]
    if not rows:
        return f"No entity matches '{entity_name}'."
    out = []
    for r in rows:
        decayed = effective_confidence(r["conf"], r["ts"]) if r["ts"] else 0.0
        out.append({
            "name": r["name"],
            "stored_confidence": round(r["conf"], 2),
            "effective_confidence": round(decayed, 3),
            "normalized_0_1": round(normalized_confidence(r["conf"], r["ts"]), 3) if r["ts"] else 0.0,
            "label": label_for(decayed),
            "last_reinforced_seconds_ago": round(time.time() - r["ts"]) if r["ts"] else None,
        })
    return json.dumps(out, indent=2, default=str)


# ---------------------------------------------------------------------------
# Health endpoint (the dashboard pings this)
# ---------------------------------------------------------------------------

# The FastMCP HTTP transport already provides /mcp; we add /health beside it.
# When running in HTTP mode, the underlying ASGI app is reachable.

def main() -> None:
    parser = argparse.ArgumentParser(description="cogram MCP server")
    parser.add_argument("--http", action="store_true", help="run HTTP/SSE transport")
    parser.add_argument("--host", default=os.environ.get("MCP_HTTP_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("MCP_HTTP_PORT", "7800")))
    args = parser.parse_args()

    if not args.http:
        mcp.run()
        return

    # ------------------------------------------------------------------
    # HTTP transport. FastMCP's streamable_http_app() already serves the
    # MCP protocol at /mcp/ internally — so we serve THAT app directly at
    # root. Mounting it under another /mcp prefix would create /mcp/mcp/.
    # We add /health and /info as siblings using a Starlette wrapper.
    # ------------------------------------------------------------------
    import uvicorn
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Mount, Route

    # Pick the best transport available. Streamable HTTP is what Claude Desktop
    # custom connectors use as of 2025+. SSE is older but more widely supported.
    inner_app = None
    transport_used = "none"
    for fn_name in ("streamable_http_app", "sse_app"):
        fn = getattr(mcp, fn_name, None)
        if callable(fn):
            try:
                inner_app = fn()
                transport_used = fn_name.replace("_app", "")
                break
            except Exception as exc:
                print(f"[mcp] {fn_name}() failed: {exc}")

    if inner_app is None:
        raise RuntimeError("No HTTP transport available on FastMCP instance")

    print(f"[mcp] serving '{SERVER_NAME}' via {transport_used} on :{args.port}")

    async def health(_request):
        return JSONResponse({
            "ok": True,
            "server": SERVER_NAME,
            "transport": transport_used,
            "profile_injection": PROFILE_INJECTION,
        })

    async def info(_request):
        return JSONResponse({
            "service": "cogram-mcp",
            "server_name": SERVER_NAME,
            "transport": transport_used,
            "endpoint": "/mcp/" if transport_used == "streamable_http" else "/sse",
            "profile_injection": PROFILE_INJECTION,
        })

    # Trailing-slash compatibility: FastMCP's streamable_http_app serves at
    # `/mcp` (no slash). Older clients configured `/mcp/` need to keep working,
    # so we intercept the trailing-slash variant and forward to the canonical
    # path WITHOUT a 307 redirect (which some MCP clients fail to follow with
    # POST body intact).
    async def mcp_trailing_slash_proxy(request):
        # Reroute /mcp/ → forward inside the inner_app at /mcp
        scope = dict(request.scope)
        scope["path"] = "/mcp"
        scope["raw_path"] = b"/mcp"
        return await inner_app(scope, request.receive, request._send)  # type: ignore[arg-type]

    routes = [
        Route("/health", endpoint=health, methods=["GET"]),
        Route("/info", endpoint=info, methods=["GET"]),
        Route("/mcp/", endpoint=mcp_trailing_slash_proxy, methods=["GET", "POST", "DELETE"]),
        # Everything else falls through to FastMCP (owns /mcp, /sse, /messages, etc.)
        Mount("/", app=inner_app),
    ]

    # CRITICAL: FastMCP's session_manager.run() must be entered as part of the
    # outer ASGI lifespan, otherwise the task group isn't initialized and POSTs
    # to /mcp/ fail with "Task group is not initialized" 500 errors.
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app):
        sm = getattr(mcp, "session_manager", None)
        if sm is not None and hasattr(sm, "run"):
            async with sm.run():
                yield
        else:
            yield

    wrapper = Starlette(routes=routes, lifespan=lifespan)

    uvicorn.run(wrapper, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

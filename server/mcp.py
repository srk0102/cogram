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
from cogram.utils.confidence import effective_confidence, label_for

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


# NOTE: `get_knot` was removed from the MCP surface in v0.2.1.
# Until knot synthesis runs reliably (the 120s pipeline timeout currently
# squeezes it on bigger writes), agents that called get_knot first to "see
# what's in there" got an empty hint back ~95% of the time and had to retry
# with get_node_narrative. The underlying machinery still lives in
# cogram.utils.maintenance.knot_synthesis (and is used by the post-write
# pipeline). It will be re-exposed once synthesis is reliable.


@mcp.tool()
async def get_director_profile(
    group_id: str = "default",
    top_patterns: int = 10,
    examples_per_pattern: int = 2,
) -> str:
    """Compressed Director profile for ONE group: working-style summary + recurring visions + top cognitive patterns with reinforcing-edge examples.
    When: after list_groups picks a group_id and you want the user's reasoning model for it.
    Inputs: group_id (required — get from list_groups), top_patterns, examples_per_pattern.
    Pair with: edges_by_pattern(name) to drill into any pattern; for cross-group merge use get_unified_profile.
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
    """Semantic search across all edges. Profile-aware on subjective queries ('would I...', 'how do I feel...'). Hot-tier (Redis) accelerated — <1ms after first warm. Filters retracted edges automatically.
    When: fuzzy concept lookup or 'find anything related to X'.
    Inputs: query (any text), group_id (required — list_groups first), limit.
    Pair with: get_entity_view(name, mode='edges') when you have a known entity name. Pair with get_director_profile when query is subjective.
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


# NOTE: `find_connections` was merged into `get_entity_view(name, mode="edges")` in v0.2.1.
# Kept its query body inline below as part of the merged tool.


@mcp.tool()
async def list_groups(query: str = "", limit: int = 20, offset: int = 0) -> str:
    """**First call of any session.** Returns each group_id with episode/entity/pattern/knot counts so you know what contexts exist before searching. Paginated.
    When: at session start, or before ANY group-scoped tool call (search_graph, get_director_profile, edges_by_pattern).
    Inputs: query (substring filter, case-insensitive), limit (max 200), offset.
    Pair with: get_director_profile(group_id=X) once you've picked a group.
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
    """Discover distinct cognitive_pattern labels with edge counts. Sorted most-reinforced-first.
    When: agent wants to know what thinking styles exist before drilling into one.
    Inputs: query (substring filter, e.g. 'legal'), limit (max 200), offset.
    Pair with: edges_by_pattern(pattern_name) for the actual evidence behind each pattern.
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
    """**Cross-decision routing primitive.** Every edge whose cognitive_pattern matches, with intent_meta + director_vision intact. Substring match.
    When: about to make a constrained decision — query the pattern name to surface every prior decision the user made under the same thinking style. Turns 'user rejected X once' into 'user has a recurring rule'.
    Inputs: pattern (substring — see list_cognitive_patterns for exact names), limit.
    Pair with: list_cognitive_patterns first to discover names.
    """
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

_ENTITY_NARRATIVE_CYPHER = """
MATCH (n:Entity)
WHERE toLower(n.name) CONTAINS toLower($q)
OPTIONAL MATCH (n)-[r:RELATES_TO]-()
WITH n, count(r) AS degree
ORDER BY degree DESC
RETURN
    n.uuid AS uuid,
    n.name AS name,
    coalesce(n.summary, '') AS summary,
    n.vllm_narrative AS narrative,
    coalesce(n.vllm_confidence, 0.0) AS confidence,
    coalesce(n.vllm_last_reinforced, 0.0) AS last_reinforced,
    degree
LIMIT 5
"""

_ENTITY_EDGES_CYPHER = """
MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
WHERE toLower(a.name) CONTAINS toLower($q) OR toLower(b.name) CONTAINS toLower($q)
RETURN a.name AS a, b.name AS b,
       coalesce(r.fact, r.name, '') AS fact,
       r.intent_meta AS intent_meta
LIMIT $limit
"""

_ENTITY_EPISODES_CYPHER = """
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


async def _entity_narrative_view(g, entity_name: str) -> list[dict]:
    async with g.driver.session() as session:
        rows = [r.data() async for r in await session.run(_ENTITY_NARRATIVE_CYPHER, q=entity_name)]
    out = []
    for r in rows:
        narr = _parse_json(r["narrative"])
        out.append({
            "uuid": r["uuid"],
            "name": r["name"],
            "degree": r.get("degree", 0),
            "summary": r["summary"],
            "narrative": narr if narr else "(no narrative generated yet — request via re-narration)",
            "confidence_stored": r["confidence"],
            "confidence_decayed": round(effective_confidence(
                r["confidence"], r["last_reinforced"]
            ), 3) if r["last_reinforced"] else 0.0,
            "confidence_label": label_for(effective_confidence(r["confidence"], r["last_reinforced"])) if r["last_reinforced"] else "stale",
        })
    return out


async def _entity_edges_view(g, entity_name: str, limit: int) -> list[dict]:
    async with g.driver.session() as session:
        rows = [r.data() async for r in await session.run(_ENTITY_EDGES_CYPHER, q=entity_name, limit=limit)]
    return [
        {
            "a": r["a"], "b": r["b"], "fact": r["fact"],
            "intent_meta": _parse_json(r.get("intent_meta")),
        }
        for r in rows
    ]


async def _entity_episodes_view(g, entity_name: str, n: int) -> list[dict]:
    async with g.driver.session() as session:
        rows = [r.data() async for r in await session.run(_ENTITY_EPISODES_CYPHER, q=entity_name, n=n)]
    return [
        {
            "uuid": r["uuid"],
            "name": r["name"],
            "content_excerpt": (r["content"] or "")[:600],
            "timestamp": r["ts"],
        }
        for r in rows
    ]


@mcp.tool()
async def get_entity_view(
    entity_name: str,
    mode: str = "narrative",
    limit: int = 25,
) -> str:
    """**Primary 'tell me about X' tool.** Three layers of view selectable via `mode`. Substring match on entity_name (up to 5 entities for narrative; up to `limit` edges/episodes).
    When: any time you have a known entity name and want context — narrative for compressed view, edges for raw provenance, episodes to find uuids for retract, all for everything at once.
    Inputs: entity_name (substring); mode ∈ 'narrative' (default) | 'edges' | 'episodes' | 'all'; limit (caps edges + episodes; narrative is always max 5).
    Pair with: edges_by_pattern for cross-decision routing; search_graph for fuzzy semantic lookup; retract(target=uuid) using uuids from mode='episodes'.
    """
    g = _g()
    mode_l = (mode or "narrative").strip().lower()
    if mode_l not in {"narrative", "edges", "episodes", "all"}:
        return json.dumps({
            "ok": False,
            "error": f"unknown mode {mode!r}; expected narrative|edges|episodes|all",
        }, indent=2)

    n_episodes = max(1, min(int(limit or 25), 50))

    if mode_l == "narrative":
        rows = await _entity_narrative_view(g, entity_name)
        if not rows:
            return f"No entity matches '{entity_name}'."
        return json.dumps(rows, indent=2, default=str)

    if mode_l == "edges":
        rows = await _entity_edges_view(g, entity_name, limit=limit)
        if not rows:
            return f"No edges connected to '{entity_name}'."
        return json.dumps(rows, indent=2, default=str)

    if mode_l == "episodes":
        rows = await _entity_episodes_view(g, entity_name, n=n_episodes)
        if not rows:
            return f"No episodes mention '{entity_name}'."
        return json.dumps(rows, indent=2, default=str)

    # mode == "all"
    narrative = await _entity_narrative_view(g, entity_name)
    edges = await _entity_edges_view(g, entity_name, limit=limit)
    episodes = await _entity_episodes_view(g, entity_name, n=n_episodes)
    return json.dumps({
        "entity_name": entity_name,
        "narrative": narrative,
        "edges": edges,
        "episodes": episodes,
    }, indent=2, default=str)


# NOTE: `get_node_narrative` and `recent_episodes` were merged into
# `get_entity_view(name, mode=...)` in v0.2.1. Use mode='narrative' (default)
# for narratives and mode='episodes' for episode excerpts.


@mcp.tool()
async def get_episode(uuid: str) -> str:
    """Full content of one episode by uuid.
    When: drilling into episode bodies after get_entity_view(name, mode='episodes') returns a uuid.
    Inputs: uuid (the EPISODIC uuid — NOT the friendly `episode_name` like 'mcp_<ts>' that add_episode returns).
    Pair with: get_entity_view(name, mode='episodes') to find uuids; retract(target=uuid) to roll back facts from this episode.
    """
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
    """Write text to the graph — graphiti extracts entities/edges, cogram annotates async. Returns task_id (~3s); pipeline runs ~15s in background.
    When: user states a new fact, decision, or context worth remembering.
    Inputs: content (any text), group_id (use list_groups first to pick), source_description.
    Pair with: get_episode_task(task_id, wait_seconds=20) BEFORE reading the graph to avoid stale reads. Use record_fact for clean SPO sentences. Returns `episode_name` (friendly slug) and `cogram.task_id` — for retract use `get_episode` to fetch the EPISODIC uuid, not episode_name.
    """
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

        # Surface the silent-failure pattern: entities extracted but no edges.
        # Without this, agents see ok=true and treat the write as successful,
        # only to find search_graph returns nothing later. Common causes are
        # snake_case predicates, pronoun-only subjects, or single-noun fragments.
        if len(nodes) > 0 and len(edges) == 0:
            response["warning"] = (
                "Graphiti extracted entities but no edges from this content, so "
                "the intent annotation pipeline has nothing to annotate. Common "
                "causes: snake_case predicates ('name_is' → 'name is'), pronouns "
                "instead of named subjects ('I am' → 'Kartik is'), or terse "
                "fragments. See docs/agent_playbook.md format gotchas for working "
                "patterns. The episode is stored as raw text and reachable via "
                "get_entity_view(name, mode='episodes')."
            )

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
                        "Peek with get_episode_task(task_id) or block with "
                        "get_episode_task(task_id, wait_seconds=20). Subscribe to "
                        "cogram:events:pipeline_done for the same signal via Redis."
                    ),
                }

        return json.dumps(response, indent=2)
    except Exception as exc:
        return json.dumps({"ok": False, "error": str(exc)})


@mcp.tool()
async def record_fact(subject: str, predicate: str, object: str, group_id: str = "default") -> str:
    """SPO wrapper around add_episode. Constructs '{subject} {predicate} {object}' and writes — locks edge structure better than free prose.
    When: user states a structural fact ('Siva builds cogram', 'cogram uses Neo4j'). Use add_episode for narrative episodes.
    Inputs: subject, predicate, object, group_id. Predicate underscores are auto-converted to spaces (`name_is` → `name is`) so snake_case identifiers form readable English the extractor can produce edges from.
    Pair with: get_episode_task(task_id, wait_seconds=20) — same async pipeline as add_episode.
    """
    # Auto-rephrase: replace underscores in the predicate with spaces. Otherwise
    # snake_case predicates (`name_is`, `has_experience_in`) form non-English
    # sentences that graphiti's extractor silently fails to produce edges from.
    natural_predicate = predicate.replace("_", " ").strip()
    return await add_episode(
        content=f"{subject} {natural_predicate} {object}",
        source_description="claude-fact",
        group_id=group_id,
    )


# ---------------------------------------------------------------------------
# Async pipeline task management
#
# When add_episode runs in async mode (the default), the cogram post-write
# pipeline (annotation / narration / profile / knot synthesis) runs in the
# background and the task_id is returned in the add_episode response.
# These three tools let an MCP client introspect and control those tasks
# without subscribing to the Redis events channel. The naming follows
# cogram's own vocabulary (`episode`, not `memory`) so the tool surface is
# consistent with add_episode.
# ---------------------------------------------------------------------------

@mcp.tool()
async def list_episode_tasks(
    group_id: str = "",
    state: str = "",
    limit: int = 50,
) -> str:
    """List in-flight + recent background pipeline tasks (newest first). Process-local — keeps last ~200; cleared on server restart.
    When: introspecting whether a write's annotations have settled, or finding a task_id you lost.
    Inputs: group_id (filter), state ('running'|'done'|'failed'|'cancelled'|''), limit.
    Pair with: get_episode_task(task_id, wait_seconds=N) to wait/peek; cancel_episode_task to abort.
    """
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
async def get_episode_task(task_id: str, wait_seconds: float = 0.0) -> str:
    """Inspect or block on a background pipeline task. wait_seconds=0 peeks; wait_seconds=N blocks up to N seconds for terminal state. State ∈ running|done|failed|cancelled.
    When: agent just called add_episode/record_fact and needs annotations settled before reading the graph (otherwise stale reads).
    Inputs: task_id (returned by add_episode/record_fact in `cogram.task_id`), wait_seconds (default 0).
    Pair with: cancel_episode_task to abort runaway tasks; list_episode_tasks to find lost task_ids.
    """
    from cogram.pipeline import tasks as _tasks

    if wait_seconds and wait_seconds > 0:
        try:
            rec = await _tasks.wait(task_id, timeout=wait_seconds)
        except KeyError:
            return json.dumps(
                {"ok": False, "error": f"unknown task_id {task_id!r}"},
                indent=2,
            )
    else:
        rec = _tasks.get(task_id)
        if rec is None:
            return json.dumps(
                {"ok": False, "error": f"unknown task_id {task_id!r}"},
                indent=2,
            )

    return json.dumps({"ok": True, **rec.to_dict()}, indent=2, default=str)


@mcp.tool()
async def cancel_episode_task(task_id: str) -> str:
    """Cooperatively cancel a still-running pipeline task. Already-written annotations stay; only future steps are skipped.
    When: a task is stuck, took the wrong direction, or rate-limit chains will block other writes.
    Inputs: task_id.
    Pair with: list_episode_tasks(state='running') to find live task_ids first.
    """
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
    """Director profile MERGED across every group_id with `appears_in_groups` per pattern. Single Cypher round-trip.
    When: user has multiple groups and you want one 'who is this person across all their work' answer.
    Inputs: top_patterns, examples_per_pattern. No group_id (merges all).
    Pair with: get_director_profile(group_id=X) when you need one group in detail; list_groups to discover the groups.
    """
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


# NOTE: `dedup_patterns` was removed from the MCP surface in v0.2.1.
# Agents shouldn't trigger pattern dedup on a live graph during a chat — it
# fires bulk embedding calls and merges names across in-flight conversations.
# The function stays callable as a CLI / cron tool via:
#   docker exec cogram-mcp python -m cogram.utils.maintenance.pattern_dedup
# (or import it directly from cogram.utils.maintenance.pattern_dedup).


@mcp.tool()
async def retract(target: str, reason: str = "") -> str:
    """Mark fact(s) as no-longer-believed. Filtered from search_graph; kept on disk for audit; bumps cache version so narratives re-generate on next read.
    When: user says 'that's wrong', 'I never said that', or you find an annotator hallucination.
    Inputs: target accepts (a) Entity name, (b) edge uuid, (c) Episodic uuid — NOT the friendly `episode_name` 'mcp_<ts>' from add_episode (use get_entity_view(name, mode='episodes') or get_episode first to find the uuid).
    Pair with: precede with get_entity_view(name, mode='edges') to verify what you'll retract.
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

    # Find unique entity uuids touched by the retracted edges so we can bust
    # their narrative cache versions. Without this, get_node_narrative serves
    # the pre-retraction prose forever (Engram's prompt-hash cache key is
    # stable across the retraction).
    affected_entity_uuids: list[str] = []
    try:
        async with g.driver.session() as session:
            rows = [
                r.data()
                async for r in await session.run(
                    """
                    MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
                    WHERE r.retracted_at = $now
                    RETURN collect(DISTINCT a.uuid) + collect(DISTINCT b.uuid) AS uuids
                    """,
                    now=now,
                )
            ]
            if rows:
                seen: set[str] = set()
                for u in (rows[0].get("uuids") or []):
                    if u and u not in seen:
                        seen.add(u)
                        affected_entity_uuids.append(u)
    except Exception:
        pass

    # Invalidate caches:
    #   - Redis active_memory subgraphs (search_graph hot tier)
    #   - Redis profile JSON cache (5-min TTL)
    #   - Per-entity narrative cache versions (incremented; node_narration's
    #     cache_key_fn includes this so the next narrate call misses Engram
    #     and re-runs the LLM with the post-retraction summary)
    try:
        import redis.asyncio as _r
        client = _r.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True)
        async for key in client.scan_iter(match="cogram:session:*:active"):
            await client.delete(key)
        async for key in client.scan_iter(match="cogram:cache:profile:*"):
            await client.delete(key)
        for uuid in affected_entity_uuids:
            await client.incr(f"cogram:cache_version:entity:{uuid}")
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


# NOTE: `confidence` was removed from the MCP surface in v0.2.1.
# Decay-weighted confidence numbers are infrastructure that leaked into the
# agent surface — agents almost never need raw scores; they need "is this
# still believed", and search_graph already filters retracted edges.
# The decay math stays in cogram.utils.confidence for internal use
# (profile distillation, cache TTLs, etc.).


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

"""COGRAM ACTIVE MEMORY — Redis hot tier for the working subgraph.

Pattern: when a session starts, pull a relevant subset of nodes/edges from
Neo4j into Redis. Within the session, all reads hit Redis (<1ms). When the
topic drifts, expand or flush+repull. At session end, persist any session-level
state back to Neo4j.

Redis keys:
  cogram:session:{session_id}:active   — JSON {entities, edges, centroid, hit_count, miss_count}
  cogram:session:{session_id}:meta     — JSON {pulled_at, last_drift_check, pull_filter}

Each entity in the cache:
  {uuid, name, summary, name_embedding, group_id, last_touched_at}

Each edge in the cache:
  {uuid, source_uuid, target_uuid, fact, intent_meta, group_id, last_touched_at}
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
SESSION_TTL_SECONDS = int(os.environ.get("ACTIVE_MEMORY_TTL", "3600"))     # 1h hard ceiling
SUBGRAPH_PULL_LIMIT = int(os.environ.get("ACTIVE_MEMORY_PULL_LIMIT", "60"))  # nodes per pull
EXPAND_PULL_LIMIT = int(os.environ.get("ACTIVE_MEMORY_EXPAND_LIMIT", "20"))


_redis_client = None


async def _get_redis():
    """Lazy redis-py asyncio client."""
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    try:
        import redis.asyncio as redis_asyncio  # type: ignore
    except ImportError:
        return None
    _redis_client = redis_asyncio.from_url(REDIS_URL, decode_responses=True)
    return _redis_client


# ---------------------------------------------------------------------------
# Subgraph data shape
# ---------------------------------------------------------------------------

@dataclass
class CachedEntity:
    uuid: str
    name: str
    summary: str
    name_embedding: list[float]
    group_id: str
    last_touched_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "summary": self.summary,
            "name_embedding": self.name_embedding,
            "group_id": self.group_id,
            "last_touched_at": self.last_touched_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CachedEntity":
        return cls(
            uuid=d["uuid"],
            name=d["name"],
            summary=d.get("summary", ""),
            name_embedding=d.get("name_embedding", []),
            group_id=d.get("group_id", ""),
            last_touched_at=d.get("last_touched_at", time.time()),
        )


@dataclass
class CachedEdge:
    uuid: str
    source_uuid: str
    target_uuid: str
    fact: str
    intent_meta: Optional[dict]
    group_id: str
    last_touched_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "uuid": self.uuid,
            "source_uuid": self.source_uuid,
            "target_uuid": self.target_uuid,
            "fact": self.fact,
            "intent_meta": self.intent_meta,
            "group_id": self.group_id,
            "last_touched_at": self.last_touched_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CachedEdge":
        return cls(
            uuid=d["uuid"],
            source_uuid=d["source_uuid"],
            target_uuid=d["target_uuid"],
            fact=d.get("fact", ""),
            intent_meta=d.get("intent_meta"),
            group_id=d.get("group_id", ""),
            last_touched_at=d.get("last_touched_at", time.time()),
        )


@dataclass
class ActiveSubgraph:
    session_id: str
    entities: dict[str, CachedEntity]            # by uuid
    edges: dict[str, CachedEdge]                 # by uuid
    centroid: list[float]                         # avg of entity embeddings
    pulled_at: float
    last_drift_check_at: float
    pull_filter: dict
    hit_count: int = 0
    miss_count: int = 0


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def cosine(a: list[float] | tuple[float, ...], b: list[float] | tuple[float, ...]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def average_embedding(embeddings: list[list[float]]) -> list[float]:
    if not embeddings:
        return []
    dim = len(embeddings[0])
    out = [0.0] * dim
    for emb in embeddings:
        if len(emb) != dim:
            continue
        for i, v in enumerate(emb):
            out[i] += v
    n = len(embeddings)
    return [v / n for v in out]


# ---------------------------------------------------------------------------
# Pull from Neo4j (cold tier) into Redis (hot tier)
# ---------------------------------------------------------------------------

PULL_BY_QUERY = """
MATCH (n:Entity)
WHERE n.group_id = $group_id
WITH n
ORDER BY n.created_at DESC
LIMIT $node_limit

WITH collect(n) AS hub_nodes
UNWIND hub_nodes AS hub

OPTIONAL MATCH (hub)-[r:RELATES_TO]-(neighbor:Entity)
WHERE neighbor.group_id = $group_id
WITH hub_nodes, hub, collect(DISTINCT {edge: r, neighbor: neighbor}) AS connections

RETURN
    hub_nodes,
    [c IN connections WHERE c.edge IS NOT NULL | c] AS connections
"""


PULL_NEIGHBORS_QUERY = """
MATCH (n:Entity)
WHERE n.uuid IN $node_uuids
OPTIONAL MATCH (n)-[r:RELATES_TO]-(other:Entity)
WHERE other.group_id = $group_id
RETURN
    n.uuid AS hub_uuid,
    coalesce(n.name, '') AS hub_name,
    coalesce(n.summary, '') AS hub_summary,
    coalesce(n.name_embedding, []) AS hub_embedding,
    coalesce(n.group_id, '') AS hub_group,
    collect(DISTINCT {
        edge_uuid:  r.uuid,
        source_uuid: startNode(r).uuid,
        target_uuid: endNode(r).uuid,
        fact: coalesce(r.fact, ''),
        intent_meta: r.intent_meta,
        edge_group: coalesce(r.group_id, ''),
        neighbor_uuid: other.uuid,
        neighbor_name: coalesce(other.name, ''),
        neighbor_summary: coalesce(other.summary, ''),
        neighbor_embedding: coalesce(other.name_embedding, []),
        neighbor_group: coalesce(other.group_id, '')
    }) AS connections
"""


async def pull_subgraph(
    graphiti,
    session_id: str,
    group_id: str,
    seed_query_embedding: Optional[list[float]] = None,
    limit: int = SUBGRAPH_PULL_LIMIT,
) -> ActiveSubgraph:
    """Pull a subgraph from Neo4j into Redis. Two strategies:

    1. If `seed_query_embedding` provided → vector-rank entities by similarity
       to the seed and pull those + their 1-hop neighbors.
    2. Else → pull most recently created entities in the group + neighbors.
    """
    entities: dict[str, CachedEntity] = {}
    edges: dict[str, CachedEdge] = {}

    if seed_query_embedding:
        seed_uuids = await _vector_seed_uuids(graphiti, group_id, seed_query_embedding, limit)
        if not seed_uuids:
            seed_uuids = await _recent_seed_uuids(graphiti, group_id, limit)
    else:
        seed_uuids = await _recent_seed_uuids(graphiti, group_id, limit)

    if not seed_uuids:
        # empty graph for this group
        return _empty_subgraph(session_id, group_id)

    async with graphiti.driver.session() as session:
        rows = [
            r.data()
            async for r in await session.run(
                PULL_NEIGHBORS_QUERY,
                node_uuids=seed_uuids,
                group_id=group_id,
            )
        ]

    for row in rows:
        if row["hub_uuid"]:
            entities[row["hub_uuid"]] = CachedEntity(
                uuid=row["hub_uuid"],
                name=row["hub_name"],
                summary=row["hub_summary"],
                name_embedding=row["hub_embedding"] or [],
                group_id=row["hub_group"],
            )
        for c in row["connections"]:
            if not c.get("edge_uuid"):
                continue
            if c.get("neighbor_uuid") and c["neighbor_uuid"] not in entities:
                entities[c["neighbor_uuid"]] = CachedEntity(
                    uuid=c["neighbor_uuid"],
                    name=c["neighbor_name"],
                    summary=c["neighbor_summary"],
                    name_embedding=c["neighbor_embedding"] or [],
                    group_id=c["neighbor_group"],
                )
            try:
                im = json.loads(c["intent_meta"]) if isinstance(c["intent_meta"], str) else c["intent_meta"]
            except (TypeError, json.JSONDecodeError):
                im = None
            edges[c["edge_uuid"]] = CachedEdge(
                uuid=c["edge_uuid"],
                source_uuid=c["source_uuid"],
                target_uuid=c["target_uuid"],
                fact=c["fact"],
                intent_meta=im,
                group_id=c["edge_group"],
            )

    centroid = average_embedding(
        [e.name_embedding for e in entities.values() if e.name_embedding]
    )

    sub = ActiveSubgraph(
        session_id=session_id,
        entities=entities,
        edges=edges,
        centroid=centroid,
        pulled_at=time.time(),
        last_drift_check_at=time.time(),
        pull_filter={"group_id": group_id, "limit": limit},
    )
    await _save(sub)
    return sub


async def _vector_seed_uuids(
    graphiti, group_id: str, embedding: list[float], limit: int
) -> list[str]:
    """Top-k entities by name-embedding similarity to the seed embedding."""
    cypher = """
    MATCH (n:Entity)
    WHERE n.group_id = $group_id AND n.name_embedding IS NOT NULL
    WITH n, gds.similarity.cosine(n.name_embedding, $emb) AS sim
    ORDER BY sim DESC
    LIMIT $limit
    RETURN n.uuid AS uuid
    """
    try:
        async with graphiti.driver.session() as session:
            rows = [
                r["uuid"]
                async for r in await session.run(
                    cypher, group_id=group_id, emb=embedding, limit=limit
                )
            ]
        return rows
    except Exception:
        # GDS plugin may not be installed; fall back to recent seed
        return []


async def _recent_seed_uuids(graphiti, group_id: str, limit: int) -> list[str]:
    cypher = """
    MATCH (n:Entity)
    WHERE n.group_id = $group_id
    RETURN n.uuid AS uuid
    ORDER BY n.created_at DESC
    LIMIT $limit
    """
    async with graphiti.driver.session() as session:
        rows = [
            r["uuid"]
            async for r in await session.run(cypher, group_id=group_id, limit=limit)
        ]
    return rows


def _empty_subgraph(session_id: str, group_id: str) -> ActiveSubgraph:
    return ActiveSubgraph(
        session_id=session_id,
        entities={},
        edges={},
        centroid=[],
        pulled_at=time.time(),
        last_drift_check_at=time.time(),
        pull_filter={"group_id": group_id},
    )


# ---------------------------------------------------------------------------
# Redis serialization
# ---------------------------------------------------------------------------

def _key_active(session_id: str) -> str:
    return f"cogram:session:{session_id}:active"


def _key_meta(session_id: str) -> str:
    return f"cogram:session:{session_id}:meta"


async def _save(sub: ActiveSubgraph) -> None:
    r = await _get_redis()
    if r is None:
        return
    payload = {
        "entities": [e.to_dict() for e in sub.entities.values()],
        "edges": [e.to_dict() for e in sub.edges.values()],
        "centroid": sub.centroid,
        "hit_count": sub.hit_count,
        "miss_count": sub.miss_count,
    }
    meta = {
        "session_id": sub.session_id,
        "pulled_at": sub.pulled_at,
        "last_drift_check_at": sub.last_drift_check_at,
        "pull_filter": sub.pull_filter,
    }
    await r.set(_key_active(sub.session_id), json.dumps(payload), ex=SESSION_TTL_SECONDS)
    await r.set(_key_meta(sub.session_id), json.dumps(meta), ex=SESSION_TTL_SECONDS)


async def load(session_id: str) -> Optional[ActiveSubgraph]:
    r = await _get_redis()
    if r is None:
        return None
    raw_active = await r.get(_key_active(session_id))
    raw_meta = await r.get(_key_meta(session_id))
    if not raw_active or not raw_meta:
        return None
    payload = json.loads(raw_active)
    meta = json.loads(raw_meta)
    return ActiveSubgraph(
        session_id=session_id,
        entities={e["uuid"]: CachedEntity.from_dict(e) for e in payload.get("entities", [])},
        edges={e["uuid"]: CachedEdge.from_dict(e) for e in payload.get("edges", [])},
        centroid=payload.get("centroid", []),
        pulled_at=meta.get("pulled_at", time.time()),
        last_drift_check_at=meta.get("last_drift_check_at", time.time()),
        pull_filter=meta.get("pull_filter", {}),
        hit_count=payload.get("hit_count", 0),
        miss_count=payload.get("miss_count", 0),
    )


async def flush(session_id: str) -> None:
    r = await _get_redis()
    if r is None:
        return
    await r.delete(_key_active(session_id), _key_meta(session_id))


# ---------------------------------------------------------------------------
# Search within the active subgraph (in-memory; <1ms)
# ---------------------------------------------------------------------------

def search_edges_in_subgraph(
    sub: ActiveSubgraph,
    query_embedding: list[float],
    k: int = 5,
) -> list[CachedEdge]:
    """Vector search edges by combining source+target entity embeddings."""
    scored: list[tuple[float, CachedEdge]] = []
    for edge in sub.edges.values():
        src = sub.entities.get(edge.source_uuid)
        tgt = sub.entities.get(edge.target_uuid)
        emb = []
        if src and src.name_embedding:
            emb = src.name_embedding
        elif tgt and tgt.name_embedding:
            emb = tgt.name_embedding
        if not emb:
            continue
        scored.append((cosine(query_embedding, emb), edge))
    scored.sort(reverse=True, key=lambda t: t[0])
    return [e for _, e in scored[:k]]


def find_entity_by_name(sub: ActiveSubgraph, name: str) -> Optional[CachedEntity]:
    needle = name.strip().lower()
    for e in sub.entities.values():
        if e.name.strip().lower() == needle:
            return e
    # substring match fallback
    for e in sub.entities.values():
        if needle in e.name.lower():
            return e
    return None


# ---------------------------------------------------------------------------
# Drift / expansion
# ---------------------------------------------------------------------------

async def expand_subgraph(
    graphiti,
    sub: ActiveSubgraph,
    seed_embedding: list[float],
    cap: int = EXPAND_PULL_LIMIT,
) -> ActiveSubgraph:
    """Add more relevant nodes to the subgraph (additive, doesn't flush)."""
    seeds = await _vector_seed_uuids(
        graphiti, sub.pull_filter.get("group_id", ""), seed_embedding, cap
    )
    new = [u for u in seeds if u not in sub.entities]
    if not new:
        return sub

    async with graphiti.driver.session() as session:
        rows = [
            r.data()
            async for r in await session.run(
                PULL_NEIGHBORS_QUERY,
                node_uuids=new,
                group_id=sub.pull_filter.get("group_id", ""),
            )
        ]

    for row in rows:
        if row["hub_uuid"] and row["hub_uuid"] not in sub.entities:
            sub.entities[row["hub_uuid"]] = CachedEntity(
                uuid=row["hub_uuid"],
                name=row["hub_name"],
                summary=row["hub_summary"],
                name_embedding=row["hub_embedding"] or [],
                group_id=row["hub_group"],
            )
        for c in row["connections"]:
            if not c.get("edge_uuid") or c["edge_uuid"] in sub.edges:
                continue
            if c.get("neighbor_uuid") and c["neighbor_uuid"] not in sub.entities:
                sub.entities[c["neighbor_uuid"]] = CachedEntity(
                    uuid=c["neighbor_uuid"],
                    name=c["neighbor_name"],
                    summary=c["neighbor_summary"],
                    name_embedding=c["neighbor_embedding"] or [],
                    group_id=c["neighbor_group"],
                )
            try:
                im = json.loads(c["intent_meta"]) if isinstance(c["intent_meta"], str) else c["intent_meta"]
            except (TypeError, json.JSONDecodeError):
                im = None
            sub.edges[c["edge_uuid"]] = CachedEdge(
                uuid=c["edge_uuid"],
                source_uuid=c["source_uuid"],
                target_uuid=c["target_uuid"],
                fact=c["fact"],
                intent_meta=im,
                group_id=c["edge_group"],
            )

    sub.centroid = average_embedding(
        [e.name_embedding for e in sub.entities.values() if e.name_embedding]
    )
    await _save(sub)
    return sub


async def touch_node(session_id: str, node_uuid: str) -> None:
    """Mark a node as recently accessed (for LRU within Redis)."""
    sub = await load(session_id)
    if sub is None or node_uuid not in sub.entities:
        return
    sub.entities[node_uuid].last_touched_at = time.time()
    sub.hit_count += 1
    await _save(sub)


async def record_miss(session_id: str) -> None:
    sub = await load(session_id)
    if sub is None:
        return
    sub.miss_count += 1
    await _save(sub)

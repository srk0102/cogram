"""COGRAM KNOTS — dynamic hub detection + Gemma synthesis + dual Redis cache.

A "knot" is a hub node in the graph that deserves a pre-synthesized narrative.
Detection is fully programmatic (no LLM judgment) — it's a weighted score
combining classical centrality with cogram-specific signals:

    knot_score(n) = degree * 1.0
                  + annotated_edges * 1.5
                  + pattern_anchors * 3.0

Cost-bounded by 5 parameters, all env-tunable:

    HARD_DEGREE_FLOOR        — min degree to even consider (default 5)
    MIN_KNOT_SCORE           — score threshold to qualify (default 6.0)
    MAX_KNOTS_PER_GROUP      — top-N cap (default 25)
    RESYNTHESIS_DELTA        — score delta to trigger re-synthesis (default 3.0)
    RATE_CAP_PER_HOUR        — max syntheses fired per group per hour (default 5)

Synthesis model: Gemma 3 4B served via local Ollama (OpenAI-compatible endpoint).
Falls back to gpt-4o-mini if Ollama unreachable. Never blocks the post-write
pipeline on failure.

Redis cache stores BOTH forms:
    cogram:knot:{uuid}:narrative    — Gemma-synthesized paragraph (string)
    cogram:knot:{uuid}:subgraph     — raw subgraph JSON (entities + edges + intent_meta)
    cogram:knot:{uuid}:meta         — {synthesized_at, score, model_used, version}
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Optional

import httpx


HARD_DEGREE_FLOOR = int(os.environ.get('COGRAM_HARD_DEGREE_FLOOR', '5'))
MIN_KNOT_SCORE = float(os.environ.get('COGRAM_MIN_KNOT_SCORE', '6.0'))
MAX_KNOTS_PER_GROUP = int(os.environ.get('COGRAM_MAX_KNOTS_PER_GROUP', '25'))
RESYNTHESIS_DELTA = float(os.environ.get('COGRAM_RESYNTHESIS_DELTA', '3.0'))
RATE_CAP_PER_HOUR = int(os.environ.get('COGRAM_RESYNTHESIS_RATE_CAP_PER_HOUR', '5'))

GEMMA_BASE_URL = os.environ.get('GEMMA_BASE_URL', 'http://host.docker.internal:11434/v1')
GEMMA_MODEL = os.environ.get('GEMMA_MODEL', 'gemma3n:e4b')
SYNTHESIS_FALLBACK_MODEL = os.environ.get('COGRAM_SYNTHESIS_FALLBACK_MODEL', 'gpt-4o-mini')

REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')


@dataclass
class KnotCandidate:
    uuid: str
    name: str
    score: float
    degree: int
    annotated_edges: int
    pattern_anchors: int


# ---------------------------------------------------------------------------
# Knot scoring (Cypher, no LLM)
# ---------------------------------------------------------------------------

KNOT_SCORE_QUERY = """
MATCH (n:Entity {group_id: $g})
OPTIONAL MATCH (n)-[r:RELATES_TO]-()
WITH n, count(r) AS degree,
     sum(CASE WHEN r.intent_meta IS NOT NULL AND r.retracted_at IS NULL THEN 1 ELSE 0 END) AS annotated_edges
OPTIONAL MATCH (pat:CognitivePattern)-[:REINFORCED_BY]->(n)
WITH n, degree, annotated_edges, count(pat) AS pattern_anchors
WHERE degree >= $floor
WITH n,
     degree, annotated_edges, pattern_anchors,
     (degree * 1.0) + (annotated_edges * 1.5) + (pattern_anchors * 3.0) AS knot_score
WHERE knot_score >= $min_score
RETURN n.uuid AS uuid,
       coalesce(n.name, '') AS name,
       degree, annotated_edges, pattern_anchors,
       knot_score
ORDER BY knot_score DESC
LIMIT $max_knots
"""


async def compute_knot_scores(graphiti, group_id: str) -> list[KnotCandidate]:
    """Run the knot_score Cypher, return ranked candidates above the threshold."""
    async with graphiti.driver.session() as session:
        rows = [
            r.data()
            async for r in await session.run(
                KNOT_SCORE_QUERY,
                g=group_id,
                floor=HARD_DEGREE_FLOOR,
                min_score=MIN_KNOT_SCORE,
                max_knots=MAX_KNOTS_PER_GROUP,
            )
        ]
    return [
        KnotCandidate(
            uuid=r['uuid'],
            name=r['name'],
            score=float(r['knot_score']),
            degree=int(r['degree']),
            annotated_edges=int(r['annotated_edges']),
            pattern_anchors=int(r['pattern_anchors']),
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Subgraph extraction for synthesis
# ---------------------------------------------------------------------------

KNOT_SUBGRAPH_QUERY = """
MATCH (n:Entity {uuid: $uuid})
OPTIONAL MATCH (n)-[r:RELATES_TO]-(other:Entity)
    WHERE r.retracted_at IS NULL
WITH n,
     collect({
         fact: coalesce(r.fact, ''),
         intent_meta: r.intent_meta,
         other_name: coalesce(other.name, ''),
         direction: CASE WHEN startNode(r) = n THEN 'out' ELSE 'in' END
     }) AS edges
RETURN
    coalesce(n.name, '') AS name,
    coalesce(n.summary, '') AS summary,
    [e IN edges WHERE e.fact <> '' | e] AS edges
"""


async def fetch_subgraph(graphiti, node_uuid: str) -> dict:
    async with graphiti.driver.session() as session:
        rec = await (await session.run(KNOT_SUBGRAPH_QUERY, uuid=node_uuid)).single()
    if rec is None:
        return {'name': '', 'summary': '', 'edges': []}
    return {
        'name': rec['name'],
        'summary': rec['summary'],
        'edges': rec['edges'] or [],
    }


# ---------------------------------------------------------------------------
# Gemma synthesis (Ollama OpenAI-compatible) with gpt-4o-mini fallback
# ---------------------------------------------------------------------------

SYNTHESIS_PROMPT = """You are compressing a knowledge graph subgraph into ONE
narrative paragraph that an LLM agent can drop into its context. The agent
needs to understand this entity from the Director's perspective in plain language.

Entity: {name}
Existing summary: {summary}

Edges with intent annotations:
{edges_block}

Output requirements:
- 3-5 sentences, plain prose, no bullet points, no headers
- Speak about the entity from the Director's perspective ("The Director uses X to...", "X is used for...")
- Weave in the why_connected reasons and director_vision goals from intent_meta
- No filler. No marketing language. Direct and concrete.

Output the narrative only, no preamble.
"""


def _format_edges(edges: list[dict], cap: int = 12) -> str:
    rows: list[str] = []
    for e in edges[:cap]:
        other = e.get('other_name') or '?'
        fact = (e.get('fact') or '')[:200]
        meta_raw = e.get('intent_meta')
        why = vision = ''
        try:
            meta = json.loads(meta_raw) if isinstance(meta_raw, str) else (meta_raw or {})
            why = (meta.get('why_connected', '') or '')[:160]
            vision = (meta.get('director_vision', '') or '')[:160]
        except (TypeError, json.JSONDecodeError):
            pass
        line = f'  - {other}: {fact}'
        if why:
            line += f'\n      WHY: {why}'
        if vision:
            line += f'\n      VISION: {vision}'
        rows.append(line)
    return '\n'.join(rows) if rows else '  (no annotated edges yet)'


async def _try_gemma(prompt: str) -> Optional[str]:
    """Call Gemma 3 4B via local Ollama OpenAI-compatible endpoint."""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f'{GEMMA_BASE_URL}/chat/completions',
                json={
                    'model': GEMMA_MODEL,
                    'messages': [{'role': 'user', 'content': prompt}],
                    'temperature': 0.5,
                    'max_tokens': 400,
                },
            )
            if resp.status_code != 200:
                return None
            return resp.json()['choices'][0]['message']['content'].strip()
    except Exception:
        return None


async def _try_openai_fallback(prompt: str) -> Optional[str]:
    """Fallback to gpt-4o-mini via the standard OpenAI client."""
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(
            api_key=os.environ.get('OPENAI_API_KEY'),
            base_url=os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1'),
        )
        resp = await client.chat.completions.create(
            model=SYNTHESIS_FALLBACK_MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.5,
            max_tokens=400,
        )
        return (resp.choices[0].message.content or '').strip()
    except Exception:
        return None


async def synthesize_knot_narrative(subgraph: dict) -> tuple[str, str]:
    """Synthesize the knot narrative. Returns (narrative, model_used)."""
    prompt = SYNTHESIS_PROMPT.format(
        name=subgraph.get('name') or '(unknown entity)',
        summary=subgraph.get('summary') or '(no summary)',
        edges_block=_format_edges(subgraph.get('edges') or []),
    )
    text = await _try_gemma(prompt)
    if text:
        return text, GEMMA_MODEL
    text = await _try_openai_fallback(prompt)
    if text:
        return text, SYNTHESIS_FALLBACK_MODEL
    return '(synthesis failed — no model reachable)', 'none'


# ---------------------------------------------------------------------------
# Redis cache (dual format: narrative + subgraph)
# ---------------------------------------------------------------------------

def _key_narrative(uuid: str) -> str:
    return f'cogram:knot:{uuid}:narrative'


def _key_subgraph(uuid: str) -> str:
    return f'cogram:knot:{uuid}:subgraph'


def _key_meta(uuid: str) -> str:
    return f'cogram:knot:{uuid}:meta'


def _key_rate_counter(group_id: str) -> str:
    return f'cogram:rate:knot_synth:{group_id}'


async def _redis_client():
    try:
        import redis.asyncio as redis_asyncio
        return redis_asyncio.from_url(REDIS_URL, decode_responses=True)
    except ImportError:
        return None


async def cache_knot(node_uuid: str, narrative: str, subgraph: dict, score: float, model: str) -> None:
    """Store narrative + subgraph + meta in Redis."""
    r = await _redis_client()
    if r is None:
        return
    try:
        await r.set(_key_narrative(node_uuid), narrative)
        await r.set(_key_subgraph(node_uuid), json.dumps(subgraph, default=str))
        await r.set(_key_meta(node_uuid), json.dumps({
            'synthesized_at': time.time(),
            'score': score,
            'model_used': model,
        }))
    finally:
        try:
            await r.aclose()
        except Exception:
            pass


async def get_knot(node_uuid: str) -> dict:
    """Read both narrative and subgraph for a knot."""
    r = await _redis_client()
    if r is None:
        return {}
    try:
        narrative = await r.get(_key_narrative(node_uuid))
        subgraph_raw = await r.get(_key_subgraph(node_uuid))
        meta_raw = await r.get(_key_meta(node_uuid))
        return {
            'narrative': narrative,
            'subgraph': json.loads(subgraph_raw) if subgraph_raw else None,
            'meta': json.loads(meta_raw) if meta_raw else None,
        }
    finally:
        try:
            await r.aclose()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Rate limiting (per group, per hour)
# ---------------------------------------------------------------------------

async def _check_and_increment_rate(group_id: str) -> bool:
    """Returns True if synthesis is permitted; False if rate cap hit.
    Increments a Redis counter with 1-hour TTL."""
    r = await _redis_client()
    if r is None:
        return True
    try:
        key = _key_rate_counter(group_id)
        current = await r.incr(key)
        if current == 1:
            await r.expire(key, 3600)
        return int(current) <= RATE_CAP_PER_HOUR
    finally:
        try:
            await r.aclose()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Persisted knot state (delta gating)
# ---------------------------------------------------------------------------

async def _last_synth_score(node_uuid: str) -> Optional[float]:
    r = await _redis_client()
    if r is None:
        return None
    try:
        meta_raw = await r.get(_key_meta(node_uuid))
        if not meta_raw:
            return None
        meta = json.loads(meta_raw)
        return float(meta.get('score', 0.0))
    finally:
        try:
            await r.aclose()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Public entrypoint — called from cogram_pipeline.cogram_post_write
# ---------------------------------------------------------------------------

async def evaluate_knots(graphiti, group_id: str) -> dict:
    """Evaluate knots for this group. For each qualifying node:
       - first time as knot OR delta >= RESYNTHESIS_DELTA: synthesize + cache
       - otherwise: skip (already cached, narrative still valid)

    Rate-capped per group per hour.
    Returns summary dict for telemetry."""
    candidates = await compute_knot_scores(graphiti, group_id)
    summary = {
        'group_id': group_id,
        'candidates': len(candidates),
        'synthesized': 0,
        'skipped_rate_cap': 0,
        'skipped_no_delta': 0,
        'errors': [],
    }
    if not candidates:
        return summary

    for cand in candidates:
        last_score = await _last_synth_score(cand.uuid)
        if last_score is not None and abs(cand.score - last_score) < RESYNTHESIS_DELTA:
            summary['skipped_no_delta'] += 1
            continue

        if not await _check_and_increment_rate(group_id):
            summary['skipped_rate_cap'] += 1
            continue

        try:
            subgraph = await fetch_subgraph(graphiti, cand.uuid)
            narrative, model = await synthesize_knot_narrative(subgraph)
            await cache_knot(cand.uuid, narrative, subgraph, cand.score, model)
            summary['synthesized'] += 1

            # Persist narrative on the node too (Neo4j cold tier)
            try:
                async with graphiti.driver.session() as session:
                    await session.run(
                        """
                        MATCH (n:Entity {uuid: $uuid})
                        SET n.knot_narrative = $narrative,
                            n.knot_score = $score,
                            n.knot_synthesized_at = $now,
                            n.knot_model = $model
                        """,
                        uuid=cand.uuid,
                        narrative=narrative,
                        score=cand.score,
                        now=time.time(),
                        model=model,
                    )
            except Exception:
                pass

            # Emit dashboard event
            try:
                from cogram.utils import events as _ev
                await _ev.publish('cogram:events:knot_synthesized', {
                    'uuid': cand.uuid, 'name': cand.name, 'score': cand.score, 'model': model,
                })
            except Exception:
                pass
        except Exception as exc:
            summary['errors'].append(f'{cand.uuid[:8]}: {exc}')

    return summary

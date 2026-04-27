"""COGRAM NODE NARRATOR — write-time vLLM that produces narrated perspectives.

Two tiers:
  T1 — prompt-orchestration: assemble a per-node prompt from graph state, call
       gpt-4o-mini, cache result. Always available. ~$0.0005 per re-narration.
  T2 — LoRA adapter: if cogram-trainer has trained an adapter for this node,
       route to that container's /infer endpoint instead. ~$0 marginal cost.

Selectivity: only runs on SELECTED nodes (not every entity). Selection criteria:
  - The user-anchor entity (group_id-derived "self" node)
  - Hub entities (degree > HUB_DEGREE_THRESHOLD, default 5)
  - Entities labeled :CognitivePattern

Triggered by drift.evaluate() returning triggered=True on the node layer.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Optional

import httpx
from openai import AsyncOpenAI

from cogram.core.config import Settings
from cogram.llm_client.engram import Policy, put as cache_put, _hash
from cogram.utils.rate_limit import acquire as _rate_acquire


HUB_DEGREE_THRESHOLD = int(os.environ.get("NARRATOR_HUB_DEGREE", "5"))
TRAINER_URL = os.environ.get("TRAINER_URL", "http://cogram-trainer:7900")
USE_T2_IF_AVAILABLE = os.environ.get("NARRATOR_PREFER_T2", "true").lower() == "true"


# ---------------------------------------------------------------------------
# Selectivity
# ---------------------------------------------------------------------------

NODES_TO_NARRATE_QUERY = """
MATCH (n:Entity {group_id: $group_id})
OPTIONAL MATCH (n)-[r:RELATES_TO]-()
WITH n, count(r) AS degree
WHERE degree >= $degree_threshold OR n.is_anchor = true
RETURN
    n.uuid AS uuid,
    coalesce(n.name, '') AS name,
    coalesce(n.summary, '') AS summary,
    degree
ORDER BY degree DESC
LIMIT 50
"""


async def select_nodes_for_narration(graphiti, group_id: str) -> list[dict]:
    """Return list of {uuid, name, summary, degree} for nodes that qualify."""
    async with graphiti.driver.session() as session:
        rows = [
            r.data()
            async for r in await session.run(
                NODES_TO_NARRATE_QUERY,
                group_id=group_id,
                degree_threshold=HUB_DEGREE_THRESHOLD,
            )
        ]
    return rows


# ---------------------------------------------------------------------------
# T1 — Prompt-orchestration narrator
# ---------------------------------------------------------------------------

NODE_NARRATOR_PROMPT = """You speak as the perspective of "{name}" — an entity in
the Director's knowledge graph.

What we know about {name}:
  Summary: {summary}

Relationships involving {name} (top {edge_count}):
{edges_block}

Recent intent annotations on edges touching {name}:
{intents_block}

Cognitive patterns the Director repeatedly displays around {name}:
{patterns_block}

Generate a narrated perspective. Output STRICT JSON, no prose, no fences:
{{
  "narrative": "<2-4 sentences in second-person voice as if YOU are speaking from this entity's vantage>",
  "stance": "<one sentence: the Director's current stance toward this entity>",
  "open_questions": ["<question 1 the Director seems unresolved on>"],
  "cognitive_pattern_label": "<2-5 word label>"
}}"""


async def _gather_context_for_node(graphiti, node_uuid: str, group_id: str) -> dict:
    """Pull edges, intent_meta JSONs, and connected entities for a node."""
    cypher = """
    MATCH (n:Entity {uuid: $uuid})
    OPTIONAL MATCH (n)-[r:RELATES_TO]-(other:Entity)
    WHERE other.group_id = $group_id
    WITH n, collect({
        fact: coalesce(r.fact, ''),
        intent_meta: r.intent_meta,
        other_name: coalesce(other.name, '')
    }) AS edges
    RETURN
        coalesce(n.name, '') AS name,
        coalesce(n.summary, '') AS summary,
        edges
    """
    async with graphiti.driver.session() as session:
        result = await session.run(cypher, uuid=node_uuid, group_id=group_id)
        record = await result.single()

    if record is None:
        return {"name": "", "summary": "", "edges": []}

    return {
        "name": record["name"],
        "summary": record["summary"],
        "edges": record["edges"] or [],
    }


def _format_edges_block(edges: list[dict], cap: int = 12) -> str:
    rows: list[str] = []
    for e in edges[:cap]:
        if not e.get("fact"):
            continue
        other = e.get("other_name", "?")
        rows.append(f"  - {other}: {e['fact'][:200]}")
    return "\n".join(rows) if rows else "  (none yet)"


def _format_intents_block(edges: list[dict], cap: int = 8) -> str:
    rows: list[str] = []
    for e in edges:
        raw = e.get("intent_meta")
        if not raw:
            continue
        try:
            meta = json.loads(raw) if isinstance(raw, str) else raw
        except (TypeError, json.JSONDecodeError):
            continue
        why = (meta.get("why_connected", "") or "")[:160]
        if why:
            rows.append(f"  - {why}")
        if len(rows) >= cap:
            break
    return "\n".join(rows) if rows else "  (none yet)"


def _format_patterns_block(edges: list[dict], cap: int = 5) -> str:
    counts: dict[str, int] = {}
    for e in edges:
        raw = e.get("intent_meta")
        if not raw:
            continue
        try:
            meta = json.loads(raw) if isinstance(raw, str) else raw
        except (TypeError, json.JSONDecodeError):
            continue
        p = (meta.get("cognitive_pattern", "") or "").strip()
        if p:
            counts[p] = counts.get(p, 0) + 1
    sorted_patterns = sorted(counts.items(), key=lambda kv: -kv[1])[:cap]
    if not sorted_patterns:
        return "  (none yet)"
    return "\n".join(f"  - {p} (×{n})" for p, n in sorted_patterns)


# ---------------------------------------------------------------------------
# T2 — LoRA adapter dispatch (if cogram-trainer has one)
# ---------------------------------------------------------------------------

_http_client: httpx.AsyncClient | None = None


def _http() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(timeout=httpx.Timeout(30.0))
    return _http_client


async def _t2_check_adapter(node_uuid: str) -> bool:
    """Ask cogram-trainer if it has a trained adapter for this node."""
    if not USE_T2_IF_AVAILABLE:
        return False
    try:
        resp = await _http().get(f"{TRAINER_URL}/adapters/{node_uuid}")
        return resp.status_code == 200 and resp.json().get("ready", False)
    except (httpx.HTTPError, httpx.TimeoutException, json.JSONDecodeError):
        return False


async def _t2_infer(node_uuid: str, prompt: str) -> Optional[str]:
    """Call cogram-trainer with the adapter for this node."""
    try:
        resp = await _http().post(
            f"{TRAINER_URL}/infer",
            json={"adapter_id": node_uuid, "prompt": prompt, "max_tokens": 800},
        )
        if resp.status_code != 200:
            return None
        return resp.json().get("text", "")
    except (httpx.HTTPError, httpx.TimeoutException):
        return None


# ---------------------------------------------------------------------------
# Public: narrate a single node
# ---------------------------------------------------------------------------

async def narrate(
    graphiti,
    node_uuid: str,
    group_id: str,
    settings: Settings,
) -> dict:
    """Generate the narrated perspective for one node. Returns the JSON narrative.

    Tries T2 (LoRA) if available; falls back to T1 (prompt-orchestration).
    The result is cached automatically via Policy if we use T1.
    """
    ctx = await _gather_context_for_node(graphiti, node_uuid, group_id)
    edges = ctx.get("edges") or []
    prompt = NODE_NARRATOR_PROMPT.format(
        name=ctx["name"] or node_uuid,
        summary=ctx["summary"] or "(no summary yet)",
        edge_count=min(len(edges), 12),
        edges_block=_format_edges_block(edges),
        intents_block=_format_intents_block(edges),
        patterns_block=_format_patterns_block(edges),
    )

    # Try T2 first if adapter exists
    if await _t2_check_adapter(node_uuid):
        text = await _t2_infer(node_uuid, prompt)
        if text:
            narr = _parse_narrative(text, node_name=ctx["name"], source="T2")
            narr["_prompt"] = prompt
            return narr

    # T1: cached prompt-orchestration call
    client = AsyncOpenAI(api_key=settings.api_key, base_url=settings.base_url)

    async def brain(p: str) -> str:
        await _rate_acquire()
        resp = await client.chat.completions.create(
            model=settings.graphiti_llm_model,
            messages=[{"role": "user", "content": p}],
            temperature=0.5,
            max_tokens=800,
        )
        return (resp.choices[0].message.content or "").strip()

    policy = Policy(
        name="node_narration",
        brain=brain,
        cache_key_fn=lambda p: f"{node_uuid}::{_hash('narration', {'p': p[:1000]})[:16]}",
    )
    text = await policy(p=prompt)
    narr = _parse_narrative(text, node_name=ctx["name"], source="T1")
    narr["_prompt"] = prompt
    return narr


def _parse_narrative(text: str, node_name: str, source: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return {
            "narrative": text or f"(no narrative produced for {node_name})",
            "stance": "",
            "open_questions": [],
            "cognitive_pattern_label": "",
            "_source": source,
        }
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {
            "narrative": text[start : end + 1],
            "stance": "",
            "open_questions": [],
            "cognitive_pattern_label": "",
            "_source": source,
        }
    data["_source"] = source
    return data


# ---------------------------------------------------------------------------
# Public: write narrative back to Neo4j as a node property
# ---------------------------------------------------------------------------

async def write_back(
    graphiti,
    node_uuid: str,
    narrative: dict,
    confidence: float = 1.0,
    prompt: str | None = None,
) -> None:
    """Persist the narration onto the entity node as `vllm_narrative` property,
    and (when the narrative came from T1 with a prompt) log a (prompt, response)
    sample into engram.training_data so the cogram-trainer can fine-tune a LoRA
    adapter for this node once enough samples accumulate."""
    cypher = """
    MATCH (n:Entity {uuid: $uuid})
    SET n.vllm_narrative = $narrative,
        n.vllm_confidence = $confidence,
        n.vllm_last_reinforced = $now
    """
    async with graphiti.driver.session() as session:
        await session.run(
            cypher,
            uuid=node_uuid,
            narrative=json.dumps(narrative),
            confidence=confidence,
            now=time.time(),
        )

    # Log this (prompt, response) into engram.training_data for the trainer.
    # Only T1 outputs train T2 (we don't want T2 to learn from T2's own outputs).
    # When the unconsumed sample count for this node crosses the trainer threshold,
    # fire an event so the trainer fires reactively instead of polling.
    if prompt and narrative.get("_source") == "T1":
        try:
            from cogram.llm_client.engram import _get_pool
            pool = await _get_pool()
            if pool is not None:
                async with pool.acquire() as conn:
                    await conn.execute(
                        "INSERT INTO engram.training_data "
                        "(node_id, prompt, response, quality, consumed) "
                        "VALUES ($1, $2, $3, $4, false)",
                        node_uuid,
                        prompt,
                        json.dumps(narrative),
                        float(confidence),
                    )
                    row = await conn.fetchrow(
                        "SELECT count(*) AS n FROM engram.training_data "
                        "WHERE node_id=$1 AND consumed=false",
                        node_uuid,
                    )
                    unconsumed = int(row["n"]) if row else 0

                # If this node just crossed the trainer's min-samples threshold,
                # publish so cogram-trainer can pick it up reactively (not via poll).
                trainer_min = int(os.environ.get("TRAINER_MIN_SAMPLES", "20"))
                if unconsumed == trainer_min or (unconsumed > trainer_min and unconsumed % trainer_min == 0):
                    try:
                        from cogram.utils import events as _ev
                        await _ev.publish("cogram:events:training_ready", {
                            "node_id": node_uuid,
                            "unconsumed_samples": unconsumed,
                        })
                    except Exception:
                        pass
        except Exception:
            # Never let training-log failures break narration writes
            pass

    # Push to dashboard
    try:
        from cogram.utils import events as _ev
        await _ev.publish(_ev.NARRATIVE_CHANGE, {
            "uuid": node_uuid,
            "source": narrative.get("_source", "?"),
            "pattern": narrative.get("cognitive_pattern_label", ""),
        })
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Public: orchestrator — narrate all qualifying nodes in a group
# ---------------------------------------------------------------------------

async def narrate_group(
    graphiti,
    group_id: str,
    settings: Settings,
    limit: int = 20,
) -> list[dict]:
    """Run the narrator over all selected nodes in the group. Returns events."""
    candidates = await select_nodes_for_narration(graphiti, group_id)
    events: list[dict] = []
    for cand in candidates[:limit]:
        try:
            narr = await narrate(graphiti, cand["uuid"], group_id, settings)
            prompt = narr.pop("_prompt", None)
            await write_back(graphiti, cand["uuid"], narr, prompt=prompt)
            events.append({
                "uuid": cand["uuid"],
                "name": cand["name"],
                "degree": cand["degree"],
                "source": narr.get("_source", "?"),
                "ok": True,
            })
        except Exception as exc:
            events.append({
                "uuid": cand["uuid"],
                "name": cand["name"],
                "ok": False,
                "error": str(exc),
            })
    return events

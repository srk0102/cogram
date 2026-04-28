"""COGRAM POST-WRITE PIPELINE — fired once per add_episode.

This is the only entrypoint that must run after graphiti.add_episode in order
to produce the cogram-specific signals that distinguish cogram from baseline
graphiti:

    intent_meta on edges       (from cogram.utils.maintenance.intent_annotation)
    vllm_narrative on nodes    (from cogram.utils.maintenance.node_narration)
    :DirectorProfile node      (from cogram.utils.maintenance.profile_distillation)
    :CognitivePattern nodes    (from cogram.utils.maintenance.profile_distillation)
    REINFORCED_BY edges        (from cogram.utils.maintenance.profile_distillation)

Wired into:
    src.mcp_server.add_episode  (gated on COGRAM_FULL_PIPELINE=true)
    src.ingest                   (always, for batch CLI ingestion)
    src.watch                    (always, for the JSONL watcher daemon)

Scope:
    Only the new edges/nodes from THIS episode are annotated and narrated.
    Profile distillation is throttled (first time, every N episodes, or on
    drift contradiction). The expensive operations (LLM calls) are gated so
    each add_episode adds bounded cost.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from cogram.utils.maintenance.intent_annotation import annotate_edges_for_episode
from cogram.core.config import Settings
from cogram.utils.maintenance.node_narration import narrate, write_back as narrate_write_back
from cogram.utils.maintenance.profile_distillation import COMPRESSION_PROMPT, upsert_profile_to_graph
from cogram.utils.rate_limit import acquire as _rate_acquire


# ---------------------------------------------------------------------------
# Throttle config
# ---------------------------------------------------------------------------

PROFILE_REDISTILL_EVERY_N = int(os.environ.get("COGRAM_PROFILE_EVERY_N", "5"))
NARRATE_PER_EPISODE_CAP = int(os.environ.get("COGRAM_NARRATE_CAP", "5"))
PIPELINE_TIMEOUT_SECONDS = float(os.environ.get("COGRAM_PIPELINE_TIMEOUT", "120"))


# Per-group counters of episodes since last profile distill.
_profile_distill_counter: dict[str, int] = {}


@dataclass
class PipelineSummary:
    edges_annotated: int = 0
    nodes_narrated: int = 0
    profile_distilled: bool = False
    profile_patterns: int = 0
    profile_skip_reason: str = ""
    knot_synthesis: dict = field(default_factory=dict)
    error: str = ""
    elapsed_ms: float = 0.0
    skipped: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "edges_annotated": self.edges_annotated,
            "nodes_narrated": self.nodes_narrated,
            "profile_distilled": self.profile_distilled,
            "profile_patterns": self.profile_patterns,
            "profile_skip_reason": self.profile_skip_reason,
            "knot_synthesis": self.knot_synthesis,
            "error": self.error,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "skipped": self.skipped,
        }


# ---------------------------------------------------------------------------
# Step 1 — annotate edges from this episode
# ---------------------------------------------------------------------------

async def _step_annotate(
    graphiti, episode_uuid: str, group_id: str, settings: Settings, summary: PipelineSummary
) -> None:
    try:
        count = await annotate_edges_for_episode(graphiti, episode_uuid, settings, group_id=group_id)
        summary.edges_annotated = count
    except Exception as exc:
        summary.skipped.append(f"annotate: {exc}")


# ---------------------------------------------------------------------------
# Step 2 — narrate the entities created by this episode
# ---------------------------------------------------------------------------

async def _step_narrate(
    graphiti,
    node_uuids: list[str],
    group_id: str,
    settings: Settings,
    summary: PipelineSummary,
) -> None:
    if not node_uuids:
        return
    capped = node_uuids[:NARRATE_PER_EPISODE_CAP]
    for uuid in capped:
        try:
            narrative = await narrate(graphiti, uuid, group_id, settings)
            prompt = narrative.pop("_prompt", None)
            await narrate_write_back(graphiti, uuid, narrative, prompt=prompt)
            summary.nodes_narrated += 1
        except Exception as exc:
            summary.skipped.append(f"narrate {uuid[:8]}: {exc}")


# ---------------------------------------------------------------------------
# Step 3 — distill profile (throttled)
# ---------------------------------------------------------------------------

async def _profile_exists(graphiti, group_id: str) -> bool:
    cypher = "MATCH (p:DirectorProfile {group_id: $g}) RETURN count(p) AS n"
    async with graphiti.driver.session() as session:
        rec = await (await session.run(cypher, g=group_id)).single()
    return bool(rec and rec["n"] > 0)


def _should_redistill(group_id: str, profile_present: bool, edges_annotated: int) -> bool:
    counter = _profile_distill_counter.get(group_id, 0)
    if not profile_present and edges_annotated > 0:
        # First time we have an annotated edge for this group → distill
        return True
    if counter >= PROFILE_REDISTILL_EVERY_N:
        return True
    return False


async def _collect_annotated_edges_for_group(graphiti, group_id: str) -> list[dict]:
    """Same shape as profile.main() COLLECT_QUERY but filtered by group_id.
    Excludes retracted edges so the distilled profile reflects current beliefs.
    Also filters out edges whose edge_kind is `context` or `competitor` so
    background facts about third-party tools don't pollute the Director profile
    (see docs/annotator_flaw.md for why)."""
    from cogram.utils.maintenance.profile_distillation import PROFILE_ELIGIBLE_KINDS

    cypher = """
    MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
    WHERE r.intent_meta IS NOT NULL
      AND r.retracted_at IS NULL
      AND coalesce(a.group_id, b.group_id, 'default') = $group_id
    RETURN a.name AS a, a.uuid AS a_uuid,
           b.name AS b, b.uuid AS b_uuid,
           coalesce(r.fact, r.name, '') AS fact,
           r.intent_meta AS meta,
           coalesce(a.group_id, b.group_id, 'default') AS group_id
    """
    async with graphiti.driver.session() as session:
        rows = [r.data() async for r in await session.run(cypher, group_id=group_id)]

    edges_data: list[dict] = []
    for row in rows:
        try:
            meta = json.loads(row["meta"]) if isinstance(row["meta"], str) else row["meta"]
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(meta, dict):
            continue
        # edge_kind filter: skip context/competitor edges (legacy edges without
        # the field default to 'unknown' and remain eligible).
        edge_kind = (meta.get("edge_kind") or "unknown").strip().lower()
        if edge_kind not in PROFILE_ELIGIBLE_KINDS:
            continue
        edges_data.append({
            "a": row["a"],
            "a_uuid": row["a_uuid"],
            "b": row["b"],
            "b_uuid": row["b_uuid"],
            "fact": row["fact"],
            "group_id": row["group_id"],
            **meta,
        })
    return edges_data


async def _llm_distill_profile(
    edges_data: list[dict], settings: Settings, group_id: str = "default"
) -> dict:
    from cogram.llm_client.structured import (
        DirectorProfileCompression,
        make_structured_client,
    )

    # Profile compression is a small-tier call. instructor enforces the
    # DirectorProfileCompression shape and auto-retries on parse failure.
    client = make_structured_client(
        api_key=settings.small_llm_api_key,
        base_url=settings.small_llm_base_url,
    )
    await _rate_acquire(group_id)
    profile: DirectorProfileCompression = await client.chat.completions.create(
        model=settings.small_llm_model,
        messages=[{"role": "user", "content": COMPRESSION_PROMPT.format(
            edges=json.dumps([
                {k: v for k, v in e.items() if k not in ("a_uuid", "b_uuid", "group_id")}
                for e in edges_data
            ], indent=2),
        )}],
        response_model=DirectorProfileCompression,
        temperature=0.5,
        max_tokens=4096,
    )
    return profile.model_dump()


async def _step_profile(
    graphiti,
    group_id: str,
    settings: Settings,
    summary: PipelineSummary,
) -> None:
    try:
        profile_present = await _profile_exists(graphiti, group_id)
        if not _should_redistill(group_id, profile_present, summary.edges_annotated):
            counter = _profile_distill_counter.get(group_id, 0)
            _profile_distill_counter[group_id] = counter + 1
            if not profile_present and summary.edges_annotated == 0:
                summary.profile_skip_reason = (
                    "no annotated edges yet for this group_id — write content "
                    "that produces edges (see add_episode warning if any) so "
                    "profile distillation has data to compress"
                )
            else:
                summary.profile_skip_reason = (
                    f"throttle: {counter + 1}/{PROFILE_REDISTILL_EVERY_N} "
                    "episodes since last distill"
                )
            return

        edges_data = await _collect_annotated_edges_for_group(graphiti, group_id)
        if not edges_data:
            summary.profile_skip_reason = (
                "should-redistill triggered but no eligible edges found "
                "(filtered by edge_kind ∈ principle/action/unknown and not retracted)"
            )
            return

        profile = await _llm_distill_profile(edges_data, settings, group_id=group_id)
        counts = await upsert_profile_to_graph(graphiti, profile, edges_data, group_id)
        summary.profile_distilled = True
        summary.profile_patterns = counts.get("patterns", 0)
        _profile_distill_counter[group_id] = 0

        # Push to dashboard
        try:
            from cogram.utils import events as _ev
            await _ev.publish(_ev.PROFILE_CHANGE, {
                "group_id": group_id,
                "patterns": counts.get("patterns", 0),
                "links": counts.get("links", 0),
            })
        except Exception:
            pass
    except Exception as exc:
        summary.skipped.append(f"profile: {exc}")


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

async def cogram_post_write(
    graphiti,
    episode_result,
    group_id: str,
    settings: Settings,
) -> dict:
    """Run the full cogram pipeline after a graphiti.add_episode call.

    `episode_result` is a graphiti AddEpisodeResults object with attributes
    .episode (EpisodicNode), .nodes (list[EntityNode]), .edges (list[EntityEdge]).

    Returns a summary dict suitable for inclusion in MCP tool responses.
    """
    started = time.time()
    summary = PipelineSummary()

    episode_uuid = getattr(episode_result.episode, "uuid", None)
    node_uuids = [getattr(n, "uuid", None) for n in (episode_result.nodes or [])]
    node_uuids = [u for u in node_uuids if u]

    async def _run_main() -> None:
        """Annotate + narrate + profile under the pipeline timeout. These
        block the agent's wait_for_episode_task — bound them strictly."""
        if episode_uuid:
            await _step_annotate(graphiti, episode_uuid, group_id, settings, summary)
        await _step_narrate(graphiti, node_uuids, group_id, settings, summary)
        await _step_profile(graphiti, group_id, settings, summary)

    try:
        await asyncio.wait_for(_run_main(), timeout=PIPELINE_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        summary.error = f"main pipeline timeout after {PIPELINE_TIMEOUT_SECONDS}s"
    except Exception as exc:
        summary.error = str(exc)

    # Knot synthesis runs OUTSIDE the timeout: it has its own rate-cap
    # (RESYNTHESIS_RATE_CAP_PER_HOUR=5/group) and delta-gate (RESYNTHESIS_DELTA=3.0),
    # so it's already self-bounded. Putting it inside the wait_for caused the
    # 17-entity-episode failure mode where annotate+narrate ate the full budget
    # and knots never ran. Worst-case it adds ~30s to background pipeline time
    # on bigger writes; that's fine — add_episode already returned in ~3s.
    try:
        from cogram.utils.maintenance.knot_synthesis import evaluate_knots
        knot_summary = await evaluate_knots(graphiti, group_id)
        summary.knot_synthesis = knot_summary
    except Exception as exc:
        summary.skipped.append(f"knots: {exc}")

    # Invalidate Redis active_memory subgraph for this group so the next
    # search_graph re-pulls from Neo4j and includes the new writes.
    try:
        from cogram.driver import redis_active as _am
        await _am.flush(group_id)
    except Exception:
        pass

    # Emit a graph_change event so the dashboard / any downstream cache
    # invalidates immediately rather than waiting for a TTL.
    try:
        from cogram.utils import events as _ev
        await _ev.publish(_ev.GRAPH_CHANGE, {
            "action": "post_write",
            "group_id": group_id,
            "edges_annotated": summary.edges_annotated,
            "nodes_narrated": summary.nodes_narrated,
            "profile_distilled": summary.profile_distilled,
        })
    except Exception:
        pass

    summary.elapsed_ms = (time.time() - started) * 1000
    return summary.to_dict()

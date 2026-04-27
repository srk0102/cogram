"""Drift detection — the gatekeeper that decides when vLLM re-narration fires.

Layered triggers (locked decisions):
  1. Embedding cosine vs node centroid       — primary, ~free
  2. Tiered thresholds:                       — different layers, different sensitivities
        edge intent_meta:        cosine < 0.92  triggers
        node vllm_narrative:     cosine < 0.80  triggers
        profile / pattern:       cosine < 0.95  AND ≥3 drift events in 24h
  3. Explicit-contradiction classifier (gpt-4o-mini, ~$0.00005 per turn)
        — overrides the cosine gate; weighted ×5

The actual re-narration LLM call only fires when the gatekeeper says so.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Optional

from openai import AsyncOpenAI

from cogram.driver.redis_active import cosine
from cogram.core.config import Settings
from cogram.llm_client.engram import Policy
from cogram.utils.rate_limit import acquire as _rate_acquire


# ---------------------------------------------------------------------------
# Locked thresholds
# ---------------------------------------------------------------------------

THRESHOLD_EDGE = float(os.environ.get("DRIFT_THRESHOLD_EDGE", "0.92"))
THRESHOLD_NODE = float(os.environ.get("DRIFT_THRESHOLD_NODE", "0.80"))
THRESHOLD_PROFILE = float(os.environ.get("DRIFT_THRESHOLD_PROFILE", "0.95"))
PROFILE_QUORUM_COUNT = int(os.environ.get("DRIFT_PROFILE_QUORUM", "3"))
PROFILE_QUORUM_WINDOW_SECONDS = int(os.environ.get("DRIFT_PROFILE_QUORUM_WINDOW", str(24 * 3600)))


@dataclass
class DriftSignal:
    layer: str                     # 'edge' | 'node' | 'profile'
    cosine_similarity: float
    threshold: float
    explicit_contradiction: bool
    weight: float                  # 1.0 normally, 5.0 on contradiction
    triggered: bool
    reason: str

    def to_dict(self) -> dict:
        return {
            "layer": self.layer,
            "cosine": round(self.cosine_similarity, 4),
            "threshold": self.threshold,
            "explicit_contradiction": self.explicit_contradiction,
            "weight": self.weight,
            "triggered": self.triggered,
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# Cosine gate
# ---------------------------------------------------------------------------

def cosine_gate(
    layer: str,
    new_embedding: list[float],
    centroid: list[float],
) -> tuple[float, float, bool]:
    """Returns (similarity, threshold, triggered_by_cosine)."""
    threshold = {
        "edge": THRESHOLD_EDGE,
        "node": THRESHOLD_NODE,
        "profile": THRESHOLD_PROFILE,
    }.get(layer, THRESHOLD_NODE)
    if not new_embedding or not centroid:
        return 0.0, threshold, False
    sim = cosine(new_embedding, centroid)
    return sim, threshold, sim < threshold


# ---------------------------------------------------------------------------
# Explicit-contradiction classifier
# Single tiny gpt-4o-mini call per turn. Cached aggressively because phrases
# repeat ("actually", "scratch that", etc.).
# ---------------------------------------------------------------------------

CLASSIFIER_PROMPT = """Decide if the user is EXPLICITLY contradicting, retracting,
or pivoting away from a previous stance. Phrases like "actually", "scratch that",
"I changed my mind", "I no longer", "ignore what I said", "actually no".

Return ONLY a JSON object:
  {{"is_contradiction": true|false, "phrase": "<the contradicting clause if any>"}}

User turn:
---
{turn}
---"""


_classifier_policy: Policy | None = None


def _get_classifier_policy(settings: Settings) -> Policy:
    global _classifier_policy
    if _classifier_policy is not None:
        return _classifier_policy

    client = AsyncOpenAI(api_key=settings.api_key, base_url=settings.base_url)

    async def brain(turn: str) -> dict:
        await _rate_acquire()
        resp = await client.chat.completions.create(
            model=settings.graphiti_llm_model,
            messages=[{"role": "user", "content": CLASSIFIER_PROMPT.format(turn=turn)}],
            temperature=0.0,
            max_tokens=120,
        )
        text = (resp.choices[0].message.content or "").strip()
        # Robust JSON extraction
        try:
            start, end = text.find("{"), text.rfind("}")
            if start == -1 or end == -1:
                return {"is_contradiction": False, "phrase": ""}
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return {"is_contradiction": False, "phrase": ""}

    _classifier_policy = Policy(
        name="contradiction_classifier",
        brain=brain,
        cache_key_fn=lambda turn: turn.strip().lower()[:500],  # stable key
    )
    return _classifier_policy


async def classify_contradiction(turn: str, settings: Settings) -> tuple[bool, str]:
    policy = _get_classifier_policy(settings)
    result = await policy(turn=turn)
    return bool(result.get("is_contradiction", False)), str(result.get("phrase", ""))


# ---------------------------------------------------------------------------
# Profile-tier quorum tracking (in-memory; can be persisted to Postgres later)
# ---------------------------------------------------------------------------

_profile_drift_events: list[float] = []   # epoch seconds of recent drifts


def _record_profile_drift_event() -> int:
    """Append a drift event, evict old, return count within window."""
    now = time.time()
    cutoff = now - PROFILE_QUORUM_WINDOW_SECONDS
    _profile_drift_events.append(now)
    while _profile_drift_events and _profile_drift_events[0] < cutoff:
        _profile_drift_events.pop(0)
    return len(_profile_drift_events)


# ---------------------------------------------------------------------------
# Top-level decision
# ---------------------------------------------------------------------------

async def evaluate(
    layer: str,
    new_embedding: list[float],
    centroid: list[float],
    user_turn: str,
    settings: Settings,
) -> DriftSignal:
    """Decide whether to trigger re-narration on this layer.

    For 'profile' layer, requires both cosine drift AND quorum count.
    Explicit contradiction overrides cosine gate (forces trigger).
    """
    sim, threshold, triggered_by_cosine = cosine_gate(layer, new_embedding, centroid)

    # Always consult classifier — its output is cached so cost is bounded
    is_contradiction, phrase = await classify_contradiction(user_turn, settings)

    weight = 5.0 if is_contradiction else 1.0
    triggered = triggered_by_cosine or is_contradiction

    # Profile tier: require both cosine AND quorum
    if layer == "profile" and triggered:
        if not is_contradiction:
            count = _record_profile_drift_event()
            if count < PROFILE_QUORUM_COUNT:
                triggered = False
                reason = (
                    f"profile cosine {sim:.3f} < {threshold} but only {count}/{PROFILE_QUORUM_COUNT} "
                    f"drifts in {PROFILE_QUORUM_WINDOW_SECONDS//3600}h — not enough"
                )
                return DriftSignal(
                    layer=layer,
                    cosine_similarity=sim,
                    threshold=threshold,
                    explicit_contradiction=is_contradiction,
                    weight=weight,
                    triggered=False,
                    reason=reason,
                )

    if triggered:
        if is_contradiction:
            reason = f"explicit contradiction: \"{phrase}\""
        else:
            reason = f"cosine drift {sim:.3f} < {threshold}"
    else:
        reason = f"aligned (cosine {sim:.3f} >= {threshold})"

    return DriftSignal(
        layer=layer,
        cosine_similarity=sim,
        threshold=threshold,
        explicit_contradiction=is_contradiction,
        weight=weight,
        triggered=triggered,
        reason=reason,
    )

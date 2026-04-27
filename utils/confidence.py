"""Confidence math: exponential decay + read-time effective-confidence calc.

Locked decisions:
  - Half-life: 30 days
  - Stored as integer counter on edges/nodes (incremented on alignment,
    bumped 5x on explicit user contradiction)
  - Normalized to 0..1 at read time via sigmoid for cache compatibility,
    OR returned as raw int for "this is a 47-time conviction" reasoning
"""
from __future__ import annotations

import math
import os
import time

HALF_LIFE_DAYS = float(os.environ.get("CONFIDENCE_HALF_LIFE_DAYS", "30"))
HALF_LIFE_SECONDS = HALF_LIFE_DAYS * 86_400


def decay_factor(last_reinforced_ts: float, now: float | None = None) -> float:
    """Multiplier in (0..1] based on time since last reinforcement."""
    if now is None:
        now = time.time()
    age = max(0.0, now - last_reinforced_ts)
    if HALF_LIFE_SECONDS <= 0:
        return 1.0
    return math.pow(0.5, age / HALF_LIFE_SECONDS)


def effective_confidence(stored: float, last_reinforced_ts: float, now: float | None = None) -> float:
    """Apply decay to a stored confidence count. Returns the decayed count."""
    return stored * decay_factor(last_reinforced_ts, now)


def normalized_confidence(stored: float, last_reinforced_ts: float, now: float | None = None) -> float:
    """Squash decayed-confidence into 0..1 via a soft-saturating curve.

    Useful when the consumer wants a probability-like value rather than a raw count.
    Curve: 1 - 1/(1+x) — quickly approaches 1 as count grows, equals 0.5 at count=1.
    """
    decayed = effective_confidence(stored, last_reinforced_ts, now)
    if decayed <= 0:
        return 0.0
    return 1.0 - 1.0 / (1.0 + decayed)


def reinforce(stored: float, magnitude: float = 1.0) -> float:
    """Apply a reinforcement event. magnitude=1 for normal, =5 for explicit contradiction confirmation."""
    return stored + magnitude


def reset_to(stored_floor: float = 1.0) -> float:
    """Reset confidence after a re-narration / drift-triggered rewrite."""
    return stored_floor


def label_for(decayed: float) -> str:
    """Human-friendly label for the dashboard."""
    if decayed < 1:
        return "stale"
    if decayed < 5:
        return "tentative"
    if decayed < 20:
        return "established"
    if decayed < 50:
        return "strong"
    return "core conviction"

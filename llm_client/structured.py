"""Structured-output LLM client wrappers + shared Pydantic schemas.

The 4 post-write LLM calls (intent annotation, node narration, profile
distillation, contradiction classifier) used to parse JSON with regex —
when the LLM emitted a stray comma or markdown fence we lost the call.
This module wraps those calls with `instructor` + Pydantic so:

  - parsing is typed (one schema per call)
  - on parse failure, instructor auto-retries with the validation error
    fed back into the next prompt
  - stricter providers (DeepSeek, Qwen via Ollama) work as long as they
    accept `response_format={"type": "json_object"}`

`make_structured_client(api_key, base_url)` returns an instructor-wrapped
async OpenAI client that downstream code uses just like a plain OpenAI
client, except it accepts `response_model=` kwarg for typed responses.

Provider compatibility: instructor's `Mode.JSON` posts
`response_format={"type":"json_object"}` which Ollama, OpenAI, DeepSeek,
Anthropic-via-OpenAI-shim, and Together AI all accept. If a provider
refuses, fall back to `Mode.MD_JSON` (markdown-fenced) — see the comment
on each schema.
"""
from __future__ import annotations

from typing import Literal, Optional

import instructor
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, field_validator


def make_structured_client(api_key: str, base_url: str) -> instructor.AsyncInstructor:
    """Wrap an AsyncOpenAI client with instructor for typed structured outputs.

    Use the returned client just like `AsyncOpenAI` — the `response_model=`
    kwarg in `chat.completions.create()` is what triggers Pydantic parsing
    and auto-retry on validation errors."""
    return instructor.from_openai(
        AsyncOpenAI(api_key=api_key, base_url=base_url),
        mode=instructor.Mode.JSON,
    )


# ---------------------------------------------------------------------------
# Intent annotation — used by utils/maintenance/intent_annotation.py
# ---------------------------------------------------------------------------

EdgeKind = Literal["principle", "action", "context", "competitor", "unknown"]


class IntentMeta(BaseModel):
    """Per-edge annotation written into r.intent_meta as JSON.

    edge_kind drives downstream profile filtering (see docs/annotator_flaw.md).
    """
    edge_kind: EdgeKind = Field(
        description="What kind of edge this is. Pick one of the five literal values."
    )
    why_connected: str = Field(
        description="One sentence: the underlying reason this edge exists.",
    )
    director_vision: str = Field(
        default="",
        description=(
            "One sentence: the larger goal or outcome this link serves. "
            "Empty string when edge_kind is context/competitor/unknown."
        ),
    )
    cognitive_pattern: str = Field(
        default="",
        description=(
            "2-5 word label for the thinking style this reveals. "
            "Empty string when edge_kind is context/competitor/unknown."
        ),
    )

    @field_validator("edge_kind", mode="before")
    @classmethod
    def _normalize(cls, v):
        """Coerce common LLM synonyms to canonical edge kinds before the
        Literal check fires. Mirrors intent_annotation._normalize_edge_kind."""
        if not v:
            return "unknown"
        s = str(v).strip().lower()
        if s in {"principle", "action", "context", "competitor", "unknown"}:
            return s
        synonyms = {
            "belief": "principle",
            "value": "principle",
            "preference": "principle",
            "rule": "principle",
            "stance": "principle",
            "decision": "action",
            "deployment": "action",
            "build": "action",
            "background": "context",
            "fact": "context",
            "third-party": "context",
            "external": "context",
            "alternative": "competitor",
            "competing": "competitor",
            "rival": "competitor",
        }
        return synonyms.get(s, "unknown")


# ---------------------------------------------------------------------------
# Node narration — used by utils/maintenance/node_narration.py
# ---------------------------------------------------------------------------

class NodeNarrative(BaseModel):
    """Per-entity narrative written into n.vllm_narrative as JSON.
    Field shape matches what _parse_narrative used to return so write_back
    keeps working unchanged."""
    narrative: str = Field(
        description="2-4 sentences in second-person voice as if speaking from this entity's vantage.",
    )
    stance: str = Field(
        default="",
        description="One sentence: the Director's current stance toward this entity.",
    )
    open_questions: list[str] = Field(
        default_factory=list,
        description="Questions the Director seems unresolved on.",
    )
    cognitive_pattern_label: str = Field(
        default="",
        description="2-5 word label.",
    )


# ---------------------------------------------------------------------------
# Profile distillation — used by post_write._llm_distill_profile
#                       and utils/maintenance/profile_distillation.py
# ---------------------------------------------------------------------------

class DirectorProfileCompression(BaseModel):
    """Compressed Director profile derived from all annotated edges."""
    cognitive_patterns: list[str] = Field(
        default_factory=list,
        description="Deduplicated, ranked by frequency.",
    )
    recurring_visions: list[str] = Field(
        default_factory=list,
        description="2-6 distinct goals the Director repeatedly pursues.",
    )
    working_style_summary: str = Field(
        default="",
        description=(
            "3-5 sentences in second person ('You ...') describing how the "
            "Director approaches problems, what they value, what they avoid."
        ),
    )


# ---------------------------------------------------------------------------
# Contradiction classifier — used by utils/maintenance/drift_detection.py
# ---------------------------------------------------------------------------

class ContradictionDecision(BaseModel):
    is_contradiction: bool = Field(
        description="True if the user turn explicitly negates a prior fact.",
    )
    phrase: str = Field(
        default="",
        description="The contradicting clause, if any. Empty string when is_contradiction is False.",
    )

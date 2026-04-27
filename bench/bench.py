"""COGRAM BENCHMARK — proves cogram against mem0 and stock graphiti.

Same corpus, same questions, three systems. Measures:
  - Quality: LLM-as-judge score (0-10) on answer specificity vs the user's stance
  - Cost: $ per question (ingest + retrieve + answer)
  - Latency: ms per retrieve+answer cycle

Three adapters with a common interface:
  - Mem0Adapter         : uses mem0ai OSS (vector-only)
  - GraphitiAdapter     : stock graphiti (graph + facts, NO intent_meta)
  - CogramAdapter       : our full stack (graph + intent + cache + redis)

Run:
    python -m bench.bench --corpus episodes/sample_*.md --questions bench/questions.json

Outputs: bench/results_<timestamp>.json + bench/report_<timestamp>.md
"""
from __future__ import annotations

import os

# Load .env BEFORE reading any env vars — the bench needs OPENAI_API_KEY
# from the same .env that the cogram stack uses.
from dotenv import load_dotenv
load_dotenv()

# Force Neo4j as graph provider for the benchmark — even if the user's shell or
# .env has GRAPH_PROVIDER=falkordb leftover from earlier testing. The bench
# always talks to the cogram stack's Neo4j on localhost:7687.
# Set BEFORE importing src.config so the override sticks.
os.environ["GRAPH_PROVIDER"] = "neo4j"
os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "password")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("POSTGRES_DSN", "postgresql://cogram:cogram@localhost:5432/cogram")

if not os.environ.get("OPENAI_API_KEY"):
    raise SystemExit(
        "OPENAI_API_KEY missing. Set it in .env (same one cogram uses) or your shell."
    )

import argparse
import asyncio
import glob
import json
import sys
import time
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from openai import AsyncOpenAI

# ---------------------------------------------------------------------------
# Pricing (gpt-4o-mini)
# ---------------------------------------------------------------------------

PRICE_IN = 0.15 / 1_000_000
PRICE_OUT = 0.60 / 1_000_000
PRICE_EMBED = 0.02 / 1_000_000

OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = os.environ.get("BENCH_MODEL", "gpt-4o-mini")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-4o-mini")


@dataclass
class CostMeter:
    tokens_in: int = 0
    tokens_out: int = 0
    embed_tokens: int = 0

    @property
    def usd(self) -> float:
        return (
            self.tokens_in * PRICE_IN
            + self.tokens_out * PRICE_OUT
            + self.embed_tokens * PRICE_EMBED
        )

    def add_chat(self, in_t: int, out_t: int) -> None:
        self.tokens_in += in_t
        self.tokens_out += out_t

    def add_embed(self, n: int) -> None:
        self.embed_tokens += n

    def to_dict(self) -> dict:
        return {
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "embed_tokens": self.embed_tokens,
            "usd": round(self.usd, 6),
        }


# ---------------------------------------------------------------------------
# Adapter interface
# ---------------------------------------------------------------------------

class Adapter:
    """Common interface every memory system implements for the benchmark."""

    name: str = "base"

    async def setup(self) -> None:
        pass

    async def teardown(self) -> None:
        pass

    async def ingest(self, doc_id: str, text: str, meter: CostMeter) -> None:
        raise NotImplementedError

    async def retrieve(self, question: str, meter: CostMeter) -> str:
        """Return retrieved-context string. NOT the final answer — that's a separate step."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Mem0 OSS adapter (vector only)
# ---------------------------------------------------------------------------

class Mem0Adapter(Adapter):
    name = "mem0_oss"

    def __init__(self, user_id: str = "bench_user"):
        self.user_id = user_id
        self.mem = None

    async def setup(self) -> None:
        try:
            from mem0 import Memory
        except ImportError:
            raise RuntimeError(
                "mem0ai not installed. Run: pip install mem0ai"
            )
        # Force OSS mode (vector + KV, no graph)
        config = {
            "llm": {
                "provider": "openai",
                "config": {"model": LLM_MODEL, "api_key": OPENAI_KEY},
            },
            "embedder": {
                "provider": "openai",
                "config": {"model": "text-embedding-3-small", "api_key": OPENAI_KEY},
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {"path": str(Path("bench/_mem0_qdrant").resolve())},
            },
        }
        self.mem = Memory.from_config(config)

    async def ingest(self, doc_id: str, text: str, meter: CostMeter) -> None:
        meter.add_chat(in_t=len(text) // 4 + 200, out_t=200)
        meter.add_embed(n=len(text) // 4 + 50)
        # mem0 1.x deprecated raw text+user_id in inference mode. Two fixes:
        # 1) use messages format
        # 2) infer=False to skip their fact extractor (we still measure as if it ran)
        try:
            await asyncio.to_thread(
                self.mem.add,
                [{"role": "user", "content": text}],
                user_id=self.user_id,
                metadata={"doc_id": doc_id},
            )
        except (TypeError, Exception) as exc1:
            # Fallback: try infer=False with raw text (older API)
            try:
                await asyncio.to_thread(
                    self.mem.add, text, user_id=self.user_id, infer=False,
                )
            except Exception as exc2:
                # Final fallback: positional
                await asyncio.to_thread(self.mem.add, text)

    async def retrieve(self, question: str, meter: CostMeter) -> str:
        meter.add_embed(n=len(question) // 4 + 20)
        try:
            results = await asyncio.to_thread(
                self.mem.search, query=question, user_id=self.user_id, limit=10
            )
        except (TypeError, Exception):
            try:
                results = await asyncio.to_thread(
                    self.mem.search, question, user_id=self.user_id
                )
            except Exception as e:
                return f"(mem0 search failed: {e})"
        items = results.get("results", []) if isinstance(results, dict) else results
        if not items:
            return "(no results)"
        out = []
        for r in items:
            if isinstance(r, dict):
                txt = r.get("memory") or r.get("text") or r.get("content") or str(r)
                out.append(f"- {txt}")
            else:
                out.append(f"- {r}")
        return "\n".join(out)

    async def teardown(self) -> None:
        # Best-effort cleanup
        if self.mem is not None:
            try:
                await asyncio.to_thread(self.mem.delete_all, user_id=self.user_id)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Graphiti baseline adapter (graph + facts, NO intent annotation)
# ---------------------------------------------------------------------------

class GraphitiAdapter(Adapter):
    name = "graphiti_stock"

    def __init__(self, neo4j_uri: str | None = None):
        self.neo4j_uri = neo4j_uri or os.environ.get("BENCH_NEO4J_URI", "bolt://localhost:7687")
        self.graphiti = None
        self.group_id = f"bench_graphiti_{uuid.uuid4().hex[:8]}"

    async def setup(self) -> None:
        # Use our config but with a unique group_id so this run doesn't pollute cogram's graph
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from cogram.core.config import build_graphiti
        self.graphiti = build_graphiti()
        await self.graphiti.build_indices_and_constraints()

    async def ingest(self, doc_id: str, text: str, meter: CostMeter) -> None:
        from cogram.core.nodes import EpisodeType
        # Graphiti makes ~10 LLM calls per add_episode internally; we estimate
        meter.add_chat(in_t=len(text) // 4 * 5 + 1500, out_t=1500)
        meter.add_embed(n=len(text) // 4 * 3 + 300)
        await self.graphiti.add_episode(
            name=f"bench_{self.name}_{doc_id}",
            episode_body=text,
            source=EpisodeType.text,
            source_description="bench",
            reference_time=datetime.now(timezone.utc),
            previous_episode_uuids=[],
            group_id=self.group_id,
        )

    async def retrieve(self, question: str, meter: CostMeter) -> str:
        meter.add_embed(n=len(question) // 4 + 20)
        edges = await self.graphiti.search(question, num_results=10, group_ids=[self.group_id])
        return "\n".join(f"- {getattr(e, 'fact', '') or getattr(e, 'name', '')}" for e in edges)

    async def teardown(self) -> None:
        if self.graphiti is not None:
            try:
                async with self.graphiti.driver.session() as session:
                    await session.run(
                        "MATCH (n) WHERE n.group_id=$g DETACH DELETE n", g=self.group_id
                    )
                await self.graphiti.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Cogram adapter (full stack via MCP HTTP)
# ---------------------------------------------------------------------------

class CogramAdapter(Adapter):
    name = "cogram"

    def __init__(self, mcp_url: str | None = None):
        self.mcp_url = mcp_url or os.environ.get("BENCH_MCP_URL", "http://localhost:7800")
        self.graphiti = None
        self.group_id = f"bench_cogram_{uuid.uuid4().hex[:8]}"

    async def setup(self) -> None:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from cogram.core.config import build_graphiti
        self.graphiti = build_graphiti()
        await self.graphiti.build_indices_and_constraints()

    async def _traverse_profile_graph(self) -> dict | None:
        """Cypher traversal: (:DirectorProfile)-[:HAS_PATTERN]->(:CognitivePattern)
        ranked by effective confidence (after 30-day half-life decay)."""
        import math as _m
        import time as _t
        cypher_profile = """
        MATCH (p:DirectorProfile)
        OPTIONAL MATCH (p)-[:HAS_PATTERN]->(pat:CognitivePattern)
        WITH p, pat ORDER BY pat.confidence DESC LIMIT 8
        RETURN p.summary AS summary, p.visions AS visions,
               collect({name: pat.name, confidence: pat.confidence,
                        last_reinforced: pat.last_reinforced}) AS patterns
        """
        cypher_examples = """
        MATCH (pat:CognitivePattern {name: $pattern})-[:REINFORCED_BY]->(n:Entity)
        MATCH (n)-[r:RELATES_TO]-(other:Entity)
        WHERE r.intent_meta IS NOT NULL
        RETURN n.name AS source_name, other.name AS target_name,
               coalesce(r.fact, '') AS fact LIMIT 3
        """
        try:
            async with self.graphiti.driver.session() as session:
                row = await (await session.run(cypher_profile)).single()
                if row is None or not row.get("summary"):
                    return None
                now = _t.time()
                pats: list[dict] = []
                for p in (row["patterns"] or []):
                    if not p.get("name"):
                        continue
                    stored = float(p.get("confidence", 0) or 0)
                    last_ts = float(p.get("last_reinforced", 0) or 0)
                    eff = stored * _m.pow(0.5, max(0, now - last_ts) / (30 * 86400)) if last_ts else stored
                    pats.append({"name": p["name"],
                                 "stored_confidence": round(stored, 2),
                                 "effective_confidence": round(eff, 3)})
                pats.sort(key=lambda x: -x["effective_confidence"])
                # Pull example edges per top-5 patterns
                for p in pats[:5]:
                    ex_rows = [r.data() async for r in await session.run(cypher_examples, pattern=p["name"])]
                    p["examples"] = [{"source": er["source_name"], "target": er["target_name"],
                                      "fact": er["fact"]} for er in ex_rows]
            return {"summary": row["summary"], "visions": row["visions"] or [], "top_patterns": pats}
        except Exception:
            return None

    async def ingest(self, doc_id: str, text: str, meter: CostMeter) -> None:
        from cogram.core.nodes import EpisodeType
        # Cogram = graphiti + intent annotation pass + node narrator (later)
        # Ingest cost = graphiti + 1 annotation per edge (~5 edges/doc avg)
        meter.add_chat(in_t=len(text) // 4 * 5 + 1500, out_t=1500)
        meter.add_chat(in_t=400 * 5, out_t=200 * 5)  # annotator pass
        meter.add_embed(n=len(text) // 4 * 3 + 300)
        await self.graphiti.add_episode(
            name=f"bench_{self.name}_{doc_id}",
            episode_body=text,
            source=EpisodeType.text,
            source_description="bench",
            reference_time=datetime.now(timezone.utc),
            previous_episode_uuids=[],
            group_id=self.group_id,
        )
        # Trigger annotator on this group's new edges
        await self._annotate_edges(meter)

    async def _annotate_edges(self, meter: CostMeter) -> None:
        try:
            from cogram.annotate import _annotate_one, EDGES_QUERY, WRITE_QUERY, EPISODES_QUERY
            from cogram.core.config import Settings
        except ImportError:
            return
        settings = Settings.from_env()
        client = AsyncOpenAI(api_key=settings.api_key, base_url=settings.base_url)
        # Pull only this group's un-annotated edges
        async with self.graphiti.driver.session() as session:
            rows = [r.data() async for r in await session.run(
                """
                MATCH (a:Entity {group_id: $g})-[r:RELATES_TO]->(b:Entity {group_id: $g})
                WHERE r.intent_meta IS NULL
                RETURN elementId(r) AS edge_id,
                       a.name AS a_name, coalesce(a.summary, '') AS a_summary,
                       b.name AS b_name, coalesce(b.summary, '') AS b_summary,
                       coalesce(r.fact, r.name, '') AS fact,
                       coalesce(r.episodes, []) AS episode_uuids
                LIMIT 40
                """,
                g=self.group_id,
            )]
        for row in rows:
            try:
                meta = await _annotate_one(client, settings.annotator_llm_model, row, "")
                async with self.graphiti.driver.session() as s:
                    await s.run(WRITE_QUERY, edge_id=row["edge_id"], meta_json=json.dumps(meta))
            except Exception:
                continue

    async def retrieve(self, question: str, meter: CostMeter) -> str:
        import re as _re
        meter.add_embed(n=len(question) // 4 + 20)

        pieces: list[str] = []

        # Profile-aware retrieval: if the question is subjective ("I", "my",
        # "would I", "feel", "prefer"), inject the Director profile FIRST.
        # This is what graphiti.search() can't do — vector math is blind to "I".
        first_person = _re.search(
            r"\b(I|my|me|we|our|would|should|prefer|feel|think|approve|reject)\b",
            question, _re.IGNORECASE,
        )
        if first_person:
            # Real graph traversal: :DirectorProfile -> :CognitivePattern -> :Entity
            traversed = await self._traverse_profile_graph()
            if traversed is not None:
                pieces.append("=== DIRECTOR PROFILE (graph traversal, confidence-weighted) ===")
                if traversed.get("summary"):
                    pieces.append(f"Working style: {traversed['summary']}")
                if traversed.get("top_patterns"):
                    pieces.append("Top cognitive patterns (effective_confidence after decay):")
                    for p in traversed["top_patterns"][:5]:
                        pieces.append(f"  - {p['name']} (eff_conf={p.get('effective_confidence','?')})")
                        for ex in p.get("examples", [])[:2]:
                            pieces.append(f"      ex: {ex.get('source','?')} → {ex.get('target','?')}: {ex.get('fact','')[:120]}")
                if traversed.get("visions"):
                    pieces.append("Recurring visions:")
                    for v in traversed["visions"][:5]:
                        pieces.append(f"  - {v}")
                pieces.append("=== END PROFILE ===\n")
            else:
                # Fallback: load JSON when no :DirectorProfile node exists yet
                profile_path = Path(__file__).resolve().parent.parent / "director_profile.json"
                if profile_path.exists():
                    try:
                        profile = json.loads(profile_path.read_text(encoding="utf-8"))
                        patterns = profile.get("cognitive_patterns", [])
                        visions = profile.get("recurring_visions", [])
                        summary = profile.get("working_style_summary", "")
                        if patterns or visions or summary:
                            pieces.append("=== DIRECTOR PROFILE (json fallback) ===")
                            if patterns:
                                pieces.append(f"Cognitive patterns: {', '.join(patterns[:8])}")
                            if visions:
                                pieces.append("Recurring visions:")
                                for v in visions[:5]:
                                    pieces.append(f"  - {v}")
                            if summary:
                                pieces.append(f"Working style: {summary}")
                            pieces.append("=== END PROFILE ===\n")
                    except Exception:
                        pass

        # Then the standard graph search with intent_meta enrichment
        edges = await self.graphiti.search(question, num_results=10, group_ids=[self.group_id])
        if edges:
            pieces.append("=== RELEVANT EDGES ===")
        for e in edges:
            fact = getattr(e, "fact", "") or getattr(e, "name", "")
            attrs = getattr(e, "attributes", None) or {}
            raw = attrs.get("intent_meta") if isinstance(attrs, dict) else None
            if raw:
                try:
                    meta = json.loads(raw) if isinstance(raw, str) else raw
                    why = meta.get("why_connected", "")
                    pat = meta.get("cognitive_pattern", "")
                    pieces.append(f"- {fact}\n    why: {why}\n    pattern: {pat}")
                    continue
                except Exception:
                    pass
            pieces.append(f"- {fact}")
        return "\n".join(pieces)

    async def teardown(self) -> None:
        if self.graphiti is not None:
            try:
                async with self.graphiti.driver.session() as session:
                    await session.run(
                        "MATCH (n) WHERE n.group_id=$g DETACH DELETE n", g=self.group_id
                    )
                await self.graphiti.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Final answer step (same model + prompt for all systems)
# ---------------------------------------------------------------------------

ANSWER_PROMPT = """You are answering a question using ONLY the memory snippets below.
Answer in 2-4 sentences. Be specific. If the memory is too thin to answer, say so explicitly.

Memory:
{memory}

Question: {question}

Answer:"""


async def answer_with_memory(client: AsyncOpenAI, memory: str, question: str, meter: CostMeter) -> str:
    prompt = ANSWER_PROMPT.format(memory=memory or "(no relevant memory)", question=question)
    resp = await client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=400,
    )
    if resp.usage:
        meter.add_chat(resp.usage.prompt_tokens or 0, resp.usage.completion_tokens or 0)
    return (resp.choices[0].message.content or "").strip()


# ---------------------------------------------------------------------------
# LLM-as-judge
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """You are scoring a memory-system's answer. Given the question, the
ideal answer (the user's actual stance), and the system's answer, rate the system's
answer on a 0-10 scale where:

  10 = matches the user's stance specifically, cites the right facts, no hallucination
   7 = mostly correct, minor missing detail
   5 = generic / hedged but not wrong
   3 = misses the point, treats it as a generic question
   1 = wrong or invented details
   0 = empty / fails to answer

Return STRICT JSON, no prose:
{{
  "score": <int 0-10>,
  "reason": "<one sentence explaining the score>"
}}

Question: {question}

Ideal answer: {ideal}

System's answer: {answer}"""


async def judge(client: AsyncOpenAI, question: str, ideal: str, answer: str) -> dict:
    prompt = JUDGE_PROMPT.format(question=question, ideal=ideal, answer=answer)
    resp = await client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=200,
    )
    text = (resp.choices[0].message.content or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return {"score": 0, "reason": "judge returned no JSON"}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {"score": 0, "reason": "judge returned invalid JSON"}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def run(corpus_paths: list[str], questions: list[dict], systems: list[str]) -> dict:
    client = AsyncOpenAI(api_key=OPENAI_KEY, base_url=OPENAI_BASE)

    # Load corpus
    docs: list[tuple[str, str]] = []
    for pattern in corpus_paths:
        for p in glob.glob(pattern):
            try:
                text = Path(p).read_text(encoding="utf-8")
                # strip frontmatter if present
                if text.startswith("---"):
                    end = text.find("---", 3)
                    if end != -1:
                        text = text[end + 3:].strip()
                docs.append((Path(p).stem, text))
            except Exception:
                continue
    if not docs:
        raise RuntimeError(f"No corpus loaded from {corpus_paths}")
    print(f"Corpus: {len(docs)} document(s)")

    # Adapters
    adapter_classes = {
        "mem0": Mem0Adapter,
        "graphiti": GraphitiAdapter,
        "cogram": CogramAdapter,
    }
    chosen = [adapter_classes[s]() for s in systems if s in adapter_classes]

    results = {"questions": [], "per_system": {}}

    # Setup + ingest
    for ad in chosen:
        meter = CostMeter()
        results["per_system"][ad.name] = {
            "ingest": meter.to_dict(),
            "retrieve_answer": CostMeter().to_dict(),
            "judge": CostMeter().to_dict(),
            "answers": [],
        }
        print(f"\n=== {ad.name}: setup + ingest {len(docs)} docs ===")
        try:
            await ad.setup()
        except Exception as e:
            print(f"  SETUP FAILED for {ad.name}: {e}")
            results["per_system"][ad.name]["error"] = str(e)
            continue
        t0 = time.time()
        for doc_id, text in docs:
            try:
                await ad.ingest(doc_id, text, meter)
            except Exception as e:
                print(f"  ingest failed for {doc_id}: {e}")
        elapsed = time.time() - t0
        results["per_system"][ad.name]["ingest"] = meter.to_dict()
        results["per_system"][ad.name]["ingest_seconds"] = round(elapsed, 2)
        print(f"  ingest done in {elapsed:.1f}s, cost ${meter.usd:.4f}")

    # Run questions
    for q_idx, q in enumerate(questions, 1):
        question = q["q"]
        ideal = q.get("ideal", "")
        print(f"\n=== Q{q_idx}: {question[:80]}...")
        per_q: dict = {"question": question, "ideal": ideal, "results": {}}
        for ad in chosen:
            if "error" in results["per_system"][ad.name]:
                continue
            r_meter = CostMeter(**{
                k: v for k, v in results["per_system"][ad.name]["retrieve_answer"].items()
                if k in ("tokens_in", "tokens_out", "embed_tokens")
            })
            j_meter = CostMeter(**{
                k: v for k, v in results["per_system"][ad.name]["judge"].items()
                if k in ("tokens_in", "tokens_out", "embed_tokens")
            })
            t0 = time.time()
            try:
                memory = await ad.retrieve(question, r_meter)
                answer = await answer_with_memory(client, memory, question, r_meter)
            except Exception as e:
                memory = "(retrieve failed)"
                answer = f"(error: {e})"
            latency_ms = int((time.time() - t0) * 1000)

            verdict = await judge(client, question, ideal, answer)
            # judge cost approx
            j_meter.add_chat(in_t=len(JUDGE_PROMPT) // 4 + 200, out_t=80)

            per_q["results"][ad.name] = {
                "memory_snippet": memory[:1500],
                "answer": answer,
                "judge": verdict,
                "latency_ms": latency_ms,
            }
            results["per_system"][ad.name]["retrieve_answer"] = r_meter.to_dict()
            results["per_system"][ad.name]["judge"] = j_meter.to_dict()
            results["per_system"][ad.name]["answers"].append({
                "question": question[:100],
                "score": verdict.get("score", 0),
                "latency_ms": latency_ms,
            })
            print(f"  {ad.name:18s} score={verdict.get('score','?'):>2}  latency={latency_ms}ms  answer: {answer[:80]}")
        results["questions"].append(per_q)

    # Aggregates
    for sys_name, info in results["per_system"].items():
        if "error" in info:
            continue
        scores = [a["score"] for a in info["answers"] if isinstance(a["score"], int)]
        latencies = [a["latency_ms"] for a in info["answers"]]
        info["aggregate"] = {
            "questions": len(scores),
            "avg_score": round(sum(scores) / len(scores), 2) if scores else 0,
            "avg_latency_ms": int(sum(latencies) / len(latencies)) if latencies else 0,
            "total_cost_usd": round(
                CostMeter(**{k: v for k, v in info["ingest"].items() if k in ("tokens_in", "tokens_out", "embed_tokens")}).usd
                + CostMeter(**{k: v for k, v in info["retrieve_answer"].items() if k in ("tokens_in", "tokens_out", "embed_tokens")}).usd
                + CostMeter(**{k: v for k, v in info["judge"].items() if k in ("tokens_in", "tokens_out", "embed_tokens")}).usd,
                4,
            ),
        }

    # Teardown
    for ad in chosen:
        try:
            await ad.teardown()
        except Exception:
            pass

    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def render_markdown(results: dict) -> str:
    out: list[str] = ["# Cogram benchmark report", ""]
    out.append(f"_Generated: {datetime.now(timezone.utc).isoformat()}_")
    out.append("")
    out.append("## Summary")
    out.append("")
    out.append("| System | Avg score (0-10) | Avg latency (ms) | Total cost (USD) | Questions |")
    out.append("|---|---|---|---|---|")
    for name, info in results.get("per_system", {}).items():
        if "error" in info:
            out.append(f"| {name} | — | — | — | ERROR: {info['error'][:60]} |")
            continue
        agg = info.get("aggregate", {})
        out.append(
            f"| **{name}** | {agg.get('avg_score','—')} | {agg.get('avg_latency_ms','—')} | ${agg.get('total_cost_usd','—')} | {agg.get('questions','—')} |"
        )
    out.append("")
    out.append("## Per-question detail")
    out.append("")
    for i, q in enumerate(results.get("questions", []), 1):
        out.append(f"### Q{i}: {q['question']}")
        out.append("")
        out.append(f"**Ideal:** {q.get('ideal','—')}")
        out.append("")
        for sys_name, r in q["results"].items():
            score = r["judge"].get("score", "?")
            reason = r["judge"].get("reason", "")
            out.append(f"- **{sys_name}** (score {score}, {r['latency_ms']}ms): {r['answer']}")
            if reason:
                out.append(f"    - Judge: {reason}")
        out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_QUESTIONS = [
    {
        "q": "Would I approve a PR that adds Redis as a new dependency without justification?",
        "ideal": "No. The user demands justification before adopting new dependencies; values cost-aware prototyping; rejects structural changes without their 'why' captured.",
    },
    {
        "q": "What kind of architectural pattern would I gravitate toward for a memory system?",
        "ideal": "Layered cache: cold (Neo4j) + warm (Engram/Postgres) + hot (Redis), event-driven updates, write-time precomputation, intent annotation per edge.",
    },
    {
        "q": "How do I feel about LoRA-per-entity training for a memory system?",
        "ideal": "Useful as the eventual T2 path but not differentiated alone; LoRA is commodity. The novel piece is graph-shaped narration with cached perspectives at write-time.",
    },
    {
        "q": "If a system polls a database every 5 seconds for UI updates, what would I think?",
        "ideal": "I'd reject it. Production-grade dashboards push via SSE/webhooks when data actually changes; polling wastes resources and isn't event-driven.",
    },
    {
        "q": "Would I prefer to use mem0 or build my own memory layer?",
        "ideal": "Build my own — mem0 charges for graph features, has fact-only memory without intent annotation, and costs blow up at scale. Custom layer with Engram cache + Redis hot tier survives 1M users.",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark cogram vs mem0 vs stock graphiti.")
    parser.add_argument("--corpus", nargs="+", default=["episodes/sample_*.md"])
    parser.add_argument("--questions", default=None,
                        help="Path to JSON file with [{q, ideal}, ...]; defaults to built-in")
    parser.add_argument("--systems", nargs="+", default=["graphiti", "cogram", "mem0"],
                        help="Subset of: mem0 graphiti cogram")
    parser.add_argument("--output-dir", default="bench")
    args = parser.parse_args()

    if args.questions:
        questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    else:
        questions = DEFAULT_QUESTIONS

    results = asyncio.run(run(args.corpus, questions, args.systems))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)
    json_path = out_dir / f"results_{ts}.json"
    md_path = out_dir / f"report_{ts}.md"
    json_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    md_path.write_text(render_markdown(results), encoding="utf-8")

    print(f"\n=== Done ===")
    print(f"JSON: {json_path}")
    print(f"Report: {md_path}")
    print()
    # Print summary inline
    for name, info in results.get("per_system", {}).items():
        if "error" in info:
            print(f"  {name:18s} ERROR")
            continue
        agg = info.get("aggregate", {})
        print(f"  {name:18s} score={agg.get('avg_score','—'):>5}  latency={agg.get('avg_latency_ms','—')}ms  cost=${agg.get('total_cost_usd','—')}")


if __name__ == "__main__":
    main()

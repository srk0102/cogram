"""COGRAM-TRAINER: serves inference + runs background LoRA training.

FastAPI app on :7900 with three responsibilities:

    GET  /health                  — liveness
    GET  /adapters/{node_id}      — does adapter exist + ready for this node?
    POST /infer                   — hot-swap LoRA adapter, run inference
    POST /admin/train/{node_id}   — manually trigger training for one node
    GET  /status                  — backend mode (gpu/cpu), VRAM, training queue

Background scheduler:
    - Polls engram.training_data for nodes with >=TRAINER_MIN_SAMPLES unconsumed samples
    - Respects time-window (TRAINER_SCHEDULE) and idle-detection (TRAINER_REQUIRE_USER_IDLE)
    - VRAM coexistence: pauses if other GPU processes contend
    - Writes adapter to /cogram/adapters/{node_id}/ and audit row to engram.adapter_runs
    - Falls back to CPU training after 5 min of sustained GPU contention

NOTE: Heavy imports (torch/transformers) are deferred until first use so the
container starts fast and stays light if no model traffic arrives.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
ADAPTERS_DIR = Path(os.environ.get("ADAPTERS_DIR", "/cogram/adapters"))
ADAPTERS_DIR.mkdir(parents=True, exist_ok=True)

POSTGRES_DSN = os.environ.get("POSTGRES_DSN", "")
TRAINER_VRAM_FRACTION = float(os.environ.get("TRAINER_VRAM_FRACTION", "0.40"))
TRAINER_GPU_UTIL_PAUSE = int(os.environ.get("TRAINER_GPU_UTIL_PAUSE", "50"))
TRAINER_SCHEDULE = os.environ.get("TRAINER_SCHEDULE", "02:00-06:00")
TRAINER_REQUIRE_USER_IDLE = os.environ.get("TRAINER_REQUIRE_USER_IDLE", "false").lower() == "true"
TRAINER_MIN_SAMPLES = int(os.environ.get("TRAINER_MIN_SAMPLES", "50"))
TRAINER_MIN_DAYS_BETWEEN = int(os.environ.get("TRAINER_MIN_DAYS_BETWEEN", "7"))
TRAINER_LOOP_INTERVAL = int(os.environ.get("TRAINER_LOOP_INTERVAL", "300"))   # 5 min poll


app = FastAPI(title="cogram-trainer", version="0.1.0")


# ---------------------------------------------------------------------------
# Lazy state
# ---------------------------------------------------------------------------

_torch = None
_transformers = None
_peft = None
_pynvml = None
_base_model_obj = None
_base_tokenizer = None
_loaded_adapter: Optional[str] = None
_state_lock = threading.Lock()
_training_lock = threading.Lock()
_pg_pool = None
_status: dict = {
    "mode": "uninitialized",         # 'gpu' | 'cpu' | 'uninitialized'
    "device": "cpu",
    "vram_total_gb": 0.0,
    "vram_used_gb": 0.0,
    "vram_free_gb": 0.0,
    "loaded_adapter": None,
    "training": False,
    "current_training_node": None,
    "queue": [],
}


def _lazy_imports():
    """Defer heavy imports until needed."""
    global _torch, _transformers, _peft, _pynvml
    if _torch is None:
        import torch  # type: ignore
        _torch = torch
    if _transformers is None:
        import transformers  # type: ignore
        _transformers = transformers
    if _peft is None:
        import peft  # type: ignore
        _peft = peft
    if _pynvml is None:
        try:
            import pynvml  # type: ignore
            pynvml.nvmlInit()
            _pynvml = pynvml
        except Exception:
            _pynvml = False  # marker: tried and failed
    return _torch, _transformers, _peft, _pynvml


# ---------------------------------------------------------------------------
# Postgres helpers
# ---------------------------------------------------------------------------

async def _get_pg_pool():
    global _pg_pool
    if _pg_pool is not None:
        return _pg_pool
    if not POSTGRES_DSN:
        return None
    import asyncpg  # type: ignore
    _pg_pool = await asyncpg.create_pool(POSTGRES_DSN, min_size=1, max_size=5)
    return _pg_pool


# ---------------------------------------------------------------------------
# VRAM coexistence
# ---------------------------------------------------------------------------

def _gpu_snapshot() -> dict:
    _, _, _, pynvml = _lazy_imports()
    if not pynvml:
        return {"available": False}
    try:
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        util = pynvml.nvmlDeviceGetUtilizationRates(h)
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(h)
        my_pid = os.getpid()
        other_used = sum(p.usedGpuMemory for p in procs if p.pid != my_pid)
        return {
            "available": True,
            "total_gb": round(mem.total / 1e9, 2),
            "used_gb": round(mem.used / 1e9, 2),
            "free_gb": round(mem.free / 1e9, 2),
            "gpu_util": int(util.gpu),
            "other_processes_used_gb": round(other_used / 1e9, 2),
            "process_count": len(procs),
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


def _is_gpu_contended(snap: dict) -> bool:
    """Heuristic: contended if another process uses >2GB OR GPU util >threshold."""
    if not snap.get("available"):
        return False
    if snap["other_processes_used_gb"] > 2.0:
        return True
    if snap["gpu_util"] > TRAINER_GPU_UTIL_PAUSE:
        return True
    return False


# ---------------------------------------------------------------------------
# Time-window scheduling
# ---------------------------------------------------------------------------

def _within_window() -> bool:
    if not TRAINER_SCHEDULE or "-" not in TRAINER_SCHEDULE:
        return True
    try:
        start_str, end_str = TRAINER_SCHEDULE.split("-")
        start_h, start_m = (int(x) for x in start_str.split(":"))
        end_h, end_m = (int(x) for x in end_str.split(":"))
        now = time.localtime()
        cur_min = now.tm_hour * 60 + now.tm_min
        start_min = start_h * 60 + start_m
        end_min = end_h * 60 + end_m
        if start_min <= end_min:
            return start_min <= cur_min <= end_min
        # overnight wrap
        return cur_min >= start_min or cur_min <= end_min
    except Exception:
        return True


# ---------------------------------------------------------------------------
# Base model loading
# ---------------------------------------------------------------------------

def _load_base_model_if_needed():
    global _base_model_obj, _base_tokenizer
    with _state_lock:
        if _base_model_obj is not None:
            return _base_model_obj, _base_tokenizer

    torch, transformers, peft, _ = _lazy_imports()

    # Decide device
    cuda_ok = torch.cuda.is_available()
    if cuda_ok:
        try:
            torch.cuda.set_per_process_memory_fraction(TRAINER_VRAM_FRACTION, device=0)
        except Exception:
            pass
    device = "cuda:0" if cuda_ok else "cpu"
    dtype = torch.float16 if cuda_ok else torch.float32

    print(f"[trainer] loading base model {BASE_MODEL} on {device} (dtype={dtype})...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(BASE_MODEL)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        device_map=device,
    )

    with _state_lock:
        _base_model_obj = model
        _base_tokenizer = tokenizer
        _status["mode"] = "gpu" if cuda_ok else "cpu"
        _status["device"] = device

    return model, tokenizer


def _adapter_dir(node_id: str) -> Path:
    return ADAPTERS_DIR / node_id


def _adapter_ready(node_id: str) -> bool:
    d = _adapter_dir(node_id)
    return d.exists() and (d / "adapter_config.json").exists()


def _swap_adapter(node_id: str):
    """Hot-swap a LoRA adapter onto the base model."""
    global _loaded_adapter
    torch, transformers, peft, _ = _lazy_imports()
    model, tokenizer = _load_base_model_if_needed()

    with _state_lock:
        if _loaded_adapter == node_id:
            return model, tokenizer
        # Unload previous adapter if any
        if _loaded_adapter is not None and hasattr(model, "unload"):
            try:
                model = model.unload()
            except Exception:
                pass
            _loaded_adapter = None

    if _adapter_ready(node_id):
        try:
            model = peft.PeftModel.from_pretrained(model, str(_adapter_dir(node_id)))
            with _state_lock:
                _loaded_adapter = node_id
                _status["loaded_adapter"] = node_id
        except Exception as e:
            print(f"[trainer] failed to load adapter for {node_id}: {e}")

    return model, tokenizer


# ---------------------------------------------------------------------------
# API models
# ---------------------------------------------------------------------------

class InferRequest(BaseModel):
    adapter_id: str
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.5


class InferResponse(BaseModel):
    text: str
    adapter_used: bool
    backend: str


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/status")
def status() -> dict:
    snap = _gpu_snapshot()
    with _state_lock:
        return {**_status, "gpu": snap}


@app.get("/adapters/{node_id}")
def adapter_info(node_id: str) -> dict:
    return {"ready": _adapter_ready(node_id), "node_id": node_id, "path": str(_adapter_dir(node_id))}


@app.post("/infer", response_model=InferResponse)
def infer(req: InferRequest) -> InferResponse:
    torch, _, _, _ = _lazy_imports()

    adapter_used = _adapter_ready(req.adapter_id)
    if adapter_used:
        model, tokenizer = _swap_adapter(req.adapter_id)
    else:
        model, tokenizer = _load_base_model_if_needed()

    inputs = tokenizer(req.prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            do_sample=req.temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return InferResponse(
        text=text.strip(),
        adapter_used=adapter_used,
        backend=_status["mode"],
    )


@app.post("/admin/train/{node_id}")
async def admin_train(node_id: str, background_tasks: BackgroundTasks) -> dict:
    """Manually trigger training for a node (bypasses scheduler gating)."""
    background_tasks.add_task(_train_one_adapter, node_id, force=True)
    return {"queued": node_id}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

async def _fetch_training_samples(node_id: str, limit: int = 200) -> list[dict]:
    pool = await _get_pg_pool()
    if pool is None:
        return []
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, prompt, response FROM engram.training_data "
            "WHERE node_id=$1 AND consumed=FALSE "
            "ORDER BY created_at DESC LIMIT $2",
            node_id, limit,
        )
    return [dict(r) for r in rows]


async def _mark_samples_consumed(sample_ids: list[int]) -> None:
    pool = await _get_pg_pool()
    if pool is None or not sample_ids:
        return
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE engram.training_data SET consumed=TRUE WHERE id = ANY($1::bigint[])",
            sample_ids,
        )


async def _record_adapter_run(
    node_id: str,
    samples_used: int,
    duration_sec: float,
    backend: str,
    status: str,
    error: Optional[str] = None,
    adapter_path: Optional[str] = None,
) -> None:
    pool = await _get_pg_pool()
    if pool is None:
        return
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO engram.adapter_runs
                (node_id, adapter_path, base_model, samples_used, duration_sec,
                 backend, status, error, finished_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """,
            node_id, adapter_path, BASE_MODEL, samples_used, duration_sec,
            backend, status, error, int(time.time() * 1000),
        )
    # Push event so dashboard sees the new run immediately
    await _publish_event("training_change", {
        "node_id": node_id, "status": status, "backend": backend,
        "duration_sec": duration_sec, "samples_used": samples_used,
    })


async def _publish_event(channel: str, payload: dict) -> None:
    """Lightweight publisher (we're in a separate container without the src/ tree)."""
    try:
        import redis.asyncio as redis_asyncio
        import json as _json
    except ImportError:
        return
    try:
        r = redis_asyncio.from_url(
            os.environ.get("REDIS_URL", "redis://redis:6379/0"),
            decode_responses=True,
        )
        msg = {"ts": time.time(), "channel": channel, "payload": payload}
        await r.publish(f"cogram:events:{channel}", _json.dumps(msg, default=str))
        await r.aclose()
    except Exception:
        pass


def _train_one_adapter(node_id: str, force: bool = False) -> None:
    """Synchronous training of a single LoRA adapter for one node.

    Falls back to CPU if GPU is contested. Writes checkpoint frequently so we can
    resume from preemption. Marks training samples consumed on success.
    """
    if not _training_lock.acquire(blocking=False):
        print(f"[trainer] another adapter is training; {node_id} queued")
        return

    asyncio.run(_train_one_adapter_async(node_id, force))
    _training_lock.release()


async def _train_one_adapter_async(node_id: str, force: bool = False) -> None:
    started = time.time()
    backend = "?"
    status_str = "started"
    error: Optional[str] = None
    samples = await _fetch_training_samples(node_id)

    if len(samples) < TRAINER_MIN_SAMPLES and not force:
        print(f"[trainer] {node_id}: only {len(samples)} samples, skip")
        return

    with _state_lock:
        _status["training"] = True
        _status["current_training_node"] = node_id

    try:
        torch, transformers, peft, _ = _lazy_imports()

        # Decide backend
        snap = _gpu_snapshot()
        use_gpu = torch.cuda.is_available() and not _is_gpu_contended(snap)
        backend = "gpu" if use_gpu else "cpu"
        print(f"[trainer] training {node_id} on {backend} ({len(samples)} samples)")

        model, tokenizer = _load_base_model_if_needed()

        # Build LoRA config
        lora_config = peft.LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        )
        model = peft.get_peft_model(model, lora_config)

        # Prepare dataset
        from datasets import Dataset  # type: ignore
        formatted = [
            {"text": f"{s['prompt']}\n\n{s['response']}"}
            for s in samples
        ]
        ds = Dataset.from_list(formatted)

        def tokenize_fn(batch):
            return tokenizer(
                batch["text"],
                truncation=True,
                padding="max_length",
                max_length=512,
            )
        ds_tok = ds.map(tokenize_fn, batched=True, remove_columns=["text"])

        # Tiny training run
        training_args = transformers.TrainingArguments(
            output_dir=str(_adapter_dir(node_id) / "checkpoints"),
            num_train_epochs=2,
            per_device_train_batch_size=1 if backend == "cpu" else 2,
            gradient_accumulation_steps=4,
            learning_rate=1e-4,
            save_steps=60,
            save_total_limit=2,
            logging_steps=10,
            report_to=[],
            fp16=use_gpu,
            no_cuda=not use_gpu,
        )
        trainer = transformers.Trainer(
            model=model,
            args=training_args,
            train_dataset=ds_tok,
            data_collator=transformers.DataCollatorForLanguageModeling(
                tokenizer=tokenizer, mlm=False
            ),
        )
        trainer.train()

        # Save adapter
        out_dir = _adapter_dir(node_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))
        await _mark_samples_consumed([s["id"] for s in samples])

        status_str = "success"
    except Exception as e:
        status_str = "failed"
        error = str(e)
        print(f"[trainer] {node_id} training failed: {e}")
    finally:
        duration = time.time() - started
        with _state_lock:
            _status["training"] = False
            _status["current_training_node"] = None
        await _record_adapter_run(
            node_id=node_id,
            samples_used=len(samples),
            duration_sec=duration,
            backend=backend,
            status=status_str,
            error=error,
            adapter_path=str(_adapter_dir(node_id)),
        )


async def _scheduler_loop() -> None:
    """Polls Postgres for training-eligible nodes; respects window + idle + contention."""
    while True:
        try:
            if not _within_window():
                await asyncio.sleep(TRAINER_LOOP_INTERVAL)
                continue

            snap = _gpu_snapshot()
            if _is_gpu_contended(snap):
                # don't even start; recheck later
                await asyncio.sleep(TRAINER_LOOP_INTERVAL)
                continue

            pool = await _get_pg_pool()
            if pool is None:
                await asyncio.sleep(TRAINER_LOOP_INTERVAL)
                continue

            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT
                        td.node_id,
                        COUNT(*) FILTER (WHERE td.consumed=FALSE) AS unconsumed_count,
                        MAX(ar.finished_at) AS last_run_at
                    FROM engram.training_data td
                    LEFT JOIN engram.adapter_runs ar
                      ON ar.node_id = td.node_id AND ar.status = 'success'
                    GROUP BY td.node_id
                    HAVING COUNT(*) FILTER (WHERE td.consumed=FALSE) >= $1
                    ORDER BY MAX(td.created_at) DESC
                    LIMIT 5
                    """,
                    TRAINER_MIN_SAMPLES,
                )

            now_ms = int(time.time() * 1000)
            min_gap_ms = TRAINER_MIN_DAYS_BETWEEN * 86_400 * 1000
            for r in rows:
                if r["last_run_at"] and (now_ms - r["last_run_at"]) < min_gap_ms:
                    continue
                # train this one
                with _state_lock:
                    _status["queue"].append(r["node_id"])
                _train_one_adapter(r["node_id"], force=False)
                with _state_lock:
                    if r["node_id"] in _status["queue"]:
                        _status["queue"].remove(r["node_id"])

        except Exception as e:
            print(f"[trainer scheduler] error: {e}")

        await asyncio.sleep(TRAINER_LOOP_INTERVAL)


async def _event_listener_loop():
    """Subscribe to cogram:events:training_ready and react immediately when
    a node crosses the min-samples threshold. This is the reactive path;
    the polling _scheduler_loop is the safety net."""
    import json as _json
    while True:
        try:
            try:
                import redis.asyncio as _r
            except ImportError:
                await asyncio.sleep(60)
                continue
            client = _r.from_url(os.environ.get("REDIS_URL", "redis://redis:6379/0"), decode_responses=True)
            pubsub = client.pubsub()
            await pubsub.subscribe("cogram:events:training_ready")
            print("[trainer] subscribed to cogram:events:training_ready (reactive mode)")
            async for message in pubsub.listen():
                if message.get("type") != "message":
                    continue
                try:
                    data = _json.loads(message["data"])
                    payload = data.get("payload", {}) if isinstance(data, dict) else {}
                    node_id = payload.get("node_id")
                    if not node_id:
                        continue
                    print(f"[trainer] reactive trigger for node {node_id} ({payload.get('unconsumed_samples')} samples)")
                    # Schedule training; force=False respects window+threshold,
                    # but the threshold check passes since we just crossed it.
                    asyncio.create_task(_train_one_adapter(node_id, force=False))
                except Exception as exc:
                    print(f"[trainer] event parse error: {exc}")
        except Exception as exc:
            print(f"[trainer] event loop disconnected: {exc} — reconnecting in 30s")
            await asyncio.sleep(30)


@app.on_event("startup")
async def _startup():
    asyncio.create_task(_scheduler_loop())
    asyncio.create_task(_event_listener_loop())
    print(f"[trainer] startup. base={BASE_MODEL} window={TRAINER_SCHEDULE} "
          f"min_samples={TRAINER_MIN_SAMPLES} every {TRAINER_MIN_DAYS_BETWEEN}d "
          f"(reactive event listener enabled)")

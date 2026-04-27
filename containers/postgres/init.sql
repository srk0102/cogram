-- Cogram Postgres schema. Runs once at first container start.
-- Compatible with Engram TS package's `engram` schema where it overlaps,
-- plus extra tables for trainer logs and audit.

CREATE SCHEMA IF NOT EXISTS engram;

-- ============================================================================
-- engram.patterns — the decision cache.
-- Mirrors Engram TS package layout where columns overlap; we store our LLM
-- response (entity extraction, dedup judgement, summary, narration) in `meta`
-- as JSONB. `decision` enum is wide enough to fit our call kinds.
-- ============================================================================
DO $$ BEGIN
    CREATE TYPE engram.decision_kind AS ENUM (
        'allow', 'block', 'flag', 'fraud', 'bot', 'churn_risk',
        'llm_response', 'embedding', 'narration', 'dedup', 'extract'
    );
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

CREATE TABLE IF NOT EXISTS engram.patterns (
    id              TEXT PRIMARY KEY,
    namespace       TEXT NOT NULL DEFAULT 'cogram',
    fingerprint     JSONB NOT NULL DEFAULT '{}',
    context_shape   JSONB,
    decision        engram.decision_kind NOT NULL DEFAULT 'llm_response',
    reason          JSONB,
    confidence      DOUBLE PRECISION NOT NULL DEFAULT 1.0
                        CHECK (confidence >= 0 AND confidence <= 1),
    hit_count       INTEGER NOT NULL DEFAULT 0,
    success_count   INTEGER NOT NULL DEFAULT 0,
    failure_count   INTEGER NOT NULL DEFAULT 0,
    meta            JSONB NOT NULL DEFAULT '{}',
    created_at      BIGINT NOT NULL DEFAULT (EXTRACT(EPOCH FROM NOW())::BIGINT * 1000),
    updated_at      BIGINT NOT NULL DEFAULT (EXTRACT(EPOCH FROM NOW())::BIGINT * 1000),
    expires_at      BIGINT
);

CREATE INDEX IF NOT EXISTS engram_patterns_ns ON engram.patterns(namespace);
CREATE INDEX IF NOT EXISTS engram_patterns_ns_dec ON engram.patterns(namespace, decision);
CREATE INDEX IF NOT EXISTS engram_patterns_ns_updated ON engram.patterns(namespace, updated_at DESC);
CREATE INDEX IF NOT EXISTS engram_patterns_expires ON engram.patterns(expires_at)
    WHERE expires_at IS NOT NULL;

-- ============================================================================
-- engram.audit — every cache event for debugging and stats
-- ============================================================================
CREATE TABLE IF NOT EXISTS engram.audit (
    id          BIGSERIAL PRIMARY KEY,
    pattern_id  TEXT,
    namespace   TEXT NOT NULL,
    decision    engram.decision_kind NOT NULL,
    source      TEXT NOT NULL,                    -- 'hit' | 'miss' | 'evict' | 'feedback'
    fingerprint JSONB,
    created_at  BIGINT NOT NULL DEFAULT (EXTRACT(EPOCH FROM NOW())::BIGINT * 1000)
);
CREATE INDEX IF NOT EXISTS engram_audit_ns ON engram.audit(namespace, created_at DESC);

-- ============================================================================
-- engram.training_data — captured (input, big-LLM-response) pairs for trainer
-- ============================================================================
CREATE TABLE IF NOT EXISTS engram.training_data (
    id          BIGSERIAL PRIMARY KEY,
    node_id     TEXT NOT NULL,                    -- entity uuid this sample is for
    prompt      TEXT NOT NULL,
    response    TEXT NOT NULL,
    quality     DOUBLE PRECISION,                 -- optional: validation score
    consumed    BOOLEAN NOT NULL DEFAULT FALSE,   -- set TRUE after a LoRA used it
    created_at  BIGINT NOT NULL DEFAULT (EXTRACT(EPOCH FROM NOW())::BIGINT * 1000)
);
CREATE INDEX IF NOT EXISTS engram_training_node ON engram.training_data(node_id, consumed);

-- ============================================================================
-- engram.adapter_runs — LoRA training events log
-- ============================================================================
CREATE TABLE IF NOT EXISTS engram.adapter_runs (
    id              BIGSERIAL PRIMARY KEY,
    node_id         TEXT NOT NULL,
    adapter_path    TEXT,
    base_model      TEXT NOT NULL,
    samples_used    INTEGER NOT NULL,
    duration_sec    DOUBLE PRECISION,
    validation_score DOUBLE PRECISION,
    backend         TEXT NOT NULL,                -- 'gpu' | 'cpu'
    status          TEXT NOT NULL,                -- 'started' | 'success' | 'failed' | 'preempted'
    error           TEXT,
    started_at      BIGINT NOT NULL DEFAULT (EXTRACT(EPOCH FROM NOW())::BIGINT * 1000),
    finished_at     BIGINT
);
CREATE INDEX IF NOT EXISTS engram_adapter_runs_node ON engram.adapter_runs(node_id, started_at DESC);

-- ============================================================================
-- engram.daily_spend — running total of LLM API costs per day (for safety cap)
-- ============================================================================
CREATE TABLE IF NOT EXISTS engram.daily_spend (
    day             DATE PRIMARY KEY,
    calls           INTEGER NOT NULL DEFAULT 0,
    tokens_in       BIGINT NOT NULL DEFAULT 0,
    tokens_out      BIGINT NOT NULL DEFAULT 0,
    cost_usd        DOUBLE PRECISION NOT NULL DEFAULT 0
);

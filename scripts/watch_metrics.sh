#!/usr/bin/env bash
# Cogram live metrics watcher
#
# Polls Neo4j, Postgres (Engram), Redis, cogram-mcp, cogram-trainer, ollama
# every 5s. Prints a compact A-Z snapshot per cycle so you can watch how cogram
# performs as you write episodes from Claude Desktop.
#
# Usage:
#   bash scripts/watch_metrics.sh        # uses defaults
#   INTERVAL=3 bash scripts/watch_metrics.sh
#
# Tracks:
#   GRAPH      Neo4j node/edge counts, intent_meta, vllm_narrative, profile, knots
#   ENGRAM     Postgres engram.patterns rows, audit rows, training_data rows
#   REDIS      memory used, key count, keyspace_hits/misses, active subgraph keys
#   DOCKER     CPU + memory per container
#   PIPELINE   recent OpenAI calls + Gemma calls from cogram-mcp logs
#   OLLAMA     model loaded? gemma response time?
#   COST       running OpenAI spend estimate based on call count

set -u
INTERVAL="${INTERVAL:-5}"
NEO4J_PASS="${NEO4J_PASSWORD:-password}"
PG_USER="${POSTGRES_USER:-cogram}"
PG_DB="${POSTGRES_DB:-cogram}"

# rough per-call cost estimates (gpt-4o-mini Nov 2024 pricing)
COST_PER_CALL_CENTS=0.05    # ~$0.0005 per call (input+output, average)

clear

while true; do
  TS=$(date +%H:%M:%S)
  echo "=================================================================="
  echo "[$TS] cogram metrics — refresh every ${INTERVAL}s, Ctrl+C to stop"
  echo "=================================================================="

  # ---------- GRAPH (Neo4j) ----------
  echo "GRAPH (Neo4j):"
  docker exec cogram-neo4j cypher-shell -u neo4j -p "$NEO4J_PASS" --format plain "
    MATCH (e:Episodic) WITH count(e) AS ep
    MATCH (n:Entity) WITH ep, count(n) AS ent
    OPTIONAL MATCH ()-[r:RELATES_TO]->()
      WITH ep, ent, count(r) AS edges,
           sum(CASE WHEN r.intent_meta IS NOT NULL THEN 1 ELSE 0 END) AS annotated,
           sum(CASE WHEN r.retracted_at IS NOT NULL THEN 1 ELSE 0 END) AS retracted
    OPTIONAL MATCH (en:Entity) WHERE en.vllm_narrative IS NOT NULL
      WITH ep, ent, edges, annotated, retracted, count(en) AS narrated
    OPTIONAL MATCH (k:Entity) WHERE k.knot_narrative IS NOT NULL
      WITH ep, ent, edges, annotated, retracted, narrated, count(k) AS knots
    OPTIONAL MATCH (p:DirectorProfile)
      WITH ep, ent, edges, annotated, retracted, narrated, knots, count(p) AS profiles
    OPTIONAL MATCH (pat:CognitivePattern)
      RETURN ep, ent, edges, annotated, retracted, narrated, knots, profiles, count(pat) AS patterns
  " 2>/dev/null | tail -n +2 | head -1 | awk '{
    printf "  episodes=%s  entities=%s  edges=%s  annotated=%s/%s  narrated=%s  knots=%s  profile=%s  patterns=%s",
      $1, $2, $3, $4, $3, $6, $7, $8, $9
    if ($5 > 0) printf "  retracted=%s", $5
    printf "\n"
  }' || echo "  (neo4j unreachable)"

  # ---------- ENGRAM (Postgres) ----------
  echo "ENGRAM (Postgres):"
  PG_OUT=$(docker exec cogram-postgres psql -U "$PG_USER" -d "$PG_DB" -t -A -F"|" -c "
    SELECT
      (SELECT count(*) FROM engram.patterns)         AS patterns,
      (SELECT count(*) FROM engram.audit)            AS audits,
      (SELECT count(*) FROM engram.training_data)   AS train,
      (SELECT count(*) FROM engram.training_data WHERE consumed=false) AS train_unconsumed,
      (SELECT count(*) FROM engram.adapter_runs)    AS adapters
  " 2>/dev/null | head -1)
  if [[ -n "$PG_OUT" ]]; then
    IFS='|' read -r p a t tu adp <<< "$PG_OUT"
    echo "  patterns=$p  audits=$a  training_samples=$t (unconsumed=$tu)  adapters=$adp"
  else
    echo "  (postgres unreachable)"
  fi

  # ---------- REDIS ----------
  echo "REDIS:"
  RD_MEM=$(docker exec cogram-redis redis-cli INFO memory 2>/dev/null | grep "^used_memory_human:" | cut -d: -f2 | tr -d '\r ')
  RD_KEYS=$(docker exec cogram-redis redis-cli DBSIZE 2>/dev/null | awk '{print $NF}')
  RD_HITS=$(docker exec cogram-redis redis-cli INFO stats 2>/dev/null | grep "^keyspace_hits:" | cut -d: -f2 | tr -d '\r ')
  RD_MISS=$(docker exec cogram-redis redis-cli INFO stats 2>/dev/null | grep "^keyspace_misses:" | cut -d: -f2 | tr -d '\r ')
  RD_ACTIVE=$(docker exec cogram-redis redis-cli --raw KEYS "cogram:session:*:active" 2>/dev/null | wc -l | tr -d ' ')
  RD_KNOTS=$(docker exec cogram-redis redis-cli --raw KEYS "cogram:knot:*:narrative" 2>/dev/null | wc -l | tr -d ' ')
  echo "  memory=${RD_MEM:-?}  keys=${RD_KEYS:-0}  hits=${RD_HITS:-0}  misses=${RD_MISS:-0}  active_subgraphs=${RD_ACTIVE}  cached_knots=${RD_KNOTS}"

  # ---------- DOCKER stats (CPU + memory per container) ----------
  echo "DOCKER:"
  docker stats --no-stream --format "  {{.Name}}: CPU={{.CPUPerc}}  MEM={{.MemUsage}}" 2>/dev/null | grep -E "cogram" | head -8

  # ---------- PIPELINE (recent OpenAI + Gemma calls from cogram-mcp logs) ----------
  echo "PIPELINE (last 60s):"
  OAI_CALLS=$(docker logs cogram-mcp --since 60s 2>&1 | grep -c "POST.*api.openai.com.*200" || echo 0)
  GEMMA_CALLS=$(docker logs cogram-mcp --since 60s 2>&1 | grep -c "host.docker.internal:11434" || echo 0)
  TOOL_CALLS=$(docker logs cogram-mcp --since 60s 2>&1 | grep -c "Processing request of type" || echo 0)
  EMB_CALLS=$(docker logs cogram-mcp --since 60s 2>&1 | grep -c "openai.com/v1/embeddings.*200" || echo 0)
  CHAT_CALLS=$(docker logs cogram-mcp --since 60s 2>&1 | grep -c "openai.com/v1/chat/completions.*200" || echo 0)
  echo "  mcp_tool_calls=$TOOL_CALLS  openai_chat=$CHAT_CALLS  openai_embed=$EMB_CALLS  gemma_calls=$GEMMA_CALLS"

  # ---------- OLLAMA (host) ----------
  OLLAMA_LOADED=$(curl -sf http://localhost:11434/api/ps 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(','.join(m['name'] for m in d.get('models',[])) or 'none-loaded')" 2>/dev/null)
  echo "OLLAMA: loaded=${OLLAMA_LOADED:-unknown}"

  # ---------- COST estimate ----------
  TOTAL_OAI=$((CHAT_CALLS + EMB_CALLS))
  COST_CENTS=$(awk "BEGIN { printf \"%.4f\", $TOTAL_OAI * $COST_PER_CALL_CENTS }")
  echo "COST (last 60s window): openai_calls=$TOTAL_OAI  est=\$0.${COST_CENTS}  gemma=\$0 (local)"

  echo ""
  sleep "$INTERVAL"
  clear
done

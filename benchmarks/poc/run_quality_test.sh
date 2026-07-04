#!/usr/bin/env bash
# Run quality_gsm8k.py (gsm8k generation under parallel PoC) across a
# batch_size × poc_requests × max_tokens matrix, baseline-first.
#
# This is the "inference co-existence" check: does real generation quality
# (gsm8k) hold while DECODE PoC runs in parallel?
#
# Order of runs:
#   1. Baseline (--disable_poc)  — gsm8k, no PoC
#   2. For each batch_size × poc_requests × max_tokens — gsm8k + parallel PoC
#
# Requires: lm-eval installed in the active venv. Decode is controlled
# per-request via max_tokens (no --poc-decode server flag needed).
# Server lifecycle delegated to quality_gsm8k.py:
#   - SERVER_HOST + SERVER_PORT → connect to an existing server.
#   - omit SERVER_PORT (+ SERVER_ARGS) → auto-launch per run.
#
# Environment variables
# ---------------------
# PYTHON        python with vllm+lm-eval (default: python from the active venv)
# MODEL         HF model id (default: RedHatAI/Qwen2.5-7B-Instruct-quantized.w8a16)
# SERVER_HOST   default 127.0.0.1
# SERVER_PORT   existing server port; omit to auto-launch
# SERVER_ARGS   extra 'vllm serve' flags (default: "--gpu-memory-utilization 0.9 --max-model-len 4096")
# TASKS         lm-eval task(s) (default: gsm8k)
# BATCH_SIZES   default "8 16"
# POC_REQUESTS  default "1 4"
# MAX_TOKENS    decode steps for PoC (default "256"; the proposal's purpose). Space-separated to sweep, e.g. "0 256".
#
# Examples
#   bash benchmarks/poc/run_quality_test.sh                      # decode PoC (256), auto-launch
#   SERVER_PORT=8103 MAX_TOKENS="0 256" bash benchmarks/poc/run_quality_test.sh   # prefill vs decode on existing server

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
EVAL_SCRIPT="$SCRIPT_DIR/quality_gsm8k.py"

MODEL="${MODEL:-RedHatAI/Qwen2.5-7B-Instruct-quantized.w8a16}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-}"
SERVER_ARGS="${SERVER_ARGS:---gpu-memory-utilization 0.9 --max-model-len 4096}"
TASKS="${TASKS:-gsm8k}"

read -ra BATCH_SIZES  <<< "${BATCH_SIZES:-8 16}"
read -ra POC_REQUESTS <<< "${POC_REQUESTS:-1 4}"
read -ra MAX_TOKENS   <<< "${MAX_TOKENS:-256}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
STORAGE_DIR="$SCRIPT_DIR/storage/eval_results"
LOG_DIR="$SCRIPT_DIR/storage/logs/$TIMESTAMP"
mkdir -p "$STORAGE_DIR" "$LOG_DIR"

SERVER_FORWARD=()
if [ -n "$SERVER_PORT" ]; then
    SERVER_FORWARD=(--host "$SERVER_HOST" --port "$SERVER_PORT")
else
    SERVER_FORWARD=(--server-args "$SERVER_ARGS")
fi

log() { echo "[$(date +%H:%M:%S)] $*"; }
PASS=0; FAIL=0

_run() {
    local label="$1"; shift
    local log_file="$LOG_DIR/${label}.log"
    log "[$label] Starting ..."
    if "$PYTHON" "$EVAL_SCRIPT" \
            --model_name "$MODEL" --output_path "$STORAGE_DIR" --tasks "$TASKS" \
            "${SERVER_FORWARD[@]}" "$@" 2>&1 | tee "$log_file"; then
        log "[$label] Done."; PASS=$((PASS+1))
    else
        log "[$label] FAILED."; FAIL=$((FAIL+1))
    fi
}

log "Model      : $MODEL"
log "Tasks      : $TASKS"
log "Batches    : ${BATCH_SIZES[*]}"
log "PoC reqs   : ${POC_REQUESTS[*]}"
log "Decode mt  : ${MAX_TOKENS[*]}"
log "Server     : ${SERVER_PORT:+$SERVER_HOST:$SERVER_PORT}${SERVER_PORT:-auto-launch}"
echo ""

# 1) baseline (no PoC)
_run "baseline_bs${BATCH_SIZES[0]}" --batch_size "${BATCH_SIZES[0]}" --disable_poc

# 2) PoC-enabled matrix (DECODE by default)
for bs in "${BATCH_SIZES[@]}"; do
  for poc in "${POC_REQUESTS[@]}"; do
    for mt in "${MAX_TOKENS[@]}"; do
      _run "bs${bs}_poc${poc}_mt${mt}" \
          --batch_size "$bs" --poc_requests "$poc" --max_tokens "$mt"
    done
  done
done

echo ""
log "Comparison table ..."
"$PYTHON" "$EVAL_SCRIPT" --table-only --output_path "$STORAGE_DIR" \
    --table-output "$STORAGE_DIR/results_table.md"

echo ""
printf "Runs passed: %s  failed: %s\n" "$PASS" "$FAIL"
echo "Results: $STORAGE_DIR/results_table.md"
[ "$FAIL" -eq 0 ]

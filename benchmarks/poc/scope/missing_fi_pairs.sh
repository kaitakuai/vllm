#!/usr/bin/env bash
# ONE-OFF: generate ONLY the 2 missing FlashInfer pairs into an EXISTING session
# (the 3 FlashAttn pairs are already there), then re-render. Validator = honest @
# cg-flashattn (fast); only the 2 FI *reference* gens are slow.
set -uo pipefail
POC=$(cd "$HERE/.." && pwd)
PY=$PY
export PATH=$(dirname "$PY"):$PATH
export HF_TOKEN="$(cat ~/.cache/huggingface/token)"
H=allenai/OLMoE-1B-7B-0924-Instruct
F=nm-testing/OLMoE-1B-7B-0924-Instruct-FP8
D="$1"; PORT=8203; URL="http://127.0.0.1:$PORT"
N=128; MT=256; SEQ=64

SRV=""
boot(){ local model="$1" extra="$2"
  ( cd /tmp && exec setsid "$PY" -m vllm.entrypoints.openai.api_server --model "$model" --port "$PORT" \
      --poc-decode --gpu-memory-utilization 0.8 --max-model-len 1024 --trust-remote-code $extra ) \
      > "$D/serve_missing_$(echo "$model$extra"|tr -cd 'A-Za-z0-9').log" 2>&1 &
  SRV=$!
  for i in $(seq 1 120); do
    s=$(curl -s "$URL/v1/models" 2>/dev/null|"$PY" -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null||true)
    [ "$s" = "$model" ] && { echo "[boot ok] $model $extra"; return 0; }
    ps -p "$SRV" >/dev/null 2>&1 || { echo "[BOOT DIED] $model $extra"; return 1; }
    sleep 8
  done; echo "[BOOT TIMEOUT] $model $extra"; return 1; }
kill_srv(){ [ -n "$SRV" ] && kill -- "-$SRV" 2>/dev/null; sleep 4; SRV=""; }

# 1. honest bf16 @ FlashInfer  -> H(cg,FI) reference  (the slow one)
boot "$H" "--attention-backend FLASHINFER" && \
  $PY "$POC/collect.py" --mode generate --model "$H" --profile cg-flashinfer --url "$URL" \
     --nonces $N --max-tokens $MT --seq-len $SEQ --save "$D/gen_honest_cg-flashinfer.json"
kill_srv
# 2. fraud FP8 @ FlashInfer  -> F(cg,FI) reference
boot "$F" "--attention-backend FLASHINFER" && \
  $PY "$POC/collect.py" --mode generate --model "$F" --profile cg-flashinfer --url "$URL" \
     --nonces $N --max-tokens $MT --seq-len $SEQ --save "$D/gen_fraud_cg-flashinfer.json"
kill_srv
# 3. validator honest @ cg-flashattn  -> validate vs the 2 new FI refs (fast)
boot "$H" "--attention-backend FLASH_ATTN" && {
  for ref in gen_honest_cg-flashinfer gen_fraud_cg-flashinfer; do
    $PY "$POC/collect.py" --mode validate --model "$H" --profile cg-flashattn --url "$URL" \
       --ref "$D/$ref.json" --save "$D/val_cg-flashattn__$ref.json"
  done
}
kill_srv
# 4. re-render with all 5 pairs (SINGLE simplified report)
$PY "$(dirname "$0")/simplify_report.py" "$D" --out "$D/report.html"
echo "DONE: $D/report.html"

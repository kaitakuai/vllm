#!/usr/bin/env bash
# Re-run ONLY GSM8K co-existence into an EXISTING report session, then re-render + re-push.
# Use after a TOOLING fix (e.g. the missing lm_eval[api]/tenacity dep that made lm-eval's API
# model record 0.0 accuracy) — separation/perf are untouched, only GSM8K is redone.
#   rerun_gsm.sh <honest-model> <existing-session-dir>
#   env: TP GMU GSMN GSM_MML EXTRA PROF SCOPE_COMMIT PUSH
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; REPO="$(cd "$HERE/../../.." && pwd)"
PY="${POC_PY:-$REPO/.venv/bin/python}"; command -v "$PY" >/dev/null 2>&1 || PY="$(command -v python3)"
POC="$(cd "$HERE/.." && pwd)"
HONEST="$1"; OUT="$2"; SESS="$(basename "$OUT")"
TP="${TP:-4}"; GMU="${GMU:-0.90}"; GSMN="${GSMN:-100}"; GSM_MML="${GSM_MML:-2048}"
EXTRA="${EXTRA:-}"; PROF="${PROF:-cg-flashattn}"; PUSH="${PUSH:-1}"; PORT=8200; URL="http://127.0.0.1:$PORT"
[ -d "$OUT" ] || { echo "no session dir: $OUT"; exit 1; }

echo "== boot honest @ $PROF (mml $GSM_MML) for GSM8K =="
( cd /tmp && exec setsid "$PY" -m vllm.entrypoints.openai.api_server --model "$HONEST" --port "$PORT" \
    --poc-decode --gpu-memory-utilization "$GMU" --max-model-len "$GSM_MML" --tensor-parallel-size "$TP" \
    --trust-remote-code $EXTRA --attention-backend FLASH_ATTN ) > "$OUT/serve_gsm_rerun.log" 2>&1 &
SRV_PID=$!
for i in $(seq 1 120); do
  s=$(curl -s "$URL/v1/models" 2>/dev/null | "$PY" -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || true)
  [ "$s" = "$HONEST" ] && { echo "  [boot ok]"; break; }
  ps -p "$SRV_PID" >/dev/null 2>&1 || { echo "  [BOOT DIED]"; tail -20 "$OUT/serve_gsm_rerun.log"; exit 1; }
  sleep 8
done

echo "== GSM8K on (PoC load) + off (baseline) =="
"$PY" "$POC/quality_gsm8k.py" --model_name "$HONEST" --profile "$PROF" --url "$URL" --batch_size 16 \
   --limit "$GSMN" --output_path "$OUT/gsm_${PROF}_on"  --save "$OUT/gsm_${PROF}_on.json"  || echo "  gsm on FAILED"
"$PY" "$POC/quality_gsm8k.py" --model_name "$HONEST" --profile "$PROF" --url "$URL" --batch_size 16 \
   --limit "$GSMN" --disable_poc --output_path "$OUT/gsm_${PROF}_off" --save "$OUT/gsm_${PROF}_off.json" || echo "  gsm off FAILED"

kill -- "-$SRV_PID" 2>/dev/null; sleep 3
echo "== re-render + re-push =="
POC_SCOPE_COMMIT="${SCOPE_COMMIT:-}" "$PY" "$HERE/simplify_report.py" "$OUT" --out "$OUT/report.html" || echo "render FAILED"
if [ "$PUSH" = 1 ]; then "$PY" "$HERE/inject_s3.py" "$OUT/report.html" "$SESS"; bash "$HERE/s3.sh" push-report "$OUT" "$SESS"; fi
echo "=== GSM RERUN DONE: $OUT/report.html ==="

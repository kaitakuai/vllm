#!/usr/bin/env bash
# EARLY separation signal — minimal 2-boot floor+fraud check (no FI/eager/gsm) so you see
# honest-floor vs fraud-rate ASAP after a download, before committing to the full run.
#   early_sep.sh <honest> <fraud> <out-dir>   env: N TP GMU MML EXTRA
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; REPO="$(cd "$HERE/../../.." && pwd)"
PY="${POC_PY:-$REPO/.venv/bin/python}"; command -v "$PY" >/dev/null 2>&1 || PY="$(command -v python3)"
POC="$(cd "$HERE/.." && pwd)"
HONEST="$1"; FRAUD="$2"; OUT="$3"; mkdir -p "$OUT"
N="${N:-16}"; TP="${TP:-4}"; GMU="${GMU:-0.95}"; MML="${MML:-1024}"; EXTRA="${EXTRA:-}"; PORT=8200; URL="http://127.0.0.1:$PORT"
SRV=""
boot(){ local m="$1"; ( cd /tmp && exec setsid "$PY" -m vllm.entrypoints.openai.api_server --model "$m" --port "$PORT" \
    --poc-decode --gpu-memory-utilization "$GMU" --max-model-len "$MML" --tensor-parallel-size "$TP" --trust-remote-code \
    $EXTRA --attention-backend FLASH_ATTN ) > "$OUT/serve_early_$(echo "$m"|tr /: __).log" 2>&1 &
  SRV=$!
  for i in $(seq 1 150); do
    s=$(curl -s "$URL/v1/models" 2>/dev/null | "$PY" -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || true)
    [ "$s" = "$m" ] && { echo "  [boot ok] $m"; return 0; }
    ps -p "$SRV" >/dev/null 2>&1 || { echo "  [BOOT DIED] $m"; tail -25 "$OUT/serve_early_$(echo "$m"|tr /: __).log"; return 1; }
    sleep 8
  done; echo "  [BOOT TIMEOUT] $m"; return 1; }
kill_srv(){ # reap the server AND wait for GPU VRAM to actually free (else next boot OOMs)
  [ -n "$SRV" ] && kill -- "-$SRV" 2>/dev/null
  for i in $(seq 1 30); do ps -p "${SRV:-0}" >/dev/null 2>&1 || break; sleep 1; done
  pkill -9 -f "vllm.entrypoints" 2>/dev/null
  for i in $(seq 1 20); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1); [ "${u:-99999}" -lt 2000 ] && break; sleep 2; done
  SRV=""; }
gen(){ "$PY" "$POC/collect.py" --mode generate --model "$1" --profile cg-flashattn --url "$URL" --nonces "$N" --max-tokens 256 --seq-len 256 --save "$2"; }
val(){ "$PY" "$POC/collect.py" --mode validate --model "$HONEST" --profile cg-flashattn --url "$URL" --ref "$1" --save "$2"; }

echo "== [1/2] fraud gen =="; boot "$FRAUD" || exit 1; gen "$FRAUD" "$OUT/early_fraud.json"; kill_srv
echo "== [2/2] honest gen + FLOOR + FRAUD validation =="; boot "$HONEST" || exit 1
gen "$HONEST" "$OUT/early_honest.json"
echo "=== EARLY FLOOR (honest vs honest) ==="; val "$OUT/early_honest.json" "$OUT/early_val_floor.json"
echo "=== EARLY FRAUD (honest vs fraud) ==="; val "$OUT/early_fraud.json"  "$OUT/early_val_fraud.json"
kill_srv
echo "EARLY_SEP_DONE"

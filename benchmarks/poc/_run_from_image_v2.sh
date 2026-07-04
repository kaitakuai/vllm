#!/usr/bin/env bash
# From-IMAGE report, restart-minimized: 6 vLLM boots instead of 8.
# A reboot is forced ONLY by a different engine config or a different model; every
# tool (perf/gen/gsm8k/validate) otherwise shares the live server. Validation needs
# all reference trajectories first, so we order the cudagraph validators LAST and
# generate the eager-honest + AWQ-fraud refs BEFORE them — then each validator folds
# its validation into the SAME boot that does its perf/gen/gsm8k (no re-boot).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$HERE/../../.venv/bin/python"
IMG=ghcr.io/axeltec-software/vllm:v0.20-decode-poc-cu128
HONEST=RedHatAI/Qwen2.5-7B-Instruct-quantized.w8a16
FRAUD=Qwen/Qwen2.5-7B-Instruct-AWQ
PORT=8200; URL="http://localhost:$PORT"
OUT="$HERE/runs/from_image__RTX4000__$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"; echo "=== session: $OUT ==="

boot() { local model="$1"; shift
  docker rm -f vllm_rep >/dev/null 2>&1
  docker run -d --rm --name vllm_rep --gpus all --network host \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    "$IMG" "$model" --port "$PORT" --poc-decode \
    --gpu-memory-utilization 0.85 --max-model-len 4096 "$@" >/dev/null
  for i in $(seq 1 90); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' "$URL/health" 2>/dev/null)" = 200 ] && { echo "  ready ($model ${*})"; return 0; }
    sleep 5
  done; echo "  BOOT FAILED ($model ${*})"; docker logs vllm_rep 2>&1 | tail -15; return 1; }
stop() { docker rm -f vllm_rep >/dev/null 2>&1; }
flags() { case "$1" in
  cg-flashattn) echo "--attention-backend FLASH_ATTN";; cg-flashinfer) echo "--attention-backend FLASHINFER";;
  eager-flashattn) echo "--enforce-eager --attention-backend FLASH_ATTN";; eager-flashinfer) echo "--enforce-eager --attention-backend FLASHINFER";;
esac; }
perf()  { $PY "$HERE/perfomance_nonces.py" --mode both --model "$HONEST" --profile "$1" --url "$URL" --target vllm --seq-len 64 --max-tokens 256 --duration 15 --warmup 8 --save "$OUT/perf_$1.json" && echo "[perf] $1"; }
gen()   { $PY "$HERE/collect.py" --mode generate --model "$2" --profile "$1" --url "$URL" --nonces 8 --max-tokens 256 --save "$OUT/$3" && echo "[gen] $3"; }
gsm()   { local p=$([ "$1" = cg-flashattn ] && echo cudagraph || echo eager)
  $PY "$HERE/quality_gsm8k.py" --model_name "$HONEST" --batch_size 32 --profile "$p" --limit 100 --url "$URL" --output_path "$OUT/gsm_${p}_on"  --save "$OUT/gsm_${p}_on.json"  && echo "[gsm8k] $p on"
  $PY "$HERE/quality_gsm8k.py" --model_name "$HONEST" --batch_size 32 --profile "$p" --limit 100 --disable_poc --url "$URL" --output_path "$OUT/gsm_${p}_off" --save "$OUT/gsm_${p}_off.json" && echo "[gsm8k] $p off"; }
validate(){ for ref in "$OUT"/gen_*.json; do [ -e "$ref" ] || continue; tag="$(basename "$ref" .json)"
  $PY "$HERE/collect.py" --mode validate --model "$HONEST" --profile "$1" --url "$URL" --ref "$ref" --save "$OUT/val_$1__${tag}.json" && echo "[val] $1 vs $tag"; done; }

# --- Phase A: non-validator references first (eager honest + AWQ fraud) ---
boot "$HONEST" $(flags eager-flashattn)  && { perf eager-flashattn;  gen eager-flashattn  "$HONEST" gen_honest_eager-flashattn.json;  gsm eager-flashattn; stop; }
boot "$HONEST" $(flags eager-flashinfer) && { perf eager-flashinfer; gen eager-flashinfer "$HONEST" gen_honest_eager-flashinfer.json; stop; }
boot "$FRAUD"  $(flags cg-flashattn)     && { gen cg-flashattn  "$FRAUD" gen_fraud_cg-flashattn.json;  stop; }
boot "$FRAUD"  $(flags cg-flashinfer)    && { gen cg-flashinfer "$FRAUD" gen_fraud_cg-flashinfer.json; stop; }

# --- Phase B: cudagraph validators LAST — perf+gen+gsm8k+validate in ONE boot each ---
boot "$HONEST" $(flags cg-flashattn)  && { perf cg-flashattn;  gen cg-flashattn  "$HONEST" gen_honest_cg-flashattn.json;  gsm cg-flashattn;  validate cg-flashattn;  stop; }
boot "$HONEST" $(flags cg-flashinfer) && { perf cg-flashinfer; gen cg-flashinfer "$HONEST" gen_honest_cg-flashinfer.json; validate cg-flashinfer; stop; }

$PY "$HERE/report.py" "$OUT" --out "$OUT/report.html"
echo "=== DONE: $OUT/report.html ===  (6 boots)"

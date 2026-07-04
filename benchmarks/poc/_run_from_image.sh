#!/usr/bin/env bash
# From-IMAGE report: boot the pushed vLLM image per engine config and drive the
# benchmark tooling against it over HTTP (NOT local auto-boot). Produces a report
# whose every number came from the shipped docker image.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$HERE/../../.venv/bin/python"
IMG=ghcr.io/axeltec-software/vllm:v0.20-decode-poc-cu128
HONEST=RedHatAI/Qwen2.5-7B-Instruct-quantized.w8a16
FRAUD=Qwen/Qwen2.5-7B-Instruct-AWQ
PORT=8200; URL="http://localhost:$PORT"
OUT="$HERE/runs/from_image__RTX4000__$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"
echo "=== session: $OUT ==="

boot() {  # boot(model, extra_flags...)
  local model="$1"; shift
  docker rm -f vllm_rep >/dev/null 2>&1
  docker run -d --rm --name vllm_rep --gpus all --network host \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    "$IMG" "$model" --port "$PORT" --poc-decode \
    --gpu-memory-utilization 0.85 --max-model-len 4096 "$@" >/dev/null
  for i in $(seq 1 90); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' "$URL/health" 2>/dev/null)" = 200 ] && { echo "  ready ($model ${*})"; return 0; }
    sleep 5
  done
  echo "  BOOT FAILED ($model ${*})"; docker logs vllm_rep 2>&1 | tail -15; return 1
}
stop() { docker rm -f vllm_rep >/dev/null 2>&1; }

# config -> serve flags
flags() { case "$1" in
  cg-flashattn)    echo "--attention-backend FLASH_ATTN";;
  cg-flashinfer)   echo "--attention-backend FLASHINFER";;
  eager-flashattn) echo "--enforce-eager --attention-backend FLASH_ATTN";;
  eager-flashinfer)echo "--enforce-eager --attention-backend FLASHINFER";;
esac; }

# 1) honest configs: perf(both) + honest gen ; gsm8k on the two flashattn configs
for c in cg-flashattn eager-flashattn cg-flashinfer eager-flashinfer; do
  boot "$HONEST" $(flags "$c") || continue
  $PY "$HERE/perfomance_nonces.py" --mode both --model "$HONEST" --profile "$c" \
      --url "$URL" --target vllm --seq-len 64 --max-tokens 256 --duration 15 --warmup 8 \
      --save "$OUT/perf_${c}.json" && echo "[perf] $c"
  $PY "$HERE/collect.py" --mode generate --model "$HONEST" --profile "$c" \
      --url "$URL" --nonces 8 --max-tokens 256 --save "$OUT/gen_honest_${c}.json" && echo "[gen] honest $c"
  case "$c" in cg-flashattn|eager-flashattn)
    prof=$([ "$c" = cg-flashattn ] && echo cudagraph || echo eager)
    $PY "$HERE/quality_gsm8k.py" --model_name "$HONEST" --batch_size 32 --profile "$prof" --limit 100 \
        --url "$URL" --output_path "$OUT/gsm_${prof}_on"  --save "$OUT/gsm_${prof}_on.json"  && echo "[gsm8k] $prof on"
    $PY "$HERE/quality_gsm8k.py" --model_name "$HONEST" --batch_size 32 --profile "$prof" --limit 100 --disable_poc \
        --url "$URL" --output_path "$OUT/gsm_${prof}_off" --save "$OUT/gsm_${prof}_off.json" && echo "[gsm8k] $prof off"
  ;; esac
  stop
done

# 2) fraud refs (AWQ) on two configs
for c in cg-flashattn cg-flashinfer; do
  boot "$FRAUD" $(flags "$c") || continue
  $PY "$HERE/collect.py" --mode generate --model "$FRAUD" --profile "$c" \
      --url "$URL" --nonces 8 --max-tokens 256 --save "$OUT/gen_fraud_${c}.json" && echo "[gen] fraud $c"
  stop
done

# 3) validate the matrix from two validator configs (same-backend + cross)
for v in cg-flashattn cg-flashinfer; do
  boot "$HONEST" $(flags "$v") || continue
  for ref in "$OUT"/gen_*.json; do
    [ -e "$ref" ] || continue
    tag="$(basename "$ref" .json)"
    $PY "$HERE/collect.py" --mode validate --model "$HONEST" --profile "$v" \
        --url "$URL" --ref "$ref" --save "$OUT/val_${v}__${tag}.json" && echo "[val] $v vs $tag"
  done
  stop
done

# 4) render
$PY "$HERE/report.py" "$OUT" --out "$OUT/report.html"
echo "=== DONE: $OUT/report.html ==="
ls "$OUT" | wc -l

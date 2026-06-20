#!/usr/bin/env bash
# Run the decode-PoC test matrix for ONE model into its OWN session folder, then
# render an HTML report. Discipline: one session = one model (on one box, one day) =
# one folder runs/<model>__<gpu>__<date>/ — never a shared trash-bin runs/.
#
# Thin orchestration over the existing tools (collect.py / perfomance_nonces.py /
# quality_gsm8k.py / report.py); no new measurement code.
#
#   run_model_report.sh <honest-model> [fraud-model] [--url URL] [--session NAME]
#                       [--nonces N] [--max-tokens MT] [--scope full|quick] [--no-gsm8k]
#
# scope=full  : 4 perf + 16 honest config-pairs + 8 fraud + gsm8k(on/off)  (the 24-cell matrix)
# scope=quick : 4 perf + diagonal honest/fraud (4) + gsm8k                  (fast sanity)
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# Prefer the repo venv (has httpx/requests/lm_eval + the `vllm` CLI that _server.py
# shells out to for local auto-boot); fall back to python3/PATH.
if [ -z "${PY:-}" ]; then
  if [ -x "$HERE/../../.venv/bin/python" ]; then PY="$HERE/../../.venv/bin/python";
  else PY="python3"; fi
fi
# Local auto-boot runs `vllm serve` via subprocess — its dir must be on PATH.
case "$PY" in */*) export PATH="$(cd "$(dirname "$PY")" && pwd):$PATH";; esac

HONEST="${1:?usage: run_model_report.sh <honest-model> [fraud-model] [opts]}"; shift || true
FRAUD=""; URL=""; SESSION=""; NONCES=8; MT=16; SCOPE=full; GSM=1
while [ $# -gt 0 ]; do case "$1" in
  --url) URL="$2"; shift 2;; --session) SESSION="$2"; shift 2;;
  --nonces) NONCES="$2"; shift 2;; --max-tokens) MT="$2"; shift 2;;
  --scope) SCOPE="$2"; shift 2;; --no-gsm8k) GSM=0; shift;;
  --*) echo "unknown $1"; exit 2;; *) FRAUD="$1"; shift;;
esac; done

CONFIGS=(cg-flashattn cg-flashinfer eager-flashattn eager-flashinfer)
slug(){ echo "$1" | tr '/: ' '___' | tr -cd 'A-Za-z0-9._-'; }
# Session-folder naming discipline (default): runs/<model>__<gpu>__<timestamp>/ — one
# session = one model on one box at one time. Override with --session NAME.
GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
[ -z "$GPU" ] && GPU="remote"
TS="$(date +%Y%m%d-%H%M%S 2>/dev/null || echo session)"
[ -z "$SESSION" ] && SESSION="$(slug "$HONEST")__$(slug "$GPU")__${TS}"
OUT="${HERE}/runs/${SESSION}"
mkdir -p "$OUT"
URLA=(); [ -n "$URL" ] && URLA=(--url "$URL")
echo "=== session: $OUT  (model=$HONEST fraud=${FRAUD:-none} scope=$SCOPE nonces=$NONCES mt=$MT) ==="

# 1. PERF — one per config
for c in "${CONFIGS[@]}"; do
  echo "[perf] $c"
  $PY "$HERE/perfomance_nonces.py" --model "$HONEST" --profile "$c" \
      --seq-len 64 --max-tokens "$MT" --duration 20 "${URLA[@]}" \
      --save "$OUT/perf_${c}.json" || echo "  perf $c FAILED"
done

# 2. SEPARATION — generate references, then validate the matrix
GEN_CFGS=("${CONFIGS[@]}"); [ "$SCOPE" = quick ] && GEN_CFGS=(cg-flashattn)
for c in "${GEN_CFGS[@]}"; do
  $PY "$HERE/collect.py" --mode generate --model "$HONEST" --profile "$c" \
      --nonces "$NONCES" --max-tokens "$MT" "${URLA[@]}" --save "$OUT/gen_honest_${c}.json" \
      && echo "[gen] honest $c" || echo "  gen honest $c FAILED"
done
if [ -n "$FRAUD" ]; then
  FR_CFGS=(cg-flashattn cg-flashinfer); [ "$SCOPE" = quick ] && FR_CFGS=(cg-flashattn)
  for c in "${FR_CFGS[@]}"; do
    $PY "$HERE/collect.py" --mode generate --model "$FRAUD" --profile "$c" \
        --nonces "$NONCES" --max-tokens "$MT" "${URLA[@]}" --save "$OUT/gen_fraud_${c}.json" \
        && echo "[gen] fraud $c" || echo "  gen fraud $c FAILED"
  done
fi
VAL_CFGS=("${CONFIGS[@]}"); [ "$SCOPE" = quick ] && VAL_CFGS=(cg-flashattn)
for v in "${VAL_CFGS[@]}"; do
  for ref in "$OUT"/gen_*.json; do
    [ -e "$ref" ] || continue
    tag="$(basename "$ref" .json)"
    $PY "$HERE/collect.py" --mode validate --model "$HONEST" --profile "$v" \
        --ref "$ref" "${URLA[@]}" --save "$OUT/val_${v}__${tag}.json" \
        && echo "[val] $v vs $tag" || echo "  val $v vs $tag FAILED"
  done
done

# 3. GSM8K co-existence (on vs off) — same N both runs so accuracy is comparable.
# Co-existence holds iff PoC-load accuracy ≈ baseline (within sampling margin).
GSM_LIMIT="${GSM_LIMIT:-100}"
GSM_BS="${GSM_BS:-32}"   # production inference batch size
if [ "$GSM" = 1 ]; then
  # 4 runs: pure chat vs +PoC, each on cudagraph and eager (co-existence must hold on both).
  for prof in cudagraph eager; do
    $PY "$HERE/quality_gsm8k.py" --model_name "$HONEST" --batch_size "$GSM_BS" --profile "$prof" --limit "$GSM_LIMIT" \
        "${URLA[@]}" --output_path "$OUT/gsm_${prof}_on"  --save "$OUT/gsm_${prof}_on.json"  && echo "[gsm8k] $prof on"  || echo "  gsm8k $prof on FAILED"
    $PY "$HERE/quality_gsm8k.py" --model_name "$HONEST" --batch_size "$GSM_BS" --profile "$prof" --limit "$GSM_LIMIT" --disable_poc \
        "${URLA[@]}" --output_path "$OUT/gsm_${prof}_off" --save "$OUT/gsm_${prof}_off.json" && echo "[gsm8k] $prof off" || echo "  gsm8k $prof off FAILED"
  done
fi

# 4. RENDER
$PY "$HERE/report.py" "$OUT" --out "$OUT/report.html"
echo "=== report: $OUT/report.html ==="

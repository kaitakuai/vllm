#!/usr/bin/env bash
# pair_report.sh — honest/fraud separation for a PAIR of models, end to end.
# Thin wrapper over collect.py + analyze.py (no new logic): generate each model's
# reference, validate both models against BOTH references (the 2x2 matrix:
# diagonal = honest ~0%, off-diagonal = fraud high), then print + save the report.
#
#   pair_report.sh <model_a> <model_b> <out_dir> [opts...]
#   pair_report.sh M1 M2 runs/pairAB --url $S --nonces 64 --max-tokens 256
#
# Engine config comes from named profiles (see poc_configs.json):
#   --profile NAME       same engine for both phases (prover + validator)
#   --gen-profile NAME   prover (generate) engine   } override --profile per phase,
#   --val-profile NAME   validator (validate) engine} e.g. prove on cudagraph,
#                                                      validate on eager (cross-engine
#                                                      determinism test).
# All other flags (--url/--target/--dtype/--nonces/--seq-len/--max-tokens/
# --p-mismatch/--configs) are forwarded verbatim to every collect.py call.
set -euo pipefail

A="${1:?usage: pair_report.sh <model_a> <model_b> <out_dir> [opts...]}"
B="${2:?model_b required}"
OUT="${3:?out_dir required}"
shift 3

# Own the profile flags (phase-specific); forward everything else to collect.py.
PROFILE=""; GEN_PROFILE=""; VAL_PROFILE=""; REST=()
while [ $# -gt 0 ]; do
  case "$1" in
    --profile)      PROFILE="$2";      shift 2;;
    --profile=*)    PROFILE="${1#*=}";      shift;;
    --gen-profile)  GEN_PROFILE="$2";  shift 2;;
    --gen-profile=*)GEN_PROFILE="${1#*=}";  shift;;
    --val-profile)  VAL_PROFILE="$2";  shift 2;;
    --val-profile=*)VAL_PROFILE="${1#*=}";  shift;;
    *) REST+=("$1"); shift;;
  esac
done
GEN_PROFILE="${GEN_PROFILE:-$PROFILE}"
VAL_PROFILE="${VAL_PROFILE:-$PROFILE}"
GP=(); [ -n "$GEN_PROFILE" ] && GP=(--profile "$GEN_PROFILE")
VP=(); [ -n "$VAL_PROFILE" ] && VP=(--profile "$VAL_PROFILE")

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEF_PY="$HERE/../../.venv/bin/python"
PY="${PYTHON:-$([ -x "$DEF_PY" ] && echo "$DEF_PY" || echo python3)}"
mkdir -p "$OUT"

slug() { printf '%s' "$1" | tr '/:' '__'; }
GA="$OUT/gen_$(slug "$A").json"
GB="$OUT/gen_$(slug "$B").json"

# 1. generate each model's reference trajectory (prover engine = gen-profile)
"$PY" "$HERE/collect.py" --mode generate --model "$A" --save "$GA" ${GP[@]+"${GP[@]}"} ${REST[@]+"${REST[@]}"}
"$PY" "$HERE/collect.py" --mode generate --model "$B" --save "$GB" ${GP[@]+"${GP[@]}"} ${REST[@]+"${REST[@]}"}

# 2. validate both models against both references (validator engine = val-profile)
for V in "$A" "$B"; do
  for R in "$GA" "$GB"; do
    "$PY" "$HERE/collect.py" --mode validate --model "$V" --ref "$R" \
      --save "$OUT/val_$(slug "$V")_vs_$(basename "$R" .json).json" ${VP[@]+"${VP[@]}"} ${REST[@]+"${REST[@]}"}
  done
done

# 3. report
"$PY" "$HERE/analyze.py" "$OUT"/*.json | tee "$OUT/report.txt"
echo "report -> $OUT/report.txt"

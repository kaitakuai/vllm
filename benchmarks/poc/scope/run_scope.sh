#!/usr/bin/env bash
# CUSTOM, NON-GIT grouped orchestrator (lives in ~/gonka/experiments, NOT the vllm repo).
# Boots each (model,config) server ONCE, runs ALL its ops via --url (minimal reboots),
# REUSING in-git measurement tools (perfomance_nonces.py/collect.py/quality_gsm8k.py); renders ONE report via simplify_report.py.
# Then renders the SINGLE simplified report (simplify_report.py). S3 push is a separate opt-in step (s3.sh push-report).
#
#   run_scope.sh <honest-model> <fraud-model> [--mla] [--nonces N] [--max-tokens M]
#                [--seq-len S] [--gsm-limit L] [--no-gsm] [--push]
#
# Separation = 5 logical pairs (validator fixed = honest @ production cg-FA; vary the
# reference ONE axis at a time, one direction):
#   1 V(cgFA)<=H(cgFA) floor | 2 <=H(cgFI) backend | 3 <=H(eagerFA) cudagraph
#   4 <=F(cgFA) fraud         | 5 <=F(cgFI) fraud+backend
# (MLA models force TRITON_MLA -> no FA/FI axis: pairs reduce to floor/cudagraph/fraud.)
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
PY="${POC_PY:-$REPO/.venv/bin/python}"; command -v "$PY" >/dev/null 2>&1 || PY="$(command -v python3)"
POC="$(cd "$HERE/.." && pwd)"
PORT=8200; URL="http://127.0.0.1:$PORT"
export PATH="$(dirname "$PY"):$PATH"
export HF_TOKEN="$(cat ~/.cache/huggingface/token 2>/dev/null || echo "")"

HONEST="${1:?usage: run_scope.sh <honest> <fraud> [--mla] [opts]}"; FRAUD="${2:?need fraud}"; shift 2
MLA=0; NONCES=128; MT=256; SEQ=256; GSMN=100; GSM=1; PUSH=0; PERFON=1; NOFI=0; GSM_MML=2048; TP=1; GMU=0.90; EXTRA=""; XHW=""; XHW_ONLY=0
SCOPE_COMMIT="$(git -C "$HERE" rev-parse --short HEAD 2>/dev/null || echo '?')"   # report-tooling version
while [ $# -gt 0 ]; do case "$1" in
  --mla) MLA=1; shift;; --nonces) NONCES="$2"; shift 2;; --max-tokens) MT="$2"; shift 2;;
  --seq-len) SEQ="$2"; shift 2;; --gsm-limit) GSMN="$2"; shift 2;; --no-gsm) GSM=0; shift;;
  --no-perf) PERFON=0; shift;; --no-fi) NOFI=1; shift;;   # drop FlashInfer (FI-MoE+seeded-routing deadlock)
  --gsm-mml) GSM_MML="$2"; shift 2;;   # GSM-only max-model-len (5-shot prompts > 1024; PoC stays 1024)
  --tp) TP="$2"; shift 2;;   # tensor-parallel size for multi-GPU serving of large models
  --gpu-mem) GMU="$2"; shift 2;;   # gpu-memory-utilization (per deploy config, e.g. 0.90-0.95)
  --extra) EXTRA="$2"; shift 2;;   # raw extra serve args from the deploy config (e.g. "--enable-expert-parallel --attention-backend <MLA-backend>")
  --xhw) XHW="$2"; shift 2;;   # CROSS-HW: peer session (local reports/<name> dir OR S3 name); pull its refs + validate here
  --xhw-only) XHW_ONLY=1; shift;;   # phase-2: SKIP local gen/perf/gsm, ONLY cross-validate the --xhw peer (for parallel 2-box verify)
  --push) PUSH=1; shift;; *) echo "unknown opt $1"; exit 2;;
esac; done

slug(){ echo "$1" | tr '/: ' '___' | tr -cd 'A-Za-z0-9._-'; }
GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr ' /:' '___')"; [ -z "$GPU" ] && GPU=remote
TS="$(date +%Y%m%d-%H%M%S)"; SESS="$(slug "$HONEST")__${GPU}__${TS}"; OUT="$HERE/reports/$SESS"; mkdir -p "$OUT"

# config sets by attention type
if [ "$MLA" = 1 ]; then VAL=cudagraph; PERF=(cudagraph eager); HREF=(eager); FREF=(cudagraph)
else VAL=cg-flashattn; PERF=(cg-flashattn cg-flashinfer eager-flashattn eager-flashinfer); HREF=(cg-flashinfer eager-flashattn); FREF=(cg-flashattn cg-flashinfer); fi
if [ "$NOFI" = 1 ]; then _filt(){ local o=(); for x in "$@"; do case "$x" in *flashinfer*) ;; *) o+=("$x");; esac; done; echo "${o[@]}"; }
  read -ra PERF <<< "$(_filt "${PERF[@]}")"; read -ra HREF <<< "$(_filt "${HREF[@]}")"; read -ra FREF <<< "$(_filt "${FREF[@]}")"; fi

prof_args(){ case "$1" in
  cg-flashattn) echo "--attention-backend FLASH_ATTN";;
  cg-flashinfer) echo "--attention-backend FLASHINFER";;
  eager-flashattn) echo "--attention-backend FLASH_ATTN --enforce-eager";;
  eager-flashinfer) echo "--attention-backend FLASHINFER --enforce-eager";;
  eager) echo "--enforce-eager";; cudagraph) echo "";; esac; }

SRV_PID=""
boot(){ # model profile [max_model_len]  -> boot one server (own process group); PoC needs only
        # seq_len+max_tokens (<=320) so default 1024; GSM passes a bigger value for its long prompts.
  local model="$1" prof="$2" mml="${3:-1024}"; local lg="$OUT/serve_$(slug "$model")_$prof.log"
  ( cd /tmp && exec setsid "$PY" -m vllm.entrypoints.openai.api_server --model "$model" --port "$PORT" \
      --poc-decode --gpu-memory-utilization "$GMU" --max-model-len "$mml" --tensor-parallel-size "$TP" --trust-remote-code \
      $EXTRA $(prof_args "$prof") ) > "$lg" 2>&1 &
  SRV_PID=$!
  for i in $(seq 1 100); do
    s=$(curl -s "$URL/v1/models" 2>/dev/null | "$PY" -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || true)
    [ "$s" = "$model" ] && { echo "  [boot ok] $model @ $prof"; kv_preflight "$lg"; return 0; }
    ps -p "$SRV_PID" >/dev/null 2>&1 || { echo "  [BOOT DIED] $model @ $prof (see $lg)"; return 1; }
    sleep 8
  done; echo "  [BOOT TIMEOUT] $model @ $prof"; return 1
}
kv_preflight(){ # read vLLM's logged KV size; warn (don't waste a run) if the PoC concurrent load won't fit
  local lg="$1"
  local kvtok; kvtok=$(grep -oE "GPU KV cache size: [0-9,]+ tokens" "$lg" 2>/dev/null | head -1 | grep -oE "[0-9,]+" | tr -d ,)
  [ -z "$kvtok" ] && { echo "  [kv] (KV size not logged yet)"; return 0; }
  local per=$(( SEQ + MT ))                       # KV tokens one PoC nonce needs (seq_len + trajectory)
  local load=32; [ "$NONCES" -lt 32 ] && load="$NONCES"   # concurrent in-flight (route batch_size=32)
  local cap=$(( kvtok / per )); local need=$(( load * per ))
  echo "  [kv] ${kvtok} tok -> fits ${cap} concurrent nonces @${per}; load=${load} needs ${need} tok"
  if [ "$need" -gt "$kvtok" ]; then
    echo "  ┌─ KV PREFLIGHT: INSUFFICIENT ──────────────────────────────"
    echo "  │ load ${load} > capacity ${cap}  -> KV preemption/thrash (pathologically slow)."
    echo "  │ fix: lower --max-tokens, reduce load, smaller/quantized model, or more KV."
    echo "  └───────────────────────────────────────────────────────────"
  fi
}
kill_srv(){ # kill ONLY my server's process group (safe on shared GPU), then WAIT for VRAM to actually
  # free — vLLM multiproc workers release CUDA memory AFTER the process exits, so the next boot OOMs
  # on a tight gpu-mem (e.g. MiniMax @0.90) unless we poll until it's really free.
  # vLLM v1 boots via `setsid` (new session) so `kill -- -$PID` (our subshell's group) MISSES the
  # detached python + its GPU workers. Reap by GPU PID until VRAM is actually free (dedicated box).
  [ -n "$SRV_PID" ] && kill -- "-$SRV_PID" 2>/dev/null
  for i in $(seq 1 25); do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ "${u:-99999}" -lt 3000 ] && break
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9 2>/dev/null
    sleep 2
  done
  SRV_PID=""
}
stamp(){ # post-stamp meta.attention_backend/cudagraph_mode/profile from the profile name
  local f="$1" prof="$2"; [ -f "$f" ] || return 0
  "$PY" - "$f" "$prof" <<'P'
import json,sys
f,prof=sys.argv[1:3]
be="FLASH_ATTN" if "flashattn" in prof else ("FLASHINFER" if "flashinfer" in prof else None)
d=json.load(open(f)); m=d.setdefault("meta",{})
if be: m["attention_backend"]=be
m["cudagraph_mode"]="eager" if "eager" in prof else "cudagraph"; m["profile"]=prof
json.dump(d,open(f,"w"),indent=2)
P
}
op_perf(){ [ "$PERFON" = 0 ] && return 0; local p="$1"; echo "[perf] $p"
  "$PY" "$POC/perfomance_nonces.py" --mode both --model "$HONEST" --profile "$p" --url "$URL" \
     --seq-len "$SEQ" --max-tokens "$MT" --duration 20 --save "$OUT/perf_$p.json" || echo "  perf $p FAILED"
  stamp "$OUT/perf_$p.poc.json" "$p"; stamp "$OUT/perf_$p.chat.json" "$p"; }
op_gen(){ local model="$1" p="$2" tag="$3"; echo "[gen $tag] $p"
  "$PY" "$POC/collect.py" --mode generate --model "$model" --profile "$p" --url "$URL" \
     --nonces "$NONCES" --max-tokens "$MT" --seq-len "$SEQ" --save "$OUT/gen_${tag}_$p.json" || echo "  gen $tag $p FAILED"
  stamp "$OUT/gen_${tag}_$p.json" "$p"; }
op_val(){ local p="$1" ref="$2"; local tag; tag="$(basename "$ref" .json)"; echo "[val] $p <= $tag"
  "$PY" "$POC/collect.py" --mode validate --model "$HONEST" --profile "$p" --url "$URL" \
     --ref "$ref" --save "$OUT/val_${p}__${tag}.json" || echo "  val $p <= $tag FAILED"
  stamp "$OUT/val_${p}__${tag}.json" "$p"; }
op_gsm(){ local p="$1" oo="$2" extra="$3"; echo "[gsm] $p $oo"
  "$PY" "$POC/quality_gsm8k.py" --model_name "$HONEST" --profile "$p" --url "$URL" --batch_size 16 \
     --limit "$GSMN" $extra --output_path "$OUT/gsm_${p}_$oo" --save "$OUT/gsm_${p}_$oo.json" || echo "  gsm $p $oo FAILED"; }

COMMIT="$(cd "$POC" && git rev-parse --short HEAD 2>/dev/null || echo '?')"; BRANCH="$(cd "$POC" && git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
cat > "$OUT/REPRODUCE.md" <<EOF
# Reproduce: decode-PoC report
- honest/validator: \`$HONEST\`  | fraud: \`$FRAUD\`  | attention: $([ "$MLA" = 1 ] && echo MLA || echo full)
- perf=[${PERF[*]}] | validator=$VAL | honest-refs=[$VAL ${HREF[*]}] | fraud-refs=[${FREF[*]}]
- params: nonces=$NONCES max_tokens=$MT seq_len=$SEQ gsm_limit=$GSMN | GPU=$GPU | vLLM $BRANCH@$COMMIT | poc-scope@$SCOPE_COMMIT${XHW:+ | cross-HW ref=$XHW}
- branch: https://github.com/axeltec-software/vllm/tree/poc-v0.20-decode-poc-cg
Run:  bash run_scope.sh "$HONEST" "$FRAUD" $([ "$MLA" = 1 ] && echo --mla)
Pull: bash s3.sh pull-report $SESS ./$SESS
EOF

echo "=== $SESS | honest=$HONEST fraud=$FRAUD mla=$MLA nonces=$NONCES mt=$MT ==="
# === MODE 1 (same-HW): generate refs + local separation + perf.  Skipped in --xhw-only. ===
if [ "$XHW_ONLY" = 0 ]; then
  # fraud reference trajectories (one boot per fraud config)
  for p in "${FREF[@]}"; do boot "$FRAUD" "$p" && op_gen "$FRAUD" "$p" fraud; kill_srv; done
  # honest non-production configs: perf + honest refs (pairs 2,3)
  for p in "${HREF[@]}"; do boot "$HONEST" "$p" && { op_perf "$p"; op_gen "$HONEST" "$p" honest; }; kill_srv; done
  # remaining perf-only honest configs (no ref needed)
  if [ "$PERFON" = 1 ]; then for p in "${PERF[@]}"; do
    case " $VAL ${HREF[*]} " in *" $p "*) continue;; esac   # already perf'd above
    boot "$HONEST" "$p" && op_perf "$p"; kill_srv
  done; fi
fi
# === validator (cg-FA): same-HW separation (mode 1) AND/OR cross-HW validation (mode 2, --xhw) ===
if boot "$HONEST" "$VAL"; then
  if [ "$XHW_ONLY" = 0 ]; then
    op_perf "$VAL"
    op_gen "$HONEST" "$VAL" honest                       # baseline ref H(cgFA)
    for ref in "$OUT"/gen_honest_*.json "$OUT"/gen_fraud_*.json; do [ -e "$ref" ] && op_val "$VAL" "$ref"; done
  fi
  # MODE 2 (cross-HW): ONE validator boot re-checks ALL peers' ALREADY-GENERATED refs
  # (comma-separate --xhw to fold many boxes into one boot — max artifact reuse, min runs).
  if [ -n "$XHW" ]; then
    IFS=',' read -ra PEERS <<< "$XHW"
    for peer in "${PEERS[@]}"; do
      [ -z "$peer" ] && continue
      xdir="$HERE/reports/$peer"                          # local peer session (same-box testing)...
      if [ -d "$xdir" ] && ls "$xdir"/gen_*.json >/dev/null 2>&1; then echo "[xhw] using LOCAL peer $peer"
      else xdir="$OUT/_xhw/$peer"; echo "[xhw] pulling peer $peer from S3"; bash "$HERE/s3.sh" pull-report "$peer" "$xdir" >/dev/null 2>&1 || { echo "  [xhw] pull FAILED $peer"; continue; }; fi
      pslug="$(echo "$peer" | sed -E 's/.*__(.+)__[0-9-]+$/\1/' | cut -c1-24)"   # peer GPU slug (uniquifies)
      for ref in "$xdir"/gen_honest_*.json "$xdir"/gen_fraud_*.json; do
        [ -e "$ref" ] || continue
        base="$(basename "$ref" .json)"; tag="xhw_${pslug}_${base#gen_}"   # xhw_<peergpu>_honest_cg-flashattn
        echo "[val·xhw] $VAL <= $tag"
        "$PY" "$POC/collect.py" --mode validate --model "$HONEST" --profile "$VAL" --url "$URL" \
           --ref "$ref" --save "$OUT/val_${VAL}__${tag}.json" || echo "  val·xhw $tag FAILED"
        stamp "$OUT/val_${VAL}__${tag}.json" "$VAL"
      done
    done
    [ -d "$OUT/_xhw" ] && rm -rf "$OUT/_xhw"             # peer refs belong to their sessions, not this archive
  fi
fi
kill_srv
# === GSM8K co-existence (own bigger-max-len boot; skipped in --xhw-only) ===
if [ "$GSM" = 1 ] && [ "$XHW_ONLY" = 0 ]; then
  if boot "$HONEST" "$VAL" "$GSM_MML"; then op_gsm "$VAL" on ""; op_gsm "$VAL" off "--disable_poc"; fi
  kill_srv
fi
# --- sanitize: strip machine-specific absolute paths (home dir / username) from shared evidence ---
sanitize_evidence(){ local d="$1"
  find "$d" -type f \( -name '*.log' -o -name '*.md' -o -name '*.html' -o -name '*.json' -o -name '*.jsonl' \) \
    -exec sed -i "s#${HOME}#~#g; s#/home/[A-Za-z0-9._-]\+#~#g" {} + 2>/dev/null || true
}
sanitize_evidence "$OUT"
# --- render: SINGLE report = the simplified (ideal) layout (the only report we keep) ---
POC_SCOPE_COMMIT="$SCOPE_COMMIT" "$PY" "$HERE/simplify_report.py" "$OUT" --out "$OUT/report.html" || echo "render FAILED"
# --- S3 archive (opt-in; separate approved step otherwise) ---
if [ "$PUSH" = 1 ]; then "$PY" "$HERE/inject_s3.py" "$OUT/report.html" "$SESS"; bash "$HERE/s3.sh" push-report "$OUT" "$SESS"; fi
echo "=== DONE: $OUT/report.html ==="

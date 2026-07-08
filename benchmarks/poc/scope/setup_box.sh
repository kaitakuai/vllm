#!/usr/bin/env bash
# Robust decode-PoC box setup — clone/checkout done separately; this builds the venv.
#
# THE ONE INVARIANT (all setup failures trace back to violating it):
#   torch  ==CUDA==  torchvision  ==CUDA==  vLLM's compiled `_C.abi3.so`
# The published `vllm==0.20.1` wheel ships a working `_C` for a specific CUDA (currently
# cu130). `--torch-backend=auto` installs the WHOLE torch family (torch+torchvision) for
# the host's CUDA. On a driver-580 / CUDA-13 host that's cu130 → matches the wheel. Never
# reinstall torch alone (leaves torchvision on the old CUDA → "compiled with different CUDA").
#
# Do NOT use `VLLM_USE_PRECOMPILED=1 pip install -e .` — for our fork commit it drops the
# core `_C.abi3.so` and only lays down `_C_stable_libtorch`. Install the published wheel
# and OVERLAY our PoC `.py`+`.pt` on top of it instead.
#
# HARD REQUIREMENT — the published wheel is a FIXED CUDA build (currently cu130), so the
# HOST DRIVER must be CUDA-13-capable (NVIDIA R580+). On an older driver (e.g. R550 / max
# CUDA 12.4, our RTX-4000 dev box) there is NO working torch pin for this wheel:
#   --torch-backend=auto  -> cu12 torch -> ImportError: libcudart.so.13 (wheel _C can't load)
#   --torch-backend=cu130 -> cu13 torch -> RuntimeError: NVIDIA driver too old (found 12040)
# On such a box either (a) upgrade the driver to R580+, or (b) DON'T use this script —
# build vllm from source against CUDA 12 (an editable cu12 build runs on a CUDA-12.x driver
# via minor-version compat: torch 2.11+cu128 + `_C`→libcudart.so.12 works on R550).
set -euo pipefail
REPO="${REPO:-$HOME/vllm}"
export PATH="$HOME/.local/bin:$PATH"
cd "$REPO"

echo "== 0. driver/CUDA preflight (published wheel is a fixed CUDA build) =="
WHEEL_CUDA=13   # published vllm==0.20.1 ships a cu130 _C → needs an R580+ (CUDA-13) driver
DRV_CUDA=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9]*\).*/\1/p' | head -1)
if [ -n "$DRV_CUDA" ] && [ "$DRV_CUDA" -lt "$WHEEL_CUDA" ]; then
  echo "FATAL: host driver supports only CUDA ${DRV_CUDA}.x, but the vllm==0.20.1 wheel is CUDA ${WHEEL_CUDA} (needs libcudart.so.${WHEEL_CUDA})." >&2
  echo "       Fix: (a) upgrade GPU driver to R580+ and re-run; OR (b) build vllm from source against CUDA 12 (this script does not)." >&2
  exit 3
fi
echo "   driver CUDA=${DRV_CUDA:-unknown} · wheel CUDA=${WHEEL_CUDA} · OK"

echo "== 1. clean venv (uv won't re-resolve an already-satisfied torch → stale CUDA lingers) =="
rm -rf .venv
uv venv --python 3.12 .venv
source .venv/bin/activate

echo "== 2. ONE install: published vllm wheel + torch FAMILY, CUDA auto-matched to host =="
uv pip install vllm==0.20.1 torch torchvision \
  flashinfer-python==0.6.8.post1 transformers==5.12.1 numpy==2.3.5 \
  requests httpx "lm_eval[api]" tenacity --torch-backend=auto
# NOTE: lm_eval[api] (pulls tenacity) is REQUIRED — GSM8K queries the vLLM server as an
# lm-eval "local-completions" API model; bare lm_eval → ModuleNotFoundError: tenacity → 0.0 acc.

echo "== 3. overlay our PoC .py + .pt (codebook) onto the installed wheel (site-packages) =="
# compute site-packages from /tmp so the source tree ($REPO/vllm) can't shadow it
SP=$(cd /tmp && "$REPO/.venv/bin/python" -c "import vllm,os;print(os.path.dirname(vllm.__file__))")
cd "$REPO"
find vllm \( -name '*.py' -o -name '*.pt' \) -printf '%P\n' | while read f; do
  install -D -m644 "vllm/$f" "$SP/$f"
done

echo "== 4. VERIFY consistency (fail-closed: any mismatch aborts) =="
cd /tmp && "$REPO/.venv/bin/python" - <<'PY'
import torch, torchvision, vllm, vllm._C, vllm._C_stable_libtorch
from vllm.poc import native, mixed_decode, sphere
assert torch.cuda.is_available(), "CUDA not available to torch"
cb = sphere.get_sphere_codebook()
assert sphere._codebook_sha256(cb) == sphere.EXPECTED_CODEBOOK_SHA256, "codebook hash mismatch"
print(f"OK  torch={torch.__version__}  torchvision={torchvision.__version__}  "
      f"cuda={torch.version.cuda}  gpus={torch.cuda.device_count()}")
print(f"    vllm={vllm.__file__}  poc+_C+codebook all import")
PY
echo "SETUP_OK — vLLM ready to serve"

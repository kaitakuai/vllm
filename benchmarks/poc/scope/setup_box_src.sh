#!/usr/bin/env bash
# Source build (cu12) — for any host whose DRIVER does NOT support CUDA 13 (i.e. max CUDA 12.x,
# e.g. R550 / CUDA 12.4). The published vllm==0.20.1 wheel is cu13 and CANNOT run on such a
# driver (see ../HOWTO.md §1). This compiles vLLM's `_C` against the local CUDA-12 toolkit instead.
# Counterpart to setup_box.sh (which is for CUDA-13 / R580+ hosts).
#
#   FIRST CHOICE: don't rebuild — if the host already has a working cu12 venv, just
#   `source .venv/bin/activate`. Run this ONLY for a fresh/clobbered host. Full compile ≈ 1 h.
set -euo pipefail
REPO="${REPO:-$HOME/gonka/vllm-v0.20}"
export PATH="$HOME/.local/bin:$PATH"
cd "$REPO"

echo "== 0. preflight: this script is for CUDA-12 driver boxes only =="
DRV_CUDA=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9]*\).*/\1/p' | head -1)
if [ -n "$DRV_CUDA" ] && [ "$DRV_CUDA" -ge 13 ]; then
  echo "NOTE: driver supports CUDA ${DRV_CUDA} (>=13) → use the FAST path 'setup_box.sh' (published wheel), not this source build." >&2
  exit 3
fi
NVCC="$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)"
"$NVCC" --version 2>/dev/null | grep -q "release 12" || { echo "FATAL: need a CUDA 12.x toolkit (nvcc) at /usr/local/cuda; found: $("$NVCC" --version 2>/dev/null | grep release || echo none)" >&2; exit 3; }
echo "   driver CUDA=${DRV_CUDA:-?} · toolkit $("$NVCC" --version | sed -n 's/.*release \([0-9.]*\).*/\1/p' | head -1) · OK"

echo "== 1. clean venv =="
rm -rf .venv && uv venv --python 3.12 .venv && source .venv/bin/activate

echo "== 2. cu12 torch FAMILY — torch + torchvision + torchAUDIO together, all cu128 =="
# torchaudio is a HARD dep of vllm; if it resolves on its own it pulls a cu13 build whose
# libcudart.so.13 clobbers cu12 and crashes at import (transformers->torchaudio). Pin it here.
uv pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --torch-backend=cu128

echo "== 3. build vLLM FROM SOURCE (nvcc 12.x compiles _C -> libcudart.so.12; ~1h) =="
# NOT VLLM_USE_PRECOMPILED (fetches the cu13 prebuilt _C). --no-build-isolation reuses our cu12
# torch, so vLLM's build deps (setuptools_scm etc.) must be installed explicitly first.
uv pip install setuptools setuptools_scm wheel ninja cmake packaging
uv pip install -e . --no-build-isolation

echo "== 4. runtime deps (lm_eval[api]+tenacity REQUIRED for GSM8K) =="
uv pip install flashinfer-python==0.6.8.post1 transformers==5.12.1 numpy==2.3.5 \
  requests httpx "lm_eval[api]" tenacity

echo "== 5. VERIFY (torch + torchvision + _C all on CUDA 12; codebook hash) =="
python - <<'PY'
import torch, torchvision, vllm, vllm._C
from vllm.poc import sphere
assert torch.cuda.is_available(), "CUDA not available to torch"
assert sphere._codebook_sha256(sphere.get_sphere_codebook()) == sphere.EXPECTED_CODEBOOK_SHA256, "codebook hash mismatch"
import subprocess, os
so = os.path.join(os.path.dirname(vllm.__file__), "_C.abi3.so")
print(f"OK  torch={torch.__version__}  torchvision={torchvision.__version__}  cuda={torch.version.cuda}  vllm={vllm.__version__}")
print("    _C links:", subprocess.run(["bash","-c",f"ldd {so} | grep -i cudart"],capture_output=True,text=True).stdout.strip())
PY
echo "SETUP_SRC_OK — cu12 source build ready to serve"

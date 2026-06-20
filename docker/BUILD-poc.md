# Building the decode-PoC images

Two images make up a decode-PoC deployment:
- **vLLM engine** (`ghcr.io/axeltec-software/vllm:<tag>`) — built from this vLLM fork
  (`docker/Dockerfile`); the `vllm/poc/*` decode code + `--poc-decode` flag compile in.
- **mlnode** (`ghcr.io/gonka-ai/mlnode`) — all-in-one container (the `uvicorn api.app`
  proxy **plus** the vLLM engine subprocess); built `FROM` the vLLM engine image.

> **Default to the overlay for Python-only changes.** Decode-PoC is all Python, so a
> full from-source rebuild (hours, compiles CUDA) is needed only when C++/CUDA, the
> torch/CUDA version, or the GPU arch list changes. Everything else → overlay (~5s).

## CUDA / arch / driver rules (read first)
- **CUDA major must match the GPU driver.** CUDA 12.x runs on driver ≥525 (e.g. 550);
  **CUDA 13 needs driver ≥580** — don't use it unless the hosts are on 580+.
- **CUDA minor (12.8 vs 12.9) doesn't gate where it runs** — images bundle their CUDA
  runtime and are minor-version compatible; a 12.8 image runs on a 12.9 host and vice
  versa. Pick the minor by **which torch wheel exists**:
  - **v0.20** → torch 2.11 → **CUDA 12.8** (`+cu128`; there is no `+cu129` for 2.11).
  - **v0.15** → torch 2.9.1 → **CUDA 12.9** (matches gonka's `vllm:v0.15.1` + the mlnode
    flash-attn pin).
- **Arch list** = the GPUs you must run on: `8.0`=A100, `8.9`=RTX4000-Ada (local dev),
  `9.0`=H100, `10.0/12.0`=Blackwell/B200 (separate CUDA-13 image).

## 1. vLLM engine — full build (only when CUDA/torch/arch changes)
```bash
# v0.20 (torch 2.11 / CUDA 12.8), for A100 + local + H100:
docker buildx build --target vllm-openai \
  --build-arg CUDA_VERSION=12.8.1 \
  --build-arg torch_cuda_arch_list="8.0 8.9 9.0" \
  --build-arg max_jobs=12 --build-arg nvcc_threads=2 \   # box=20 cores; Dockerfile default max_jobs=2 is ~5x too slow — ALWAYS override
  -f docker/Dockerfile \
  -t ghcr.io/axeltec-software/vllm:v0.20-decode-poc-cu128 --load .
```
Backends compiled/installed in the image: **FlashAttention** (source) + **FlashInfer**
(`flashinfer-jit-cache`, cu128) + **DeepGEMM** (sm_90a → H100 only, MoE/FP8 only).

## 2. vLLM engine — Python-only change (the common case, ~5s)
```bash
docker build -f docker/Dockerfile.poc-overlay \
  --build-arg BASE=ghcr.io/axeltec-software/vllm:v0.20-decode-poc-cu128 \
  -t ghcr.io/axeltec-software/vllm:v0.20-decode-poc-cu128-r2 .
```
Copies repo `.py` over the installed vllm package; leaves compiled CUDA untouched.
(For dev/test, the editable venv `VLLM_USE_PRECOMPILED=1 pip install -e .` makes Python
edits live instantly — no image needed.)

## 3. mlnode image (FROM the vLLM engine)
`mlnode/packages/api/Dockerfile` is parameterized:
```bash
docker buildx build -f mlnode/packages/api/Dockerfile \
  --build-arg VLLM_BASE=ghcr.io/axeltec-software/vllm:v0.20-decode-poc-cu128 \
  --build-arg FLASH_ATTN_WHEEL="" \   # skip the torch2.9 wheel: v0.20 is torch2.11; mlnode doesn't import flash_attn; runtime uses FlashInfer
  -t ghcr.io/axeltec-software/mlnode:v0.20-decode-poc --load mlnode
```
(The `VLLM_BASE` default already points at the v0.20 engine, so the overrides are only
needed to retarget.)

## 4. Push (as axeltec-gonka)
```bash
cat ~/.config/ghcr-axeltec-gonka.token | docker login ghcr.io -u axeltec-gonka --password-stdin
docker push ghcr.io/axeltec-software/vllm:v0.20-decode-poc-cu128
docker push ghcr.io/axeltec-software/mlnode:v0.20-decode-poc
```

## Backend verification (what the image supports)
| backend | A100 (sm80) | H100 (sm90) | local RTX4000 (sm89) | needs |
|---|---|---|---|---|
| FlashAttention | ✅ | ✅ | ✅ | any model — `--attention-backend FLASH_ATTN` |
| FlashInfer | ✅ | ✅ | ✅ | any model — `--attention-backend FLASHINFER` |
| DeepGEMM | ❌ (Hopper+) | ✅ | ❌ | FP8 **MoE** model + `VLLM_USE_DEEP_GEMM=1` (presence-checkable anywhere) |

B200 (Blackwell) needs a **separate** image (arch `10.0`/`12.0`, CUDA 13).

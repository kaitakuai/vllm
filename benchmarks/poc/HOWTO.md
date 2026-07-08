# Running the decode-PoC report harness

`run_scope.sh` boots a vLLM server once per `(model, engine/backend)` config, runs the PoC
measurements over HTTP, and renders one `report.html`. Inputs are two checkpoints of the **same
architecture**: an **honest** model and a cheaper **fraud** stand-in. Examples below use the
Qwen3-235B pair.

## 1. Setup — pick the script by the host driver's CUDA

`nvidia-smi` → top-right "CUDA Version: X.Y".

| driver max CUDA | run |
|---|---|
| **≥ 13.0** (R580+) | `bash benchmarks/poc/scope/setup_box.sh` — published cu13 wheel + PoC overlay |
| **< 13.0** (CUDA 12.x) | `bash benchmarks/poc/scope/setup_box_src.sh` — build vLLM from source (cu12) |

The published `vllm==0.20.1` wheel is compiled for **CUDA 13** and needs an R580+ driver.
`setup_box.sh` preflights the driver and fails closed on older ones. On a CUDA-12.x host use
`setup_box_src.sh` (builds from source against the local CUDA-12 toolkit) or reuse an existing
cu12 venv. Both scripts overlay this repo's PoC Python/codebook onto the vllm install — don't
install vLLM any other way for this fork.

Setup error decoder:
- `libcudart.so.13: cannot open` → cu13 wheel/torch on a CUDA-12 driver → use the source build.
- `RuntimeError: NVIDIA driver too old (found 12040)` → cu13 torch on a CUDA-12 driver → use cu128 torch (the source script does).
- `ModuleNotFoundError: vllm._C` → `VLLM_USE_PRECOMPILED=1 -e .` drops core `_C` → do a real source build.

## 2. Run

```bash
cd benchmarks/poc/scope
bash run_scope.sh <honest-model> <fraud-model> [options]
```

Example — dense-ish model on 4 GPUs, full defaults, push to the artifact bucket:
```bash
bash run_scope.sh \
  Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 \
  chriswritescode/Qwen3-235B-A22B-Instruct-2507-INT4-W4A16 \
  --tp 4 --push
```

Example — MLA-architecture MoE needing extra serve flags:
```bash
bash run_scope.sh <honest> <fraud> --mla --tp 8 --gpu-mem 0.95 --extra "--enable-expert-parallel"
```

### Options

| Flag | Purpose |
|---|---|
| `--tp N` | tensor-parallel degree (default 1) |
| `--gpu-mem G` | GPU memory utilization per server (default 0.90) |
| `--mla` | MLA architecture — collapses the FlashAttn/FlashInfer axis to one `cudagraph` config |
| `--no-fi` | drop the FlashInfer backend axis |
| `--nonces N` / `--seq-len S` / `--max-tokens M` | PoC trajectory shape (defaults 128 / 256 / 256) |
| `--no-perf` | skip performance benchmarking |
| `--no-gsm` | skip the co-existence (GSM8K) experiment |
| `--gsm-limit L` / `--gsm-mml M` | GSM8K sample count / max-model-len (defaults 100 / 2048) |
| `--extra "..."` | extra `vllm serve` flags the deploy needs |
| `--push` | upload the session to the public-read S3 bucket (needs `SUPABASE_SECRET`); omit to render locally |
| `--xhw <peer[,peer2]>` | cross-HW: also validate the peer session(s)' (local `reports/<name>` or S3 name) trajectories with this box's validator |
| `--xhw-only` | skip local generation; only run the `--xhw` cross-validation (for a parallel 2-box verify) |

## 3. What it produces

| Report card | What it measures | Tool |
|---|---|---|
| Performance | throughput (tok/s pure inference, steps/s decode-PoC), cudagraph vs eager | [perfomance_nonces.py](perfomance_nonces.py) |
| Separation (+ per-nonce) | honest vs fraud **producer**, re-checked by one fixed **validator** | [collect.py](collect.py) generate/validate |
| Co-existence | GSM8K accuracy with vs without concurrent PoC load | [quality_gsm8k.py](quality_gsm8k.py) |
| k-distribution | codebook-point histogram, honest vs fraud | from `collect.py` generate outputs |

Every result JSON is stamped with `attention_backend` / `cudagraph_mode` / `profile` and full
provenance (GPU, driver, vLLM commit, dtype/quant, shape). The report header names both models
(`honest <model> vs fraud <model>`).

## 4. Get / re-render the report

`run_scope.sh` writes `benchmarks/poc/scope/reports/<session>/report.html` (`<session>` =
`<honest-slug>__<gpu>__<timestamp>`), and with `--push` mirrors the folder to S3 + injects an
"Artifacts & Reproduce" section. Re-render from saved JSON without a GPU:

```bash
bash benchmarks/poc/scope/s3.sh pull-report <session-name> ./pulled   # public, no key
.venv/bin/python benchmarks/poc/scope/simplify_report.py ./pulled --out ./pulled/report.html
```

Session-folder contents: see [SESSION_LAYOUT.md](SESSION_LAYOUT.md). Per-tool reference:
[README.md](README.md).

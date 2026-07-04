# poc-scope — decode-PoC report harness

## 0. Box setup — run `setup_box.sh` FIRST (robust, verified, idempotent)
On a fresh rented box (after `git clone --branch poc-v0.20-decode-poc-cg … ~/vllm` +
`git checkout <commit>`, and installing `uv`):
```bash
bash ~/vllm/benchmarks/poc/scope/setup_box.sh     # clean venv → matched wheels → overlay → verify
# prints "SETUP_OK" only if torch+torchvision+vllm._C all import on the same CUDA.
```
**THE INVARIANT (every setup failure is a violation of it):**
> `torch` **==CUDA==** `torchvision` **==CUDA==** vLLM's compiled `_C.abi3.so`

The published `vllm==0.20.1` wheel ships a working `_C` for a specific CUDA (currently
**cu130**). `--torch-backend=auto` installs the **whole torch family** for the host CUDA
(driver-580/CUDA-13 host → cu130) → matches the wheel. Traps that cost real time:
- **Never `VLLM_USE_PRECOMPILED=1 pip install -e .`** — for our fork commit it lays down
  `_C_stable_libtorch` but **not** the core `_C` → `ModuleNotFoundError: vllm._C`. Install the
  **published wheel** + overlay our `.py`+`.pt` instead (what `setup_box.sh` does).
- **Never reinstall `torch` alone** — leaves `torchvision` on the old CUDA →
  *"compiled with different CUDA major versions"*. Reinstall the family together (`--torch-backend`).
- **Don't force `--torch-backend=cu128`** — the current wheel is cu130; forcing cu128 gives
  `libcudart.so.13: cannot open` (torch cu128 vs `_C` cu130). Use `auto`.
- Host driver must be new enough for the wheel's CUDA (CUDA-13 wheel needs driver ≥580;
  e.g. Hyperstack's driver-535 images **cannot** run it — verify `torch.cuda.is_available()`).

---

The **single** orchestrator for decode-PoC reports. Lives here (``),
**not** in the vllm git repo, because it carries S3/secrets/experiment ops. It **reuses** the
in-git measurement tools (`vllm-v0.20/benchmarks/poc/{perfomance_nonces,collect,quality_gsm8k,report}.py`)
— no measurement logic is duplicated here.

## Files (one version each — do not fork)
- **`run_scope.sh`** — THE orchestrator. Model is a parameter. Boots each `(model,config)`
  server **once**, runs all its ops via `--url` (minimal reboots), reuses the git tools,
  renders the report.
- **`s3.sh`** — push/pull a report session to the Supabase `gonka-artifacts` bucket
  (push needs `SUPABASE_SECRET`; pull is public/keyless).
- **`inject_s3.py`** — injects the public "Artifacts & Reproduce (S3)" section into `report.html`.
- **`reports/<session>/`** — local output (one folder per run).

## Run
```bash
# model is a parameter (honest = the served model, fraud = a cheaper quant of it):
bash run_scope.sh <honest-model-id> <fraud-model-id>                          # FA/FI model
bash run_scope.sh <honest-model-id> <fraud-model-id> --mla                    # MLA model
# local public proxies for dev on a small GPU:
bash run_scope.sh allenai/OLMoE-1B-7B-0924-Instruct       nm-testing/OLMoE-1B-7B-0924-Instruct-FP8
bash run_scope.sh RedHatAI/Qwen2.5-7B-Instruct-quantized.w8a16  Qwen/Qwen2.5-7B-Instruct-AWQ
# opts: --mla  --tp N  --gpu-mem G  --extra "..."  --nonces N  --max-tokens M  --xhw <peer>  --no-gsm  --push
```
`--mla` switches the config set (MLA forces TRITON_MLA → no FA/FI axis).

## What it produces
- **PERF** (cudagraph vs eager, per backend): `perf_<cfg>.{poc,chat}.json`
- **SEPARATION — 5 logical pairs** (validator fixed = honest @ production **cg-FA**; vary the
  reference one axis at a time, one direction): `val_<val>__gen_*.json`
  1. `V(cg,FA) ⇐ H(cg,FA)` — honest floor
  2. `V(cg,FA) ⇐ H(cg,FI)` — backend drift (FA↔FI)
  3. `V(cg,FA) ⇐ H(eager,FA)` — cudagraph drift (cg↔eager)
  4. `V(cg,FA) ⇐ F(cg,FA)` — fraud detection
  5. `V(cg,FA) ⇐ F(cg,FI)` — fraud cross-backend
  (Validator is **always honest**; there is no fraud-validator. MLA reduces to floor/cudagraph/fraud.)
- **GSM8K** co-existence (PoC on vs off): `gsm_<cfg>_{on,off}.json`
- `report.html`, `REPRODUCE.md`, `.manifest`

## Grouped boots (minimize reboots)
fraud@cgFA · fraud@cgFI · honest@cgFI(perf+ref) · honest@eagerFA(perf+ref) · honest@eagerFI(perf) ·
**honest@cgFA** (perf + baseline ref + all 5 validations + GSM on/off). ≈6 boots instead of ~20.
Each server is killed by its own process group (`kill -- -PGID`) — safe on the shared GPU.

## Storage (one run = one folder = one S3 prefix)
- local: `reports/<session>/`   (session = `<honest-slug>__<gpu>__<timestamp>`)
- S3 (public-read): `reports/<session>/` in `gonka-artifacts`
- push (needs secret):  `bash s3.sh push-report reports/<session> <session>`  (or `run_scope.sh --push`)
- pull (keyless):       `bash s3.sh pull-report <session> ./<session>`
- public URL:           `…/storage/v1/object/public/gonka-artifacts/reports/<session>/report.html`

## Notes
- Provenance in `--url` connect-mode is stamped from the `--profile` (engine/backend/profile);
  `run_scope.sh` also post-stamps `attention_backend`/`cudagraph_mode` so the report labels correctly.
- The in-git `benchmarks/poc/run_model_report.sh` is the simple per-call variant; for grouped
  experiment runs use **this** harness.

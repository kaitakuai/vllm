# Decode-PoC benchmark tooling

> **Building the images?** See [docker/BUILD-poc.md](../../docker/BUILD-poc.md) — vLLM engine + mlnode, with the
> fast overlay path for Python-only changes.

## Purpose
Client/server tooling to measure **decode-stage Proof-of-Compute** on vLLM across
models, hardware, and engine configurations. The **server** (any box, local or
rented) runs only **ML node + vLLM**; the **client** (here) drives it over HTTP and
writes self-describing **result files**. Collection and analysis are separate steps:
collect on each box, then compare **offline** — honest/fraud separation, throughput,
and cross-hardware/cross-engine comparison all come from the same files.

Every result file carries **full provenance** (GPU + driver, vLLM commit, cudagraph
mode, attention backend, dtype, quantization, model, seq_len/max_tokens), so any two
runs are directly comparable regardless of where or when they were produced.

Three collectors drive the server; each writes its own **role-tagged** result file.
`analyze.py` then aggregates whatever files you give it — one tool per table:
```
  client (here)                                server (any box: local or rented)
  ─────────────                                ─────────────────────────────────
  collect.py            ── /generate,/validate ─▶ ML node ─▶ vLLM (decode-PoC)
  perfomance_nonces.py  ── PoC loop + chat ───────▶      └─▶ trajectory / timing
  quality_gsm8k.py      ── GSM8K + PoC load ────▶          + provenance
        │                                                       │
        └───────── role-tagged result files (runs/*.json) ◀─────┘
                                  │
                                  ▼
            analyze.py ─▶ SEPARATION  (← collect.py validate runs)
                          PERF        (← perfomance_nonces.py)
                          GSM8K       (← quality_gsm8k.py)
```

## Tooling
| Tool | What it does |
|---|---|
| `collect.py` | Collect data from a server: `--mode generate` (produce a k-trajectory) or `--mode validate --ref F` (re-run teacher-forced against F → mismatches). Writes one result file with data + timing + provenance. |
| `analyze.py` | Offline, no server. Loads many result files → **SEPARATION** matrix, **PERF** table, **GSM8K** table. Cross-hardware/engine comparison = feed in more files. |
| `pair_report.sh` | One command for an honest/fraud pair (generate ×2, validate ×4, analyze). |
| `perfomance_nonces.py` | Sustained decode throughput, **one tool for both** `--mode poc` (nonces/min, steps/s) and `--mode chat` (req/min, tokens/s) — or `--mode both`. req/min ≈ nonce/min (same unit: one decode sequence), so the report shows how closely PoC tracks real-inference capacity. |
| `quality_gsm8k.py` | GSM8K accuracy with/without concurrent PoC load (co-existence). |
| `poc_validation.py` | Shared core: deploy/serve, profile resolution, request builder, provenance. |
| `poc_configs.json` | Named engine **profiles** (graph/eager × attention backend, …). |

Typical flow:
```bash
# collect (client → server; --url for a remote box, omit to auto-boot vLLM locally)
collect.py --mode generate --model <M> --url $S --save runs/gen_M.json
collect.py --mode validate --model <M> --ref runs/gen_M.json --url $S --save runs/val_M.json
# analyze (offline, over any set of result files)
analyze.py runs/*.json
```

## Flexibility
The tooling covers arbitrary configurations, not a fixed list:
- **Any model, any box, no hardcoded IP** — `--url` points at the ML node on any box;
  `--target vllm|mlnode` selects the engine vs the on-chain proxy.
- **Engine configs are named profiles** (`poc_configs.json`, `--profile`): graph vs
  eager, attention backend (FlashAttention / FlashInfer), async vs sync, plus any
  extra serve flags. New profiles need no code change. A pair can use two profiles
  (`--gen-profile` / `--val-profile`) to measure cross-engine divergence (graph mode is
  small/tolerated; a different attention backend is large — pin the backend).
- **Shape is configurable** — `--seq-len`, `--max-tokens`, `--nonces` (defaults match
  production) for varied input/output lengths.
- **Comparison is provenance-driven** — each file records exactly what ran, so adding
  a GPU, model, or backend to the matrix needs no tooling change: collect more files,
  re-run `analyze.py`.

## Metrics we collect
**Separation / robustness** — does validation distinguish honest from fraud?
Each decode step snaps the hidden state to a discrete codebook index (`sphere_k`).
Validation re-runs teacher-forced against a reference trajectory and counts steps
whose `sphere_k` differs (`n_sphere_mismatches`):
```
rate = Σ n_sphere_mismatches / (nonces × (max_tokens + 1))
fraud_detected = rate > p_mismatch        # p_mismatch: chain governance param, default 0.1
```
`analyze.py` prints the **SEPARATION** matrix: validator ⇐ prover → `rate` + verdict.
Honest (same model) ≈ 0; fraud (cheaper/different model) is high (≈ 0.33 in our runs).

**Performance & inference-readiness fidelity** — `perfomance_nonces.py` measures both
PoC throughput (`nonces/min`, `steps/s`) and real chat inference (`req/min`,
`tokens/s`) at the same concurrency. Because a PoC nonce and a chat request are the
same unit (one decode sequence), their ratio shows **how closely decode-PoC tracks the
node's real-inference capacity** — the proof that PoC measures genuine readiness to
serve, not an unrelated synthetic load. Reported in the **PERF** table per (mode, gpu,
engine, max_tokens) with the fidelity headline.

**Co-existence (quality)** — GSM8K strict/flexible accuracy with PoC load vs a
`--disable_poc` baseline (`quality_gsm8k.py`), in the **GSM8K** table. Shows whether
running PoC alongside inference changes answer quality.

These are independent axes: a separation `rate` (e.g. honest 0.1%) and a throughput
number (e.g. nonces/min) measure different things and are never mixed.

## Example output
*(formats are real; columns trimmed to fit. Separation/perf numbers are from actual
runs; GSM8K numbers are illustrative.)*

`perfomance_nonces.py --mode both` — PoC and real chat in one server lifetime:
```
=== poc throughput ===
req/min = 611   steps/s=662   (128 nonces in 12.6s, concurrency=32, max_tokens=64)
=== chat throughput ===
req/min = 805   tokens/s=859   (160 requests in 11.9s, concurrency=32, max_tokens=64)
  provenance: gpu=RTX 4000 Ada  attention_backend=FLASH_ATTN  cudagraph_mode=FULL_AND_PIECEWISE  dtype=bfloat16  quant=compressed-tensors
```
PoC nonce/min and chat req/min are the same unit (one decode sequence at concurrency 32),
so their ratio shows **how closely decode-PoC tracks real-inference capacity** — the
"inference-readiness fidelity" headline in the report.

`analyze.py runs/*.json` — the tables, each from its own collector:
```
=== SEPARATION (validator <= prover) ===          # from collect.py validate runs
config     validator          prover             rate     fraud  kind
cudagraph  Qwen2.5-7B-w8a16   Qwen2.5-7B-w8a16   0.140%   False  honest
cudagraph  Qwen2.5-7B-w8a16   Qwen2.5-7B-AWQ    33.800%   True   fraud
SEPARATION: PASS  (honest must be fraud=False, fraud must be fraud=True)

=== PERF (req/min — PoC & real inference) ===      # from perfomance_nonces.py
mode            gpu            engine     max_tok  req/min  tok/s·steps/s
real inference  RTX 4000 Ada   cudagraph       64      805     859 tok/s
PoC             RTX 4000 Ada   cudagraph       64      611     662 steps/s
real inference  RTX 4000 Ada   eager           64      712     760 tok/s
PoC             RTX 4000 Ada   eager           64      540     585 steps/s
# fidelity (cudagraph): PoC tracks inference at 76%

=== GSM8K (accuracy) ===                           # from quality_gsm8k.py
model             gpu            engine     poc_mt    N   strict    flex
Qwen2.5-7B-w8a16  A100           cudagraph     256  full  84.20%  85.10%
Qwen2.5-7B-w8a16  A100           (baseline)      0  full  84.30%  85.20%
```
Read it as: separation has a wide honest↔fraud gap (PASS); PoC nonce/min sits close to
chat req/min (decode-PoC measures real-inference readiness); cudagraph beats eager; and
GSM8K with PoC load ≈ the `--disable_poc` baseline (co-existence holds).

## Parameters (defaults match production)
| flag | default | meaning |
|---|---|---|
| `--nonces` | 32 | sample size = one production batch (`batch_size=32` / forward) |
| `--seq-len` | 256 | **prefill** length (`poc_seq_len`) — must match the deployed server |
| `--max-tokens` | 256 | **decode steps** (`poc_max_tokens`); trajectory has `max_tokens+1` entries |
| `--p-mismatch` | 0.1 | fraud cutoff (chain governance param) |

`k_dim` is fixed at 12. `seq_len` is prefill only; decode adds `max_tokens`, so the
engine allocates `seq_len + max_tokens` KV upfront — keep `seq_len + max_tokens ≤
--max-model-len`. `seq_len`/`max_tokens`/`k_dim` are artifact-defining: request values
must match the deployed server's `poc_*` config or the request is rejected.

## Test 1 — separation (honest vs fraud)
Does validation tell a genuine run from a cheaper/different model? Run a pair:
```bash
# prod defaults, so no shape params needed:
pair_report.sh <honest-model> <fraud-model> runs/pairAB --profile cudagraph --url $S
# -> runs/pairAB/{gen_*,val_*}.json + report.txt
```
Measured (100 samples, AWQ vs w8a16, cudagraph): honest **0.14%** / fraud **33.8%** →
clean separation.

## Test 2 — co-existence (GSM8K)
A **separate** question from separation: does running PoC alongside real user inference
degrade answer quality? Run GSM8K twice — once with PoC load, once without — and compare
accuracy:
```
  real GSM8K questions ──▶ vLLM ──▶ answers ──▶ accuracy ─┐
                            ▲                              ├─▶ same? ⇒ co-existence OK
  concurrent PoC load ──────┘ (N requests in flight)      │
  baseline (--disable_poc): no PoC load ──▶ accuracy ─────┘
```
```bash
quality_gsm8k.py --url $S --model_name <M> --profile cudagraph                 # PoC load on
quality_gsm8k.py --url $S --model_name <M> --profile cudagraph --disable_poc   # baseline
```
It prints a comparison table of the two runs (numbers illustrative):
```
| Configuration | Batch Size | PoC Batch | Decode (max_tokens) | Accuracy (Strict/Flexible) |
|---------------|------------|-----------|---------------------|----------------------------|
| Qwen2.5-7B    | 32         | 0         | 0                   | 0.8430 / 0.8520            |
| Qwen2.5-7B    | 32         | 4         | 256                 | 0.8420 / 0.8510            |
```
PoC Batch 0 = baseline (no PoC); the next row runs with PoC load. Accuracy within noise
⇒ co-existence holds. `--limit N` runs a fast subset; omit for the full set.

## Reports & session discipline
`run_model_report.sh` runs the full test matrix for **one model** and renders a
self-contained HTML report (`report.py`). Result files are organized **one session per
model** — never a shared flat `runs/`:

```
runs/<model-slug>__<gpu-slug>__<YYYYMMDD-HHMMSS>/   # one session = one model, one box, one run
    perf_*.json  gen_*.json  val_*.json  gsm_*/      # all result files for that model
    report.html                                       # rendered report
```
```bash
run_model_report.sh <honest-model> [fraud-model] --url $S            # full 24-cell matrix
run_model_report.sh <honest-model> [fraud-model] --scope quick       # fast diagonal sanity
report.py runs/<session>/ --out runs/<session>/report.html           # (re-)render one session
report.py runs/ --out all.html                                       # combine many sessions (grouped per model)
```
The report has these sections — **Performance** (PoC nonce/min **and** real chat req/min
side by side, cudagraph-vs-eager speedup, plus the **inference-readiness fidelity**
headline: how closely PoC tracks real inference), **Separation** (honest vs fraud),
**p_mismatch calibration** (the feasible per-model threshold window derived from measured
rates — `p_mismatch` is not baked in; prod loads it per-model from chain
`PoCStatTestParams`), and **GSM8K co-existence** (pure chat vs +PoC, on cudagraph and
eager) — plus a PASS/FAIL chip, per-section **coverage counts**, and a metric glossary.
`report.py` groups by model from each file's provenance, so even a mixed pile renders one
section per model — but **the runner keeps sessions separate on disk**.

**Example report:** [`example_report.html`](example_report.html) — a real prod-config run
(Qwen2.5-7B w8a16 honest vs AWQ fraud, RTX 4000 Ada): the perf section with PoC vs real
chat throughput and the inference-readiness fidelity headline, cudagraph-vs-eager speedup,
separation ladder (honest same-backend ≈0.3% / cross-backend ≈17% vs fraud ≈33%), the
p_mismatch calibration window, and 4-run GSM8K co-existence (PoC ≈ baseline on both
engines). Self-contained HTML — open it directly.

## Configuring vLLM (profiles)
Engine settings are named profiles in `poc_configs.json`, selected with `--profile`
(honored by `collect.py` / `perfomance_nonces.py` / `quality_gsm8k.py`;
`pair_report.sh` forwards it). A profile sets `eager` (graph vs eager) and `args`
(extra serve flags). Attention backend is the `--attention-backend` serve flag (works
local and remote).
```bash
collect.py --mode generate --model M --profile eager-flashinfer --save g.json
pair_report.sh <honest> <fraud> runs/x --gen-profile cudagraph --val-profile eager
```
Built-in: `cudagraph` (default), `eager`, `cg-flashattn`, `eager-flashattn`,
`cg-flashinfer`, `eager-flashinfer`, plus `cg-noasync` / `eager-noasync`. vLLM runs
**async scheduling by default** and decode-PoC supports it (no `--no-async-scheduling`
required); the `*-noasync` profiles exist only to measure the async-vs-sync delta.
`deploy_from_args` verifies the engaged attention backend matches the request and
records the effective config into every result file.

> `--url` runs need only HTTP to the server. Local auto-boot needs `.venv/bin` on
> `PATH` (the harness shells out to `vllm serve`). Deps: `requirements.txt`.

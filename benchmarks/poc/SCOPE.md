# Decode-PoC — testing methodology & scope

This document explains, from scratch, what decode-stage Proof-of-Compute is, how a test
works, what we vary it over, and what we measure. The tooling that produces every
result is in [README.md](README.md).

## The problem
In a decentralized inference network, GPU providers are paid to serve a specific large
model (say, a 200B+ MoE). Nothing physically stops a dishonest provider from quietly
running a smaller, cheaper model instead and keeping the difference. So the network
needs a cheap, automatic way to check that a provider really ran **the assigned model
on real hardware** — a *Proof-of-Compute (PoC)*.

## The idea
Give the provider a fixed pseudo-random task and have the model emit a compact
**fingerprint of its own computation** that (a) only the genuine model produces and
(b) can only be produced step-by-step. A validator then re-computes that fingerprint
independently and checks how well it matches.

How the fingerprint is built:
- The provider is fed a deterministic random prompt (`seq_len` random embeddings) and
  runs `max_tokens` decode steps.
- At each step the hidden state is projected and **snapped to the nearest point in a
  small fixed codebook** — an integer `sphere_k`. The sequence of those integers is the
  fingerprint, `k_points_steps` (length `max_tokens + 1`).
- Each step's random projection is **seeded from the previous step's `sphere_k`**
  ("chaining"). Because step *t* depends on step *t−1*, the trajectory can't be
  precomputed or parallelized — it must run sequentially on the real model.

Why this catches fraud: a cheaper or different model (or the wrong precision) lands on
different codebook points. With chaining, a single divergence cascades down the rest of
the trajectory, so an honest run and a fraudulent one separate cleanly instead of
differing by a hair.

```
  random prompt (seq_len)
        │
        ▼
  PROVER runs max_tokens decode steps:
        ┌─────────────────────────────────────┐
        │  step t:  hidden ─▶ sphere_k[t]      │
        │  seed(t+1) ◀── chained to sphere_k[t]│   (must run sequentially)
        └─────────────────────────────────────┘
        │
        ▼
  reference trajectory  k_points_steps   (the fingerprint)
        │
        ▼
  VALIDATOR re-runs teacher-forced ─▶ counts steps whose sphere_k differs
        │
        ▼
  rate = Σ mismatches / (nonces × (max_tokens+1))
        │
        ├─ rate ≤ p_mismatch  ─▶  HONEST  (same model)
        └─ rate >  p_mismatch  ─▶  FRAUD   (cheaper/different model)
```

## How a single test works
1. **Generate** — a prover produces its fingerprint for a batch of `nonces` (one
   production batch = 32 independent tasks). → `collect.py --mode generate`.
2. **Validate** — re-run the same tasks **teacher-forced** against the reference
   fingerprint and count the steps whose `sphere_k` differs (`n_sphere_mismatches`).
   → `collect.py --mode validate --ref <gen>`.
3. **Decide**:
   ```
   rate = Σ n_sphere_mismatches / (nonces × (max_tokens + 1))
   fraud_detected = rate > p_mismatch        # p_mismatch: chain governance param, default 0.1
   ```
   - **Honest** — validator runs the *same* model that generated the reference →
     `rate ≈ 0`.
   - **Fraud** — validator runs a *cheaper/different* model → `rate` high.
   - **Separation** — the gap between the two around `p_mismatch`. A test passes when
     honest sits well below the threshold and fraud well above it.

What a passing result looks like (`analyze.py` over the collected files):
```
=== SEPARATION (validator <= prover) ===
config    validator           prover              rate     fraud  kind
cudagraph Qwen2.5-7B-w8a16    Qwen2.5-7B-w8a16    0.140%   False  honest
cudagraph Qwen2.5-7B-w8a16    Qwen2.5-7B-AWQ     33.800%   True   fraud

SEPARATION: PASS  (honest must be fraud=False, fraud must be fraud=True)
```
Honest 0.14% ≪ 10% ≪ fraud 33.8% — a wide, unambiguous gap.

## Determinism & config sensitivity
The trajectory is reproducible at a **fixed** config, but config-sensitive in degrees —
the divergence between two *honest* runs grows with how different their configs are:

> same config  <  cudagraph vs eager (same backend)  <  different attention backend  <  different model (fraud)

- **async on ↔ off** (same engine + backend): **byte-identical** — the hard gate we
  enforce (`tests/poc/integration/test_async_equivalence.py` + byte-comparing
  `collect.py` outputs). Async must not change the artifact.
- **graph mode** (cudagraph ↔ eager, same backend): small — stays under a calibrated
  `p_mismatch`, so it's **tolerated**.
- **attention backend** (FlashAttention ↔ FlashInfer): large — the validator **must use
  the prover's backend** (pin it), or an honest run reads as fraud.
- **different model**: large — this is the fraud signal.

The goal is a **separable gap**: honest below the threshold, fraud above, backend pinned.
The actual per-model rates and the feasible `p_mismatch` window are **measured** (report
SEPARATION + calibration), not hardcoded.

Every result file also records **full provenance** (GPU + driver, vLLM commit, engine
mode, attention backend, dtype, quant, model, shape), so any two runs are directly
comparable and there is no ambiguity about what produced a number.

## Co-existence test (GSM8K)
A **separate** question from honest/fraud: does running PoC alongside real user
inference disturb normal answer quality? Run GSM8K twice — once with PoC load, once
without (`--disable_poc`) — and compare accuracy. Co-existence holds when the two are
within noise.
```
  real GSM8K questions ──▶ vLLM ──▶ answers ──▶ accuracy ─┐
                            ▲                              ├─▶ same? ⇒ co-existence OK
  concurrent PoC load ──────┘ (N requests in flight)      │
  baseline: --disable_poc (no PoC load) ──▶ accuracy ─────┘
```
Tool: `quality_gsm8k.py` (real GSM8K via lm-eval + a background PoC load loop). It
prints a comparison table of the two runs (numbers below illustrative):
```
| Configuration | Batch Size | PoC Batch | Decode (max_tokens) | Accuracy (Strict/Flexible) |
|---------------|------------|-----------|---------------------|----------------------------|
| Qwen2.5-7B    | 32         | 0         | 0                   | 0.8430 / 0.8520            |
| Qwen2.5-7B    | 32         | 4         | 256                 | 0.8420 / 0.8510            |
```
PoC Batch 0 = baseline (no PoC); the next row runs with PoC load. Accuracy is within
noise ⇒ co-existence holds.

## What's being tested (the implementation under test)
- **CUDA-graph support** for decode PoC — capture/replay on the *static* parts of the
  model to cut per-step overhead; the *dynamic* parts (attention, Haar-rotation hooks)
  stay out of capture; per-layer Householder is in-graph (`vllm/poc/native.py`).
- **The vLLM 0.20 port** (`poc-v0.20-decode-poc-cg`) — GPU-native chaining,
  async-scheduling safe (no `--no-async-scheduling`), pure dynamic KV.
- **The ML-node on-chain proxy** (`pow_v2_routes.py`) — exposes the trajectory; the
  chain consumes `fraud_detected` → `validated_weight`.

## What we vary (scope)
The separation and performance results are produced across a set of large models and
GPU configurations under test. **Model, hardware, tensor-parallel size,
attention backend and all other serve settings are runtime parameters** (passed per run
and recorded into the result file) — nothing model- or deployment-specific is hardcoded
in the tooling.

Beyond model and hardware, each run is one combination of these axes (the tooling
selects them per run and records the effective values into the result file):

| Axis | Values | Selected by |
|---|---|---|
| Model | table above | `--model` |
| Hardware | A100 / H100 / H200 / B200 | the box (`--url`) |
| Parallelism | TP degree (1 … N) | server `--tensor-parallel-size` |
| Engine | cudagraph / eager | profile `eager` flag (`--profile`) |
| Attention / KV | FlashAttention / FlashInfer | profile `--attention-backend` |
| Scheduling | async (default) / sync | profile (`*-noasync`) |
| Shape | seq_len, max_tokens, nonces | `--seq-len` / `--max-tokens` / `--nonces` |
| dtype / quant | model-defined | recorded from the vLLM log |

The set under test is **not a tooling limit**: any combination of the axes
above runs through the same `--profile` / `--url` / shape flags with no code change, and
`analyze.py` compares across all of them offline via per-file provenance.

## What we measure (metrics → tools)
| Result | Metric | Tool |
|---|---|---|
| Fraud/honest separation | `rate`, `fraud = rate > p_mismatch` | `collect.py` generate/validate → `analyze.py` SEPARATION (one command: `pair_report.sh`) |
| Determinism | fingerprint byte-identical for async on↔off (same engine+backend) | `tests/poc/integration/test_async_equivalence.py` + `collect.py` byte-compare |
| Config sensitivity | divergence grows: same < graph-mode (tolerated) < backend (pin) < fraud; goal = separable gap | `collect.py` validate matrix → `report.py` SEPARATION + calibration |
| Throughput | nonces/min, steps/s | `perfomance_nonces.py --mode poc` → `analyze.py` PERF |
| Inference-readiness fidelity | PoC nonce/min vs real chat req/min (same unit ⇒ closeness shows PoC measures real serving capacity) | `perfomance_nonces.py --mode both` → PERF fidelity headline |
| Co-existence | GSM8K strict/flexible accuracy ± PoC load | `quality_gsm8k.py` (`--disable_poc` baseline) → `analyze.py` GSM8K |
| Correctness suite | unit + integration (separation, equivalence, multi-batch slot-reuse) | `tests/poc/unit/*`, `tests/poc/integration/*` |

`nonces/min` (throughput) and `rate` (separation) are separate axes and are never mixed:
one says *how fast*, the other says *honest or fraud*.

## Tooling reference
`benchmarks/poc/` in this repo:

| File | Purpose |
|---|---|
| `collect.py` | data collection (client/server): `--mode generate` / `--mode validate`, full provenance |
| `analyze.py` | offline analysis: SEPARATION matrix, PERF table, GSM8K accuracy, cross-hardware |
| `pair_report.sh` | one-command honest/fraud pair (generate ×2, validate ×4, analyze); `--gen-profile`/`--val-profile` for cross-engine |
| `perfomance_nonces.py` | sustained throughput — `--mode poc` (nonces/min, steps/s), `--mode chat` (req/min, tokens/s), or `--mode both`; powers the inference-readiness fidelity comparison |
| `quality_gsm8k.py` | GSM8K accuracy ± concurrent PoC load |
| `poc_validation.py` | shared core: deploy/serve, profile resolution, request builder, provenance |
| `poc_configs.json` | named engine profiles (graph/eager × attention backend) |
| `requirements.txt` | tooling deps (`requests`; gsm8k: `aiohttp`, `numpy`, `lm_eval`) |

ML node: `mlnode/packages/api/src/api/inference/pow_v2_routes.py` (on-chain decode-PoC verification proxy).

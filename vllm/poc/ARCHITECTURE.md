# Decode-PoC + chat co-existence — architecture

How decode-PoC (Proof-of-Compute) runs **alongside** normal chat inference on vLLM, and
how it reaches the gonka ML node and chain. Read top-to-bottom = request lifecycle.

## Two request types
- **Chat** — normal request: prefill, decode, sample a token per step, return text. Uses
  vLLM's standard sampler + output path.
- **PoC** — per-nonce *deterministic* forward. Each decode step snaps the hidden state to a
  discrete codebook index `sphere_k`, chained (`prev_k` seeds the next step). Produces a
  **vector trajectory** (`k_points_steps`), **not text**, delivered on a separate channel
  (`poc_outputs`).

**Core invariant: PoC produces no token.** It must never depend on or pollute the token
sampler/output path.

## Building blocks (by file)

| Stage | File | Role |
|---|---|---|
| Entry | `entrypoints/openai/api_server.py` | registers PoC routes (`add_api_route`); `--poc-decode` flag |
| Route | `poc/routes.py` | `/api/v1/pow/generate` → builds `PoCParams`, calls generate_queue |
| Fan-out | `poc/generate_queue.py` | one engine request per nonce (`compute_one`), `generate(poc_params=...)`, no sampling_params; assembles artifact from the emit-once `poc_output` |
| Schedule | `v1/core/sched/scheduler.py` + `poc/mixed_decode.py:decode_only_mixing_gate` | decode-only mixing: chat+PoC share a forward only when both decode; prefills isolated. PoC gets paged KV like chat (pure dynamic KV) |
| Inputs | `poc/mixed_decode.py:build_unified_mixed_batch_inputs` | fused chat+PoC embeds/positions in `input_batch` order; PoC decode embeds from chained `prev_k` (`poc/gpu_random.py`) |
| Forward | `v1/worker/gpu_model_runner.py` | PoC batches force **eager** (mixed prefill isn't cudagraph-able). Two-phase under async: `execute_model` (dispatch) → `sample_tokens` (sample/bookkeep) |
| Sample | `gpu_model_runner._sample` + `_bookkeeping_sync` | Chat+PoC sample as one natural-order batch; PoC's token is ignored. Output arrays (`req_id_to_index`, `sampled_token_ids`) keep PoC rows at their **natural input_batch index** — never renumbered. All-PoC batch skips the sampler. |
| PoC output | `poc/mixed_decode.py:process_poc_outputs_from_hidden` | hidden → `sphere_k` → accumulate trajectory on-device → emit artifact once into `poc_outputs` |
| Deliver | `scheduler.update_from_output` | PoC on its own branch (drain `poc_outputs`, artifact-driven finish); chat reads its sampled token |
| ML node | gonka `decentralized-api` `pow_v2_routes.py` (Go) | proxy: validates v2 body, forwards to vLLM `/api/v1/pow/*`. Decode integration point |
| Chain | gonka `poc/decode.go` (Go) | trajectory packed in `PoCArtifactV2.Vector` (no proto change); teacher-forced validation, `fraud = mismatch_rate > p_mismatch` |

## Why "PoC off the token path" matters (the async fix)
The scheduler reads a chat token as `sampled_token_ids[req_id_to_index[req_id]]`; PoC is
handled on its own branch and never reads that array. Earlier bookkeeping "cleaned up" the
output by dropping PoC and **renumbering** `req_id_to_index` to a chat-only `0..n`, while the
token tensor stayed in full input_batch order. That was self-consistent in **sync** (the same
block also filtered the token list), but in **async** the token list is empty there (tokens
resolve later from the full GPU tensor), so the list-filter was a no-op and only the map got
renumbered → chat at input_batch index *k* was told it lived at *k'* → it read a PoC row's
ignored token (id 0 = `!`), corrupting output (gsm8k ~0.74→0.36). It stayed hidden because
decode-PoC ran `--no-async-scheduling`. **Fix:** leave PoC rows in the output arrays at their
natural input_batch index — never renumber. One order, nothing to desync.

Same disease, second field: `_sample` read `input_batch.sampling_metadata` **live**, but the
next step's `execute_model` rebuilds it before this step's `sample_tokens` runs → metadata for
the wrong-sized batch (penalty-tensor crash, or silently wrong temp/top_p). **Fix:** snapshot
`sampling_metadata` into `ExecuteModelState` (it's reassigned, not mutated, so the captured
reference is this step's) and sample against the snapshot. General rule: anything
`sample_tokens` needs about this step's batch comes from the `execute_model` snapshot, never
read live. Sampling is post-forward (outside the cudagraph), so none of this affects capture.

## Invariants
- PoC never enters the token sampler/output path.
- Decode-only mixing; prefills run isolated; PoC batches eager.
- Pure dynamic KV; chat KV untouched by PoC.
- Determinism: cudagraph ≡ eager, but **attention-backend-specific** (pin FA vs FI for
  cross-node validation).

## Tests
- `tests/poc/integration/test_chat_coexistence_concurrent.py` — concurrent chat+PoC, no corruption.
- `tests/poc/integration/test_kv_cache_integrity.py`, `test_validation_isolation.py`.
- Co-existence **accuracy**: `benchmarks/poc/quality_gsm8k.py` (PoC-on vs `--disable_poc`).

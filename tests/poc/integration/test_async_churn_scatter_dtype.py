"""Regression: decode-PoC survives a CHURNING async batch (the TP=4 235B crash).

The bug this guards (root cause OURS; crash site stock vLLM):
  The all-PoC sampler stub in `gpu_model_runner` (`if not chat_rows:` branch)
  synthesized `sampled_token_ids` as `torch.long` (int64). vLLM's token contract
  is int32 (`sampler.py` casts to int32; `input_ids` buffer is int32). Under async
  scheduling this stub becomes `prev_sampled_token_ids` and is written into
  `input_ids` next step. Two paths do that write:
    - stable/contiguous batch -> `input_ids.gpu.copy_(...)`  (silently casts -> OK)
    - reordered batch        -> `input_ids.gpu.scatter_(...)` (EXACT dtype required)
  Only `scatter_` is reached when a request is removed from the MIDDLE of the
  running batch (survivors become non-contiguous). int64 src vs int32 dst ->
  `RuntimeError: scatter(): Expected self.dtype to be equal to src.dtype` -> all
  workers die -> EngineCore dead -> empty artifacts.

Why earlier async tests missed it: `test_async_multibatch` sends equal-length
nonces that start+finish together -> no mid-batch reorder -> never `scatter_`.
It first bit us at 235B/TP=4 only because that was the first large, SUSTAINED,
churning async run (128 nonces, more nonces than slots -> staggered admission ->
staggered finishes -> mid-batch removal -> reorder). It is NOT TP-specific: the
crash is the per-rank local `scatter_`. This test reproduces it at TP=1 by
forcing churn with `--max-num-seqs` < nonces so nonces are admitted in waves and
finish at staggered steps.
"""
import httpx
import pytest

from tests.poc._server import PoCTestServer
from tests.poc.utils import poc_request_body

MODEL = "Qwen/Qwen3-0.6B"
POC_URL = "/api/v1/pow/generate"
TIMEOUT = 300
SPHERE_POINTS = 16
# async (NO --no-async-scheduling) + tiny running batch so >N nonces are admitted
# in waves -> staggered starts -> staggered finishes -> mid-batch reorder -> the
# strict scatter_ path. Pre-fix this crashes the engine on a churning step.
BASE_ARGS = ["--gpu-memory-utilization", "0.3", "--max-model-len", "1024",
             "--enforce-eager", "--max-num-seqs", "2"]


@pytest.mark.integration
def test_async_churn_scatter_dtype_no_crash():
    """12 decode-PoC nonces through a 2-slot async server -> nonces are admitted in
    waves and finish at staggered steps, forcing the reordered scatter_ path that
    pre-fix raised the int64/int32 dtype mismatch. Every nonce must still return a
    full in-range k_points_steps (len == max_tokens+1)."""
    mt = 8
    nonces = list(range(12))  # 12 > max_num_seqs(2) -> sustained wave admission
    with PoCTestServer(MODEL, list(BASE_ARGS)) as srv:
        body = poc_request_body("0xchurn_scatter", nonces, MODEL,
                                wait=True, seq_len=64, max_tokens=mt)
        r = httpx.post(f"{srv.url_root}{POC_URL}", json=body, timeout=TIMEOUT)
        r.raise_for_status()
        d = r.json()
        assert d.get("status") == "completed", d
        arts = {a["nonce"]: a for a in d["artifacts"]}
        assert set(arts) == set(nonces), (set(arts), set(nonces))
        for n in nonces:
            steps = arts[n]["k_points_steps"]
            assert len(steps) == mt + 1, (n, steps)
            assert all(0 <= k < SPHERE_POINTS for k in steps), (n, steps)

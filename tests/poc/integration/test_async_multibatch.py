"""Regression: decode-PoC survives MULTIPLE /generate batches on one async server.

The bug this guards: under async scheduling the PoC decode slot pool frees and
REUSES slots across requests; on the 2nd+ batch a reused slot carried a stale
`input_ids` (PoC rows are excluded from sampling, so async's prev_sampled_token_ids
forwarding left an out-of-vocab id) -> token-embedding gather index-out-of-bounds
-> device-side assert -> EngineCore dead.

Why earlier tests missed it: every other PoC test sends only ONE /generate per
server boot, so slots are never reused. Only sustained multi-batch load (the perf
tool) ever hit batch #2. This sends several sequential batches to ONE server with
DISTINCT nonces to force slot free+realloc, and asserts every batch still returns a
full artifact per nonce (no crash, no empty result).
"""
import httpx
import pytest

from tests.poc._server import PoCTestServer
from tests.poc.utils import poc_request_body

MODEL = "Qwen/Qwen3-0.6B"
POC_URL = "/api/v1/pow/generate"
TIMEOUT = 240
# Small footprint + async (NO --no-async-scheduling) so this exercises the async path.
BASE_ARGS = ["--gpu-memory-utilization", "0.3", "--max-model-len", "1024",
             "--enforce-eager"]
SPHERE_POINTS = 16


def _generate(srv, bh, nonces, mt):
    body = poc_request_body(bh, nonces, MODEL, wait=True, seq_len=64, max_tokens=mt)
    r = httpx.post(f"{srv.url_root}{POC_URL}", json=body, timeout=TIMEOUT)
    r.raise_for_status()
    d = r.json()
    assert d.get("status") == "completed", d
    arts = {a["nonce"]: a for a in d["artifacts"]}
    assert set(arts) == set(nonces), (set(arts), set(nonces))
    return arts


@pytest.mark.integration
def test_async_multibatch_slot_reuse_no_crash():
    """3 sequential decode batches (distinct nonces) on one async server -> slots
    are reused; pre-fix the 2nd batch crashed the engine. Each batch must return a
    full k_points_steps (len max_tokens+1, all in range) for every nonce."""
    mt = 6
    batches = [list(range(0, 8)), list(range(8, 16)), list(range(16, 24))]
    with PoCTestServer(MODEL, list(BASE_ARGS)) as srv:
        for i, nonces in enumerate(batches):
            arts = _generate(srv, "0xasync_multibatch", nonces, mt)
            for n in nonces:
                steps = arts[n]["k_points_steps"]
                assert len(steps) == mt + 1, (f"batch {i} nonce {n}", steps)
                assert all(0 <= k < SPHERE_POINTS for k in steps), (
                    f"batch {i} nonce {n}", steps)

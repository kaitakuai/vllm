"""A validation PoC (carrying enforced_k_steps) must compute the real
aligned n_sphere_mismatches even when a generation PoC runs concurrently. Only the
pure path computes the aligned count; a validation request must therefore get an
exclusive pure batch. Without isolation the generation PoC forces the whole batch
through the step-driven path and validation returns the -1 sentinel.
"""
import asyncio

import httpx
import pytest

from tests.poc._server import (
    CANONICAL_MODEL as MODEL,
    DEFAULT_SERVER_ARGS as BASE_ARGS,
    PoCTestServer,
)

POC_URL = "/api/v1/pow/generate"
TIMEOUT = 120
VAL_NONCES = [0, 1, 2, 3]
GEN_NONCES = [10, 11, 12, 13]
MAX_TOKENS = 128
MAX_MISMATCH_FRAC = 0.30   # honest level; corruption would be far higher


@pytest.fixture(scope="module")
def server():
    with PoCTestServer(MODEL, BASE_ARGS) as srv:
        yield srv


def _body(nonces, block_hash, infk=None):
    body = {
        "block_hash": block_hash, "block_height": 100, "public_key": "cafebabe" * 8,
        "node_id": 0, "node_count": 1, "nonces": nonces,
        "params": {"model": MODEL, "seq_len": 256, "k_dim": 12, "max_tokens": MAX_TOKENS},
        "wait": True,
    }
    if infk is not None:
        body["enforced_k_steps"] = infk
    return body


@pytest.mark.integration
def test_validation_isolated_from_concurrent_generation(server):
    url = server.url_root

    async def post(body):
        async with httpx.AsyncClient(timeout=TIMEOUT) as c:
            r = await c.post(f"{url}{POC_URL}", json=body)
            r.raise_for_status()
            d = r.json()
            assert d.get("status") == "completed", d
            return {a["nonce"]: a for a in d.get("artifacts", [])}

    # Reference trajectory for the validation nonces.
    ref = asyncio.run(post(_body(VAL_NONCES, "deadbeef" * 8)))
    traj = {str(n): ref[n]["k_points_steps"] for n in VAL_NONCES}

    # Validate that trajectory WHILE a generation PoC runs concurrently.
    async def run():
        val = asyncio.create_task(post(_body(VAL_NONCES, "deadbeef" * 8, infk=traj)))
        gen = asyncio.create_task(post(_body(GEN_NONCES, "feedface" * 8)))
        v = await asyncio.wait_for(val, timeout=TIMEOUT)
        await asyncio.wait_for(gen, timeout=TIMEOUT)
        return v

    val = asyncio.run(run())

    mism = [val[n].get("n_sphere_mismatches") for n in VAL_NONCES]
    assert all(m is not None and m >= 0 for m in mism), (
        f"validation returned the -1 sentinel under a concurrent generation PoC "
        f"(it was dragged into the step-driven path): {mism}"
    )
    assert max(mism) / MAX_TOKENS < MAX_MISMATCH_FRAC, \
        f"validation honest mismatch too high: {mism}/{MAX_TOKENS}"

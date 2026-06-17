"""Native PoC transform must work in EAGER mode, not just under cudagraph.

The native design (vllm/poc/native.py) injects the Householder transform + PoC
embeds as inline layer code so they ride vLLM's compiled forward. The standing
requirement is that this is correct in BOTH modes. Every other integration test
runs the default cudagraph path; this one forces ``--enforce-eager`` so the eager
path (no capture/replay) is also guarded.

Contract (>=2 concurrent nonces): an HONEST self-recompute in eager yields a REAL
n_sphere_mismatches (never the -1 sentinel) and ~0 mismatches — i.e. the transform
is applied identically whether or not cudagraph is engaged.
"""
import httpx
import pytest

from tests.poc._server import (
    CANONICAL_MODEL as MODEL,
    DEFAULT_SERVER_ARGS as BASE_ARGS,
    PoCTestServer,
)
from tests.poc.utils import poc_request_body

POC_URL = "/api/v1/pow/generate"
TIMEOUT = 240
NONCES = [1, 2]          # >=2 concurrent nonces (standing multi-nonce rule)
MAX_TOKENS = 8
BLOCK_HASH = "0xeager"
HONEST_TOL = 2           # tiny boundary-flip margin; a real divergence is ~all steps


def _post(url: str, body: dict) -> dict[int, dict]:
    resp = httpx.post(f"{url}{POC_URL}", json=body, timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    assert data.get("status") == "completed", f"status != completed: {data}"
    arts = data.get("artifacts", [])
    assert len(arts) == len(NONCES), f"expected {len(NONCES)} artifacts, got {len(arts)}"
    return {a["nonce"]: a for a in arts}


@pytest.mark.integration
def test_native_transform_honest_separation_in_eager():
    with PoCTestServer(MODEL, BASE_ARGS + ["--enforce-eager"]) as srv:
        url = srv.url_root
        # inference -> reference trajectory
        inf = _post(url, poc_request_body(BLOCK_HASH, NONCES, MODEL, wait=True,
                                          max_tokens=MAX_TOKENS))
        inf_k = {n: inf[n]["k_points_steps"] for n in NONCES}
        assert all(inf_k[n] for n in NONCES), f"empty reference trajectory: {inf_k}"
        # validation -> honest self-recompute aligned to that trajectory
        val_body = poc_request_body(BLOCK_HASH, NONCES, MODEL, wait=True,
                                    max_tokens=MAX_TOKENS)
        val_body["enforced_k_steps"] = inf_k
        val = _post(url, val_body)

    for n in NONCES:
        nsm = val[n].get("n_sphere_mismatches")
        assert nsm is not None and nsm != -1, (
            f"nonce {n}: n_sphere_mismatches={nsm} — not computed in eager")
        assert 0 <= nsm <= HONEST_TOL, (
            f"nonce {n}: eager honest self-recompute expected ~0 "
            f"(<= {HONEST_TOL}), got {nsm} — native transform differs in eager")

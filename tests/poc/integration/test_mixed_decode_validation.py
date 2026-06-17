"""Regression gates: VALIDATION recompute must work while decode-PoC generation
runs step-driven mixed (the default).

Bug (root-causes/mixed-decode-skips-validation): a *validation* request — one
carrying ``enforced_k_steps`` (``PoCParams.is_validation``) — did not
consume ``enforced_k_steps`` on the mixed-decode path. The server returned
the ``-1`` sentinel for ``n_sphere_mismatches`` instead of the real aligned count,
so an honest validator could not separate honest from fraud once mixed decode
became the default.

Fix: validation recompute runs through the shared ``aligned_step`` — each step
seeds from the reference ``enforced_k_steps`` (no cascade) and counts the
real ``n_sphere_mismatches`` — regardless of mixing, while ordinary decode-PoC
*generation* still mixes with chat.

Contract asserted (>=2 concurrent nonces, the standing multi-nonce rule):
  1. the validation metric is actually COMPUTED — never the ``-1`` sentinel;
  2. an HONEST self-recompute (same server/model) yields ~0 mismatches;
  3. this holds while generation runs mixed (the regression case) — validation
     stays aligned to the reference and computes the real count.
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
NONCES = [1, 2]          # >=2 concurrent nonces (single-nonce "works" is a trap)
MAX_TOKENS = 8           # short decode trajectory: fast, still exercises chaining
BLOCK_HASH = "0xvalidate"
# Honest self-recompute should be 0; allow a tiny boundary-flip margin for GPU
# run-to-run noise. The bug returns -1; a real divergence returns ~most steps —
# both are far outside this margin.
HONEST_TOL = 2


def _post(url: str, body: dict) -> dict[int, dict]:
    resp = httpx.post(f"{url}{POC_URL}", json=body, timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    assert data.get("status") == "completed", f"status != completed: {data}"
    arts = data.get("artifacts", [])
    assert len(arts) == len(NONCES), f"expected {len(NONCES)} artifacts, got {len(arts)}"
    return {a["nonce"]: a for a in arts}


def _validate_roundtrip(url: str) -> dict[int, dict]:
    """Inference (reference trajectory) -> validation (aligned recompute)."""
    # warmup: absorb first-request-after-boot init (lazy engine/graph init)
    try:
        httpx.post(f"{url}{POC_URL}",
                   json=poc_request_body("0xwarmup", [1], MODEL, wait=True, max_tokens=1),
                   timeout=TIMEOUT)
    except Exception:
        pass
    # 1. inference -> reference k_points_steps per nonce
    inf = _post(url, poc_request_body(BLOCK_HASH, NONCES, MODEL, wait=True, max_tokens=MAX_TOKENS))
    inf_k = {n: inf[n]["k_points_steps"] for n in NONCES}
    assert all(inf_k[n] for n in NONCES), f"empty reference trajectory: {inf_k}"
    # 2. validation -> same request + the reference trajectory (is_validation)
    val_body = poc_request_body(BLOCK_HASH, NONCES, MODEL, wait=True, max_tokens=MAX_TOKENS)
    val_body["enforced_k_steps"] = inf_k
    return _post(url, val_body)


@pytest.mark.integration
def test_validation_computes_real_mismatches():
    """A validation request returns a REAL n_sphere_mismatches (never -1) and ~0
    for an honest self-recompute, for >=2 nonces. Generation runs mixed (default);
    validation stays aligned to the reference (the regression: mixed must not
    swallow enforced_k_steps)."""
    with PoCTestServer(MODEL, BASE_ARGS) as srv:
        val = _validate_roundtrip(srv.url_root)

    for n in NONCES:
        nsm = val[n].get("n_sphere_mismatches")
        # (1) computed at all — the regression: mixed path left it at the -1 sentinel
        assert nsm is not None and nsm != -1, (
            f"nonce {n}: n_sphere_mismatches={nsm} — validation NOT computed "
            f"(mixed path swallowed enforced_k_steps)")
        # (2) honest self-recompute is ~0
        assert 0 <= nsm <= HONEST_TOL, (
            f"nonce {n}: honest self-recompute expected ~0 "
            f"(<= {HONEST_TOL}) mismatches, got {nsm}")

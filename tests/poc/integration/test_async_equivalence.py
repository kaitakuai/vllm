"""Regression: decode-PoC works under async scheduling (no --no-async-scheduling)
and the consensus result is unchanged vs async-off.

Decode-PoC accumulates its trajectory on-device and emits the full artifact ONCE
at the final forward (emit-once). It stays alive via num_output_placeholders and
finishes artifact-driven (reusing chat's async drain) so the terminal artifact is
never stranded under async's batch-queue (which previously returned 0 artifacts).

What is / isn't byte-identical (see KB sphere-k-boundary-nondeterminism):
- A GENERATION trajectory is NOT bit-exact run-to-run or across batch shape: each
  sphere_k is a discrete snap of an fp-sensitive projection; a near-boundary flip
  cascades (each step seeds the next). This is inherent, not a code bug -> assert
  prefill k0 exactly + a per-step tolerance, NOT the whole trajectory.
- The deployed consensus path is VALIDATION, which is ALIGNED (the validator
  supplies the reference trajectory; each step is teacher-forced from the
  reference), so there is no cascade. The honest mismatch rate stays low under
  async exactly as under sync -> that is the real acceptance bar asserted here.

The continuous prefill vector_b64 is dropped from the decode path (batch-FP-noisy,
not scored for decode).
"""
import httpx
import pytest

from tests.poc._server import PoCTestServer
from tests.poc.utils import poc_request_body

MODEL = "Qwen/Qwen3-0.6B"
POC_URL = "/api/v1/pow/generate"
TIMEOUT = 240
# Small footprint so it co-fits the shared GPU.
BASE_ARGS = ["--gpu-memory-utilization", "0.3", "--max-model-len", "1024",
             "--enforce-eager"]
SPHERE_POINTS = 16


def _post(url, body):
    r = httpx.post(url, json=body, timeout=TIMEOUT)
    r.raise_for_status()
    d = r.json()
    assert d.get("status") == "completed", d
    return d


def _body(bh, nonces, mt, extra=None):
    b = poc_request_body(bh, nonces, MODEL, wait=True, seq_len=64, max_tokens=mt)
    if extra:
        b.update(extra)
    return b


def _generate(srv, bh, nonces, mt):
    d = _post(f"{srv.url_root}{POC_URL}", _body(bh, nonces, mt))
    arts = {a["nonce"]: a for a in d["artifacts"]}
    assert set(arts) == set(nonces), (set(arts), set(nonces))
    return arts


def _validate(srv, bh, nonces, mt, ref):
    d = _post(f"{srv.url_root}{POC_URL}",
              _body(bh, nonces, mt, {"enforced_k_steps": {n: ref[n] for n in nonces}}))
    return {a["nonce"]: a for a in d["artifacts"]}


@pytest.mark.integration
def test_decode_poc_async_on_returns_artifacts():
    """The core async fix: with async ON, a multi-nonce decode batch returns one
    full artifact per nonce (k_points_steps len == max_tokens+1), not 0. Pre-fix
    async returned 0 artifacts (terminal output stranded by the batch-queue)."""
    nonces, mt = [0, 1, 2, 3], 6
    with PoCTestServer(MODEL, list(BASE_ARGS)) as srv:
        arts = _generate(srv, "0xasync_artifacts", nonces, mt)
    for n in nonces:
        steps = arts[n]["k_points_steps"]
        assert len(steps) == mt + 1, (n, steps)
        assert all(0 <= k < SPHERE_POINTS for k in steps), (n, steps)


@pytest.mark.integration
def test_async_on_equals_async_off_prefill_k0():
    """Same generation async OFF then ON. The prefill k0 (k_points_steps[0]) is a
    discrete snap robust to batch-shape FP noise -> must match exactly async
    on/off. (The chained GENERATION trajectory is intentionally NOT asserted bit-
    equal: one near-boundary flip cascades every later step — inherent, see KB
    sphere-k-boundary-nondeterminism. The deployed consensus path is aligned
    validation, gated in test_aligned_validation_honest_low_mismatch_async_on.)"""
    nonces, mt = [0, 1, 2, 3], 8
    bh = "0xasync_equiv"
    with PoCTestServer(MODEL, ["--no-async-scheduling", *BASE_ARGS]) as srv:
        off = _generate(srv, bh, nonces, mt)
    with PoCTestServer(MODEL, list(BASE_ARGS)) as srv:
        on = _generate(srv, bh, nonces, mt)

    for n in nonces:
        ko, kn = off[n]["k_points_steps"], on[n]["k_points_steps"]
        assert len(ko) == len(kn) == mt + 1, (n, ko, kn)
        assert ko[0] == kn[0], (
            f"nonce {n}: prefill k0 differs async on/off (must be batch-robust): "
            f"{ko[0]} != {kn[0]}")


@pytest.mark.integration
def test_aligned_validation_honest_low_mismatch_async_on():
    """The deployed consensus path: validate an honest reference under async ON.
    Aligned (teacher-forced) validation has no cascade, so an honest prover's
    mismatch rate must stay low (boundary flips only) -> not flagged fraud."""
    nonces, mt = list(range(8)), 8
    bh = "0xasync_aligned"
    with PoCTestServer(MODEL, ["--no-async-scheduling", *BASE_ARGS]) as srv:
        ref_arts = _generate(srv, bh, nonces, mt)
    ref = {n: ref_arts[n]["k_points_steps"] for n in nonces}
    with PoCTestServer(MODEL, list(BASE_ARGS)) as srv:
        val = _validate(srv, bh, nonces, mt, ref)

    n_steps = sum(len(val[n]["k_points_steps"]) for n in nonces)
    n_mismatch = sum(max(val[n]["n_sphere_mismatches"], 0) for n in nonces)
    rate = n_mismatch / n_steps
    assert rate < 0.15, (
        f"honest aligned mismatch rate {rate:.2f} under async ON exceeds the "
        f"boundary-flip tolerance (honest prover would be falsely flagged)")

"""Regression: per-row block_hash isolation under the cached reflection-vector
scatter (B1).

The decode-PoC runner reflects each row with reflection vectors seeded by that row's
block_hash (vllm/poc/native.py). B1 caches that per-row scatter and skips it when the
row->block_hash map is unchanged across steps. A broken cache would serve a previous
block's reflection vectors to a later request -> wrong sphere_k -> consensus fault.

This MUST be tested teacher-forced (enforced_k_steps), NOT by comparing self-chained
prover trajectories: a prover trajectory is sensitive to batch-composition FP noise
(a single near-tie sphere_k flip cascades down the chain), so byte-equality across
batch sizes is not a real invariant. Teacher-forced validation re-seeds each step from
the reference, so a stray FP flip counts as 1 mismatch instead of cascading — making
honest≈0 and cross-block≈high, which cleanly separates correctness from FP noise.

The key check is the CACHE-CHURN sequence: validate A, then B (flips row_hashes),
then A again — the second A must still score ~0. A stale B1 cache would make it spike.
"""
import httpx
import pytest

from tests.poc._server import PoCTestServer
from tests.poc.utils import poc_request_body

MODEL = "Qwen/Qwen3-0.6B"
POC_URL = "/api/v1/pow/generate"
TIMEOUT = 240
BASE_ARGS = ["--gpu-memory-utilization", "0.3", "--max-model-len", "1024",
             "--enforce-eager"]
BH_A = "0xblockA"
BH_B = "0xblockB"
MT = 6
NONCES = [10, 11, 12, 13]


def _post(srv, body):
    r = httpx.post(f"{srv.url_root}{POC_URL}", json=body, timeout=TIMEOUT)
    r.raise_for_status()
    d = r.json()
    assert d.get("status") == "completed", d
    return {a["nonce"]: a for a in d["artifacts"]}


def _gen(srv, bh):
    arts = _post(srv, poc_request_body(bh, NONCES, MODEL, wait=True,
                                       seq_len=64, max_tokens=MT))
    return {n: arts[n]["k_points_steps"] for n in NONCES}


def _validate(srv, bh, ref):
    """Teacher-forced validate of block `bh` against reference trajectory `ref`;
    returns {nonce: n_sphere_mismatches}."""
    body = poc_request_body(bh, NONCES, MODEL, wait=True, seq_len=64, max_tokens=MT)
    body["enforced_k_steps"] = {str(n): ref[n] for n in NONCES}
    arts = _post(srv, body)
    return {n: arts[n]["n_sphere_mismatches"] for n in NONCES}


@pytest.mark.integration
def test_block_hash_isolation_under_cache_churn():
    total = MT + 1
    with PoCTestServer(MODEL, list(BASE_ARGS)) as srv:
        refA = _gen(srv, BH_A)
        refB = _gen(srv, BH_B)

        # Honest: validate each block against its OWN reference -> ~0 mismatch.
        hA = _validate(srv, BH_A, refA)
        hB = _validate(srv, BH_B, refB)          # churns the B1 cache (A-rows -> B-rows)
        hA2 = _validate(srv, BH_A, refA)          # KEY: cache must have re-invalidated
        # Cross: validate block A against block B's reference -> high mismatch.
        xAB = _validate(srv, BH_A, refB)

        for n in NONCES:
            assert hA[n] <= 1, f"honest A mismatch too high (n={n}): {hA[n]}/{total}"
            assert hB[n] <= 1, f"honest B mismatch too high (n={n}): {hB[n]}/{total}"
            # The regression guard: after churning with B, A still scores ~0.
            assert hA2[n] <= 1, (
                f"block A contaminated after B (stale B1 cache?) n={n}: "
                f"{hA2[n]}/{total}")
            # Sanity: cross-block is clearly distinguishable (not a no-op test).
            assert xAB[n] >= total // 2, (
                f"cross-block A-vs-refB mismatch too low (n={n}): {xAB[n]}/{total}")

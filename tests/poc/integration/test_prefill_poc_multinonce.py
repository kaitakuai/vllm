"""Edge case: prefill-only PoC (max_tokens=0) with multiple nonces must return
EVERY nonce's artifact, deterministically.

Regression guard for the scheduler bug where the "stop launching once done" guard
was gated on `max_tokens > 0`, so a prefill-only PoC (max_tokens == 0) was never
stopped after its prefill -> it got re-scheduled into a stray decode step that
(a) tripped the async input_ids int32/int64 scatter (EngineDead) and (b) stranded
artifacts, so an N-nonce request returned only 1..N artifacts, varying run-to-run.
Fix: scheduler.py applies the stop-guard regardless of max_tokens (prefill stops at
seq_len). See KB prefill-poc-multinonce-artifact-drop-test-all-modes.

The decode path (max_tokens > 0) was never affected; this is purely the max_tokens=0
edge. Asserts EXACT count (not "at least"), distinct nonces, and repeatability.
"""
import httpx
import pytest

from tests.poc._server import (
    CANONICAL_MODEL as MODEL,
    DEFAULT_SERVER_ARGS as SERVER_ARGS,
    open_poc_server,
)
from tests.poc.utils import check_artifact, poc_request_body

POC_URL = "/api/v1/pow/generate"
K_DIM = 12


@pytest.fixture(scope="module")
def server(request):
    with open_poc_server(request, SERVER_ARGS) as srv:
        yield srv


def _generate_prefill(url: str, nonces: list[int]) -> list[dict]:
    body = poc_request_body(
        "0xprefill_multinonce", nonces, MODEL, k_dim=K_DIM, max_tokens=0,
    )
    r = httpx.post(f"{url}{POC_URL}", json=body, timeout=120)
    r.raise_for_status()
    data = r.json()
    assert data.get("status") == "completed", f"status={data.get('status')}"
    return data.get("artifacts", [])


@pytest.mark.integration
@pytest.mark.parametrize("n", [1, 2, 5, 8])
def test_prefill_poc_returns_all_nonces(server, n):
    """N prefill nonces -> EXACTLY N valid, distinct artifacts (no drop, no crash)."""
    nonces = list(range(1, n + 1))
    artifacts = _generate_prefill(server.url_root, nonces)
    valid = [a for a in artifacts if check_artifact(a, K_DIM)]
    assert len(valid) == n, f"expected {n} valid artifacts, got {len(valid)} (drop/strand)"
    returned = sorted(a["nonce"] for a in valid)
    assert returned == nonces, f"nonce set mismatch: {returned} != {nonces}"


@pytest.mark.integration
def test_prefill_poc_count_is_deterministic(server):
    """Repeat the multi-nonce prefill: the count must be stable, never random."""
    nonces = list(range(1, 6))  # 5 nonces - the count that used to flap 1/2/4
    counts = []
    for _ in range(4):
        artifacts = _generate_prefill(server.url_root, nonces)
        counts.append(sum(1 for a in artifacts if check_artifact(a, K_DIM)))
    assert counts == [5, 5, 5, 5], f"prefill artifact count is not deterministic: {counts}"

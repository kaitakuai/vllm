import math
import collections
import pytest
import httpx

from tests.poc._server import CANONICAL_MODEL as MODEL, DEFAULT_SERVER_ARGS as SERVER_ARGS, open_poc_server
from tests.poc.utils import poc_request_body

POC_URL = "/api/v1/pow/generate"
TIMEOUT = 120

SPHERE_POINTS = 16


@pytest.fixture(scope="module")
def server(request):
    with open_poc_server(request, SERVER_ARGS) as srv:
        yield srv


@pytest.fixture(scope="module")
def url(server):
    return server.url_root


def _poc_with_decode(url: str, block_hash: str, nonces: list[int], max_tokens: int) -> dict:
    body = poc_request_body(
        block_hash, nonces, MODEL,
        wait=True,
        max_tokens=max_tokens,
    )
    resp = httpx.post(f"{url}{POC_URL}", json=body, timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    # Guard against vacuous passes: a completed request MUST return exactly one
    # artifact per nonce. Previously an empty `artifacts` list slipped through
    # the per-artifact loops below and "passed" silently.
    assert data.get("status") == "completed", f"status != completed: {data}"
    artifacts = data.get("artifacts", [])
    assert len(artifacts) == len(nonces), (
        f"expected {len(nonces)} artifacts, got {len(artifacts)} "
        f"(empty/partial artifacts == decode-PoC produced nothing): {data}"
    )
    return data


def _shannon_entropy(values: list[int], n_bins: int) -> float:
    """Shannon entropy (bits) of a discrete distribution over [0, n_bins)."""
    counts = collections.Counter(values)
    total = len(values)
    entropy = 0.0
    for c in counts.values():
        p = c / total
        entropy -= p * math.log2(p)
    return entropy


@pytest.mark.integration
class TestDecodeStepStructure:
    def test_kpoints_steps_length(self, url):
        """With max_tokens=N, k_points_steps has N+1 entries (prefill + N decode)."""
        max_tokens = 5
        data = _poc_with_decode(url, "0xdecode_length", [1], max_tokens)
        assert data["status"] == "completed"
        artifact = data["artifacts"][0]
        steps = artifact.get("k_points_steps", [])
        assert len(steps) == max_tokens + 1, (
            f"Expected {max_tokens + 1} steps (prefill + {max_tokens} decode), "
            f"got {len(steps)}"
        )

    def test_kpoints_steps_all_in_range(self, url):
        """All k_points_steps values are in [0, SPHERE_POINTS)."""
        data = _poc_with_decode(url, "0xdecode_range", list(range(4)), max_tokens=8)
        for artifact in data["artifacts"]:
            steps = artifact.get("k_points_steps", [])
            for step_idx, k in enumerate(steps):
                assert 0 <= k < SPHERE_POINTS, (
                    f"Artifact nonce={artifact['nonce']} step {step_idx}: "
                    f"k={k} out of [0, {SPHERE_POINTS})"
                )

    def test_prefill_k_present_with_decode(self, url):
        """The prefill k is k_points_steps[0], present and in range.

        The scalar ``sphere_k`` field was dropped (it was just a redundant copy of
        k_points_steps[0]); the per-step ``k_points_steps`` ([prefill_k, decode1_k,
        ...]) is the decode-PoC artifact. Also assert sphere_k is GONE.
        """
        data = _poc_with_decode(url, "0xdecode_sphere_k", [1, 2], max_tokens=3)
        for artifact in data["artifacts"]:
            steps = artifact.get("k_points_steps", [])
            assert steps, f"Artifact nonce={artifact['nonce']} missing k_points_steps"
            assert 0 <= steps[0] < SPHERE_POINTS
            assert "sphere_k" not in artifact, \
                "scalar sphere_k should be dropped from the response"


@pytest.mark.integration
class TestDecodeEntropy:
    def test_kpoints_steps_not_constant(self, url):
        """With 20 nonces × 10 decode steps, k values are not all the same."""
        n_nonces, max_tokens = 20, 10
        data = _poc_with_decode(url, "0xentropy_test", list(range(n_nonces)), max_tokens)
        all_k: list[int] = []
        for artifact in data["artifacts"]:
            all_k.extend(artifact.get("k_points_steps", []))
        assert len(set(all_k)) > 1, \
            "All k values are identical — decode is not producing varied sphere indices"

    def test_kpoints_steps_entropy_above_threshold(self, url):
        """Shannon entropy of k distribution exceeds 1.0 bit (not near-constant)."""
        n_nonces, max_tokens = 20, 10
        data = _poc_with_decode(url, "0xentropy_bits", list(range(n_nonces)), max_tokens)
        all_k: list[int] = []
        for artifact in data["artifacts"]:
            all_k.extend(artifact.get("k_points_steps", []))
        entropy = _shannon_entropy(all_k, SPHERE_POINTS)
        assert entropy > 1.0, (
            f"Entropy of k distribution is {entropy:.2f} bits — "
            f"expected > 1.0 (uniform would be {math.log2(SPHERE_POINTS):.2f})"
        )

    def test_distinct_nonces_distinct_kpoints(self, url):
        """Different nonces produce different k_points_steps sequences."""
        data = _poc_with_decode(url, "0xkpoints_nonce_diff", list(range(5)), max_tokens=5)
        sequences = [
            tuple(a.get("k_points_steps", []))
            for a in data["artifacts"]
        ]
        assert len(set(sequences)) > 1, \
            "All nonces produced identical k_points_steps — nonces are not seeding differently"


@pytest.mark.integration
class TestDecodeKVBound:
    def test_decode_depends_on_prefill(self, url):
        """A different prefill (block_hash) for the SAME nonce yields a
        substantially different decode trajectory — proving each decode step reads
        its prefill KV, not just the nonce seed. A couple of boundary flips would
        be FP noise; most-steps-differ is real KV dependence (the security-critical
        property: you cannot decode without having computed the right prefill)."""
        nonce, mt = [5], 8
        a = _poc_with_decode(url, "0xkvbound_A", nonce, mt)["artifacts"][0]["k_points_steps"]
        b = _poc_with_decode(url, "0xkvbound_B", nonce, mt)["artifacts"][0]["k_points_steps"]
        assert len(a) == len(b) == mt + 1
        differ = sum(1 for x, y in zip(a, b) if x != y)
        assert differ > 2, (
            f"decode trajectory barely changed across different prefill "
            f"({differ}/{len(a)} steps differ) — decode may not be KV-bound\n"
            f"A={a}\nB={b}")

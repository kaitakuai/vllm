import math
import pytest
import httpx

from tests.poc._server import CANONICAL_MODEL as MODEL, DEFAULT_SERVER_ARGS as SERVER_ARGS, open_poc_server
from tests.poc.utils import poc_request_body, decode_artifact_vector

POC_URL = "/api/v1/pow/generate"
DEFAULT_BLOCK_HASH = "0xtest_validity_hash"
DEFAULT_SEQ_LEN = 256
DEFAULT_K_DIM = 12
TIMEOUT = 120


@pytest.fixture(scope="module")
def server(request):
    with open_poc_server(request, SERVER_ARGS) as srv:
        yield srv


@pytest.fixture(scope="module")
def url(server):
    return server.url_root


def _generate(url: str, block_hash: str, nonces: list[int], **kwargs) -> dict:
    body = poc_request_body(
        block_hash, nonces, MODEL,
        seq_len=kwargs.pop("seq_len", DEFAULT_SEQ_LEN),
        k_dim=kwargs.pop("k_dim", DEFAULT_K_DIM),
        **kwargs,
    )
    resp = httpx.post(f"{url}{POC_URL}", json=body, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def _decode_all(data: dict) -> list[list[float]]:
    return [decode_artifact_vector(a["vector_b64"]) for a in data["artifacts"]]


@pytest.mark.integration
class TestVectorContent:
    def test_no_nan_or_inf(self, url):
        """All decoded vector components are finite."""
        data = _generate(url, DEFAULT_BLOCK_HASH, list(range(8)))
        assert data["status"] == "completed"
        for artifact in data["artifacts"]:
            vec = decode_artifact_vector(artifact["vector_b64"])
            assert all(math.isfinite(v) for v in vec), \
                f"Artifact nonce={artifact['nonce']} contains NaN or inf"

    def test_unit_norm(self, url):
        """Each artifact vector has L2 norm ≈ 1.0."""
        data = _generate(url, DEFAULT_BLOCK_HASH, list(range(8)))
        for artifact in data["artifacts"]:
            vec = decode_artifact_vector(artifact["vector_b64"])
            norm = math.sqrt(sum(v * v for v in vec))
            assert abs(norm - 1.0) < 0.02, \
                f"Artifact nonce={artifact['nonce']} has norm={norm:.4f}, expected ≈1.0"

    def test_artifact_count_matches_nonce_count(self, url):
        """Number of returned artifacts equals number of requested nonces."""
        nonces = [10, 20, 30, 40, 50]
        data = _generate(url, DEFAULT_BLOCK_HASH, nonces)
        assert len(data["artifacts"]) == len(nonces)

    def test_artifact_nonce_values_match_input(self, url):
        """Returned artifact nonces match the input nonces in order."""
        nonces = [100, 200, 300]
        data = _generate(url, DEFAULT_BLOCK_HASH, nonces)
        returned = [a["nonce"] for a in data["artifacts"]]
        assert returned == nonces, f"Nonce mismatch: input={nonces}, returned={returned}"


@pytest.mark.integration
class TestNonceIndependence:
    def test_all_nonce_vectors_distinct(self, url):
        """10 different nonces produce 10 distinct vectors."""
        nonces = list(range(10))
        data = _generate(url, "0xnonce_independence", nonces)
        vecs = [a["vector_b64"] for a in data["artifacts"]]
        assert len(set(vecs)) == len(vecs), \
            "Duplicate vectors detected: nonces must produce distinct artifacts"

    def test_different_nonces_different_from_each_other(self, url):
        """Vectors from nonces 1 and 2 are not close (L2 distance > 0.1)."""
        data = _generate(url, "0xnonce_diff", [1, 2])
        vecs = _decode_all(data)
        v1, v2 = vecs[0], vecs[1]
        l2 = math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))
        assert l2 > 0.1, f"Nonce 1 and 2 produced nearly identical vectors (L2={l2:.4f})"


@pytest.mark.integration
class TestBlockHashEffect:
    def test_different_hashes_different_vectors(self, url):
        """3 distinct block_hashes give distinct vectors for the same nonce."""
        hashes = ["0xhash_A", "0xhash_B", "0xhash_C"]
        vectors = []
        for bh in hashes:
            data = _generate(url, bh, [1])
            vectors.append(data["artifacts"][0]["vector_b64"])
        assert len(set(vectors)) == len(hashes), \
            "Different block_hashes must produce different vectors for the same nonce"

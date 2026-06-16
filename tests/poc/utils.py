import base64
import struct
from typing import Any


def poc_request_body(
    block_hash: str,
    nonces: list[int],
    model: str,
    *,
    public_key: str = "test_key",
    block_height: int = 100,
    seq_len: int = 256,
    k_dim: int = 12,
    wait: bool = True,
    blocking: bool = False,
    max_tokens: int = 0,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Build a canonical PoC ``/api/v1/pow/generate`` request body."""
    body: dict[str, Any] = {
        "block_hash": block_hash,
        "block_height": block_height,
        "public_key": public_key,
        "node_id": 0,
        "node_count": 1,
        "nonces": nonces,
        # proposal API: max_tokens lives under params (PoCParamsModel),
        # not top-level as on the legacy collective_rpc branch.
        "params": {
            "model": model,
            "seq_len": seq_len,
            "k_dim": k_dim,
            "max_tokens": max_tokens,
        },
        "wait": wait,
        "blocking": blocking,
    }
    if batch_size is not None:
        body["batch_size"] = batch_size
    return body


def decode_artifact_vector(b64: str) -> list[float]:
    """Decode a base64-encoded FP16 little-endian vector to a list of floats."""
    raw = base64.b64decode(b64)
    count = len(raw) // 2
    return list(struct.unpack(f"<{count}e", raw))


def check_artifact(artifact: dict[str, Any], k_dim: int = 12) -> bool:
    """Return True when *artifact* has a valid nonce and a correctly-sized vector."""
    if "nonce" not in artifact or "vector_b64" not in artifact:
        return False
    try:
        vec = decode_artifact_vector(artifact["vector_b64"])
        return len(vec) == k_dim
    except Exception:
        return False

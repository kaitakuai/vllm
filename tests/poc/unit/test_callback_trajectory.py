"""Unit test: the async-generation callback carries the decode trajectory.

The chain mines via the async /init/generate loop -> /generated callback. For
decode-PoC the validator needs the prover's reference trajectory, so the callback
artifact must include k_points_steps. Prefill artifacts (k_points_steps is None) must
keep the exact {nonce, vector_b64} shape (no behavior change). CPU, no GPU/server.
"""
import asyncio

from vllm.poc.callbacks import CallbackSender
from vllm.poc.data import Artifact


def _build_payload(artifacts):
    """Drive CallbackSender far enough to build _pending_payload, without sending."""
    sender = CallbackSender(callback_url="http://unused", stop_event=asyncio.Event(),
                            k_dim=12)
    sender.add_artifacts(artifacts, {"public_key": "pk", "block_hash": "bh",
                                     "block_height": 1, "node_id": 0})
    # Mirror the payload-build block in _run_loop (buffer -> _pending_payload).
    to_send = list(sender._buffer)
    return {
        "artifacts": [
            ({"nonce": a.nonce, "vector_b64": a.vector_b64}
             if a.k_points_steps is None else
             {"nonce": a.nonce, "vector_b64": a.vector_b64,
              "k_points_steps": a.k_points_steps})
            for a in to_send
        ],
    }


def test_prefill_artifact_shape_unchanged():
    payload = _build_payload([Artifact(nonce=5, vector_b64="abc")])
    art = payload["artifacts"][0]
    assert art == {"nonce": 5, "vector_b64": "abc"}
    assert "k_points_steps" not in art   # prefill: key absent, byte-unchanged


def test_decode_artifact_carries_trajectory():
    traj = [3, 7, 1, 9, 2]
    payload = _build_payload([Artifact(nonce=8, vector_b64="", k_points_steps=traj)])
    art = payload["artifacts"][0]
    assert art["nonce"] == 8
    assert art["vector_b64"] == ""          # decode: vector dropped
    assert art["k_points_steps"] == traj    # trajectory carried for the validator


def test_artifact_dataclass_default_is_prefill():
    """Default Artifact (no k_points_steps) is prefill -> backward compatible."""
    a = Artifact(nonce=1, vector_b64="x")
    assert a.k_points_steps is None

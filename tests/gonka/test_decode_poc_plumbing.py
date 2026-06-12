"""Unit tests for decode-PoC (#1135) API plumbing (mock-based, no GPU).

Covers:
1. engine_patch.poc_request threads max_tokens + inference_k_points_steps into
   collective_rpc, scales the timeout, and surfaces k_points_steps /
   n_sphere_mismatches onto artifacts.
2. routes._slice_inference_map restricts the reference map to a chunk.
3. Backward compatibility: max_tokens omitted -> prefill-only artifacts.
"""
import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock


def _rpc_result(nonces, *, decode=False):
    vectors = np.zeros((len(nonces), 12), dtype="float16")
    res = {"vectors": vectors, "nonces": list(nonces)}
    if decode:
        res["k_points_steps"] = [[1, 2, 3] for _ in nonces]       # prefill + 2 steps
        res["n_sphere_mismatches"] = [0 for _ in nonces]
    return res


class TestPocRequestDecodePlumbing:
    @pytest.mark.asyncio
    async def test_decode_params_reach_collective_rpc(self):
        from vllm.poc.engine_patch import poc_request

        mock_self = AsyncMock()
        mock_self.vllm_config.model_config.get_hidden_size.return_value = 4096
        # MagicMock (not AsyncMock) so has_unfinished_requests() is a plain
        # bool, keeping poc_request out of the in-flight-abort branch.
        mock_self.output_processor = MagicMock()
        mock_self.output_processor.has_unfinished_requests.return_value = False
        mock_self.collective_rpc = AsyncMock(
            return_value=[_rpc_result([0, 1], decode=True)]
        )

        inf_map = {0: [1, 2, 3], 1: [1, 2, 3]}
        result = await poc_request(
            mock_self, "generate_artifacts",
            {"nonces": [0, 1], "block_hash": "abc", "public_key": "pk",
             "seq_len": 256, "k_dim": 12,
             "max_tokens": 2, "inference_k_points_steps": inf_map},
            timeout_ms=10000,
        )

        # collective_rpc received max_tokens + inference_k_points_steps as the
        # last two positional args (order must match execute_poc_forward).
        _, kwargs = mock_self.collective_rpc.call_args
        args = kwargs["args"]
        assert args[-2] == 2          # max_tokens
        assert args[-1] == inf_map    # inference_k_points_steps
        # timeout scaled by (1 + max_tokens) = 3 -> 30s
        assert kwargs["timeout"] == pytest.approx(30.0)

        # decode fields surface onto every artifact
        arts = result["artifacts"]
        assert len(arts) == 2
        for a in arts:
            assert a["k_points_steps"] == [1, 2, 3]
            assert a["n_sphere_mismatches"] == 0

    @pytest.mark.asyncio
    async def test_prefill_only_artifacts_have_no_decode_fields(self):
        from vllm.poc.engine_patch import poc_request

        mock_self = AsyncMock()
        mock_self.vllm_config.model_config.get_hidden_size.return_value = 4096
        # MagicMock (not AsyncMock) so has_unfinished_requests() is a plain
        # bool, keeping poc_request out of the in-flight-abort branch.
        mock_self.output_processor = MagicMock()
        mock_self.output_processor.has_unfinished_requests.return_value = False
        mock_self.collective_rpc = AsyncMock(
            return_value=[_rpc_result([5, 6], decode=False)]
        )

        result = await poc_request(
            mock_self, "generate_artifacts",
            {"nonces": [5, 6], "block_hash": "abc", "public_key": "pk",
             "seq_len": 256, "k_dim": 12},
            timeout_ms=10000,
        )

        _, kwargs = mock_self.collective_rpc.call_args
        args = kwargs["args"]
        assert args[-2] == 0          # max_tokens default
        assert args[-1] is None       # inference_k_points_steps default
        assert kwargs["timeout"] == pytest.approx(10.0)  # not scaled

        for a in result["artifacts"]:
            assert set(a.keys()) == {"nonce", "vector_b64"}


def test_slice_inference_map():
    from vllm.poc.routes import _slice_inference_map

    full = {0: [1, 2], 1: [3, 4], 2: [5, 6]}
    assert _slice_inference_map(full, [1, 2]) == {1: [3, 4], 2: [5, 6]}
    assert _slice_inference_map(full, [0, 99]) == {0: [1, 2]}  # missing nonce skipped
    assert _slice_inference_map(None, [0, 1]) is None

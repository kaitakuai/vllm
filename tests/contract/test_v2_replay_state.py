# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The V2 sampler's replay bookkeeping, exercised without a GPU or engine.

ReplayState touches exactly two RequestState tensors (total_len,
prompt_len), so a two-field stand-in suffices; the test names state the
pinned semantics.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from vllm.v1.worker.gpu.sample.replay import ReplayState  # noqa: E402


class _Field:
    def __init__(self, values):
        self.gpu = torch.tensor(values, dtype=torch.int32)


class _ReqStates:
    def __init__(self, total_len, prompt_len):
        self.total_len = _Field(total_len)
        self.prompt_len = _Field(prompt_len)


class _Params:
    def __init__(self, etids=None, mode=None):
        self.enforced_token_ids = etids
        self.logprobs_mode = mode


def test_enforced_follows_emitted_position_and_exhausts_to_minus_one():
    # Slot 0: replay [11, 22]; slot 1: plain request.
    # total_len - prompt_len => emitted so far: slot0 has 1, slot1 has 5.
    rs = ReplayState(
        4,
        _ReqStates(total_len=[6, 15, 0, 0], prompt_len=[5, 10, 0, 0]),
        torch.device("cpu"),
    )
    rs.add_request(0, 5, _Params(etids=[11, 22]))
    rs.add_request(1, 10, _Params())

    rows = torch.tensor([0, 1], dtype=torch.int64)
    out = rs.enforced_for_batch(rows).tolist()
    assert out == [22, -1], "slot0 emitted 1 token, so position 1 => id 22"

    # Advance slot0 past the end of the replay: nothing left to enforce.
    rs.req_states.total_len.gpu[0] = 7
    assert rs.enforced_for_batch(rows).tolist() == [-1, -1]


def test_batch_predicate_is_slot_scoped():
    rs = ReplayState(4, _ReqStates([0] * 4, [0] * 4), torch.device("cpu"))
    rs.add_request(2, 1, _Params(etids=[7]))
    import numpy as np

    assert rs.batch_has_replay(np.array([2, 3]))
    assert not rs.batch_has_replay(np.array([0, 1, 3]))
    # Slot reuse must clear the flag.
    rs.add_request(2, 1, _Params())
    assert not rs.batch_has_replay(np.array([2]))


def test_processed_mask_mixes_override_with_engine_default():
    rs = ReplayState(4, _ReqStates([0] * 4, [0] * 4), torch.device("cpu"))
    rs.add_request(0, 1, _Params(mode="raw_logprobs"))
    rs.add_request(1, 1, _Params(mode="processed_logprobs"))
    rs.add_request(2, 1, _Params())  # no override -> engine default

    rows = torch.tensor([0, 1, 2], dtype=torch.int64)
    assert rs.processed_rows_mask(rows, engine_processed=True).tolist() == [
        False,
        True,
        True,
    ]
    assert rs.processed_rows_mask(rows, engine_processed=False).tolist() == [
        False,
        True,
        False,
    ]

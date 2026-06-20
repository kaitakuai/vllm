"""Unit test for mixed_decode.slice_sampling_metadata.

Restricting SamplingMetadata to chat rows is how decode-PoC stays OUT of the
sampler (PoC rows have no token ids / sampling params). The slice must select the
right rows from every per-row tensor and REMAP the index-keyed dicts, or chat gets
the wrong sampling params (the v0.20 co-existence corruption / penalty crash).
"""
import torch

from vllm.v1.sample.metadata import SamplingMetadata
from vllm.poc.mixed_decode import slice_sampling_metadata


def _make(n: int) -> SamplingMetadata:
    return SamplingMetadata(
        temperature=torch.arange(n, dtype=torch.float32),
        all_greedy=False,
        all_random=False,
        top_p=torch.arange(n, dtype=torch.float32) + 0.1,
        top_k=torch.arange(n, dtype=torch.int32),
        generators={1: "g1", 3: "g3"},
        max_num_logprobs=None,
        no_penalties=False,
        prompt_token_ids=torch.arange(n * 2, dtype=torch.long).reshape(n, 2),
        frequency_penalties=torch.arange(n, dtype=torch.float32) + 100,
        presence_penalties=torch.arange(n, dtype=torch.float32) + 200,
        repetition_penalties=torch.arange(n, dtype=torch.float32) + 300,
        output_token_ids=[[i, i] for i in range(n)],
        allowed_token_ids_mask=None,
        bad_words_token_ids={2: [[7]]},
        logitsprocs="SENTINEL",
        logprob_token_ids={0: [9]},
        spec_token_ids=[[i] for i in range(n)],
        enforced_next_token_ids=torch.arange(n, dtype=torch.long) + 500,
    )


def test_slice_selects_rows_and_remaps_indices():
    sm = _make(4)
    rows = [0, 2, 3]  # row 1 is a "PoC" row -> dropped
    out = slice_sampling_metadata(sm, rows, device="cpu")

    # Per-row tensors: keep exactly the selected rows, in order.
    assert out.temperature.tolist() == [0.0, 2.0, 3.0]
    assert out.frequency_penalties.tolist() == [100.0, 102.0, 103.0]
    assert out.presence_penalties.tolist() == [200.0, 202.0, 203.0]
    assert out.repetition_penalties.tolist() == [300.0, 302.0, 303.0]
    assert out.prompt_token_ids.tolist() == [[0, 1], [4, 5], [6, 7]]
    assert out.enforced_next_token_ids.tolist() == [500, 502, 503]

    # Per-row lists: subset in order.
    assert out.output_token_ids == [[0, 0], [2, 2], [3, 3]]
    assert out.spec_token_ids == [[0], [2], [3]]

    # Index-keyed dicts: remap OLD index -> NEW position; drop keys not kept.
    # gen at old 1 dropped (PoC row); old 3 -> new pos 2.
    assert out.generators == {2: "g3"}
    # bad_words at old 2 -> new pos 1.
    assert out.bad_words_token_ids == {1: [[7]]}
    # logprob at old 0 -> new pos 0.
    assert out.logprob_token_ids == {0: [9]}

    # Pass-through scalars/objects unchanged.
    assert out.logitsprocs == "SENTINEL"
    assert out.no_penalties is False


def test_slice_empty_rows_all_poc():
    sm = _make(3)
    out = slice_sampling_metadata(sm, [], device="cpu")
    assert out.temperature.numel() == 0
    assert out.output_token_ids == []
    assert out.generators == {}

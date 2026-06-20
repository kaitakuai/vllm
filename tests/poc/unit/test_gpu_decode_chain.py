"""Unit tests for the GPU-native decode chaining (vllm/poc/gpu_random.py).

These prove the on-device seed path is consensus-safe WITHOUT a GPU/model:
- deterministic (same inputs -> same bytes; prover == validator),
- chained (the seed/output depends on prev_k -> a real sequential trajectory),
- batch-independent (row i depends only on base[i], prev_k[i]),
- pick returns k distinct in-range dims.

Pure tensor math -> runs on CPU. prev_k is a TENSOR throughout (never .item()'d),
which is the whole point: no host sync, so async scheduling works.
"""
import torch

from vllm.poc.gpu_random import (
    decode_base_seeds,
    _step_seeds,
    generate_decode_inputs_gpu,
    random_pick_indices_gpu,
    _SALT_DECODE_EMBED,
    _SALT_DECODE_PICK,
)

DEV = torch.device("cpu")
BH, PK = "deadbeef" * 8, "cafebabe" * 8


def _base(nonces):
    return decode_base_seeds(BH, PK, nonces, DEV)


def test_base_seeds_deterministic():
    a = _base([0, 1, 2])
    b = _base([0, 1, 2])
    assert torch.equal(a, b)
    assert a.shape == (3,) and a.dtype == torch.int64
    # different nonces -> different base seeds (no collisions in a small set)
    assert len(set(a.tolist())) == 3


def test_step_seed_deterministic_and_chained():
    base = _base([7])
    pk5 = torch.tensor([5], dtype=torch.int64)
    pk6 = torch.tensor([6], dtype=torch.int64)
    # deterministic
    s1 = _step_seeds(base, 3, pk5, _SALT_DECODE_EMBED)
    s2 = _step_seeds(base, 3, pk5, _SALT_DECODE_EMBED)
    assert torch.equal(s1, s2)
    # chained: prev_k change -> different seed (avalanche)
    assert not torch.equal(s1, _step_seeds(base, 3, pk6, _SALT_DECODE_EMBED))
    # step change -> different seed
    assert not torch.equal(s1, _step_seeds(base, 4, pk5, _SALT_DECODE_EMBED))
    # salt separates streams (embed vs pick) for the same (step, prev_k)
    assert not torch.equal(
        _step_seeds(base, 3, pk5, _SALT_DECODE_EMBED),
        _step_seeds(base, 3, pk5, _SALT_DECODE_PICK),
    )


def test_decode_inputs_deterministic_chained_shape():
    base = _base([1, 2])
    pk = torch.tensor([4, 9], dtype=torch.int64)
    e1 = generate_decode_inputs_gpu(base, pk, 2, dim=16, device=DEV, dtype=torch.float32)
    e2 = generate_decode_inputs_gpu(base, pk, 2, dim=16, device=DEV, dtype=torch.float32)
    assert e1.shape == (2, 1, 16)
    assert torch.equal(e1, e2)                      # deterministic
    pk2 = torch.tensor([4, 10], dtype=torch.int64)  # change only row 1's prev_k
    e3 = generate_decode_inputs_gpu(base, pk2, 2, dim=16, device=DEV, dtype=torch.float32)
    assert torch.equal(e1[0], e3[0])                # row 0 unchanged (batch-independent)
    assert not torch.equal(e1[1], e3[1])            # row 1 changed (chained)


def test_pick_indices_distinct_in_range_and_chained():
    base = _base([1, 2, 3])
    pk = torch.tensor([0, 1, 2], dtype=torch.int64)
    idx = random_pick_indices_gpu(base, pk, step=5, dim=64, k=8, device=DEV)
    assert idx.shape == (3, 8)
    assert idx.min() >= 0 and idx.max() < 64
    for row in idx:                                  # distinct dims per row
        assert len(set(row.tolist())) == 8
    # deterministic + chained
    assert torch.equal(idx, random_pick_indices_gpu(base, pk, 5, 64, 8, DEV))
    pk2 = torch.tensor([0, 1, 99], dtype=torch.int64)
    idx2 = random_pick_indices_gpu(base, pk2, 5, 64, 8, DEV)
    assert torch.equal(idx[:2], idx2[:2])            # rows 0,1 unchanged
    assert not torch.equal(idx[2], idx2[2])          # row 2 changed with prev_k


def test_prev_k_never_needs_host():
    """The chain runs with prev_k as a tensor end-to-end (no .item()): emulate two
    steps feeding k as a tensor and confirm it all stays tensor-typed."""
    base = _base([42])
    prev_k = torch.tensor([0], dtype=torch.int64)
    for step in range(1, 4):
        emb = generate_decode_inputs_gpu(base, prev_k, step, dim=8, device=DEV)
        idx = random_pick_indices_gpu(base, prev_k, step, dim=8, k=3, device=DEV)
        # a fake "next k" derived on-device from the embedding (tensor, no .item())
        prev_k = (emb.abs().sum(dim=-1).squeeze(1).to(torch.int64) % 360)
        assert isinstance(prev_k, torch.Tensor) and prev_k.dtype == torch.int64
        assert emb.shape == (1, 1, 8) and idx.shape == (1, 3)

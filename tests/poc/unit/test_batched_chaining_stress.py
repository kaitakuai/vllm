"""Unit stress test: batched decode chaining == per-nonce loop, byte-for-byte.

The batching optimization replaced a per-nonce Python loop (B x B=1 calls) with one
[B, ...] call into the gpu_random decode helpers. Correctness rests on: a batched call
produces, for every row, exactly what the old per-row call produced. This fuzzes many
random (base_seeds, prev_k, step) batches — including PER-ROW step tensors (the new
code path) — and asserts batched[i] == per-row(i), exactly. CPU, deterministic.
"""
import random

import torch

from vllm.poc.gpu_random import (
    generate_decode_inputs_gpu,
    random_pick_indices_gpu,
    _step_seeds,
    _SALT_DECODE_EMBED,
)

DEV = torch.device("cpu")
DIM = 64
SPHERE_DIM = 16


def _rand_batch(rng, B):
    base = torch.tensor([rng.randrange(2**31) for _ in range(B)],
                        dtype=torch.int64, device=DEV)
    prev = torch.tensor([rng.randrange(64) for _ in range(B)],
                        dtype=torch.int64, device=DEV)
    return base, prev


def test_batched_embed_and_pick_equal_per_row():
    rng = random.Random(7)
    for _ in range(60):
        B = rng.randint(1, 16)
        base, prev = _rand_batch(rng, B)
        step = rng.randint(0, 255)

        emb_b = generate_decode_inputs_gpu(base, prev, step, DIM, DEV)        # [B,1,DIM]
        pick_b = random_pick_indices_gpu(base, prev, step, DIM, SPHERE_DIM, DEV)  # [B,k]
        for i in range(B):
            emb_i = generate_decode_inputs_gpu(
                base[i:i + 1], prev[i:i + 1], step, DIM, DEV)
            assert torch.equal(emb_b[i:i + 1], emb_i), f"embed row {i} (B={B})"
            pick_i = random_pick_indices_gpu(
                base[i:i + 1], prev[i:i + 1], step, DIM, SPHERE_DIM, DEV)
            assert torch.equal(pick_b[i:i + 1], pick_i), f"pick row {i} (B={B})"


def test_per_row_step_tensor_equals_scalar():
    """The real decode case: each nonce at a DIFFERENT step (per-row step tensor) must
    equal calling the helper per row with that row's scalar step."""
    rng = random.Random(99)
    for _ in range(60):
        B = rng.randint(1, 16)
        base, prev = _rand_batch(rng, B)
        steps = torch.tensor([rng.randint(0, 255) for _ in range(B)],
                             dtype=torch.int64, device=DEV)

        emb_b = generate_decode_inputs_gpu(base, prev, steps, DIM, DEV)
        pick_b = random_pick_indices_gpu(base, prev, steps, DIM, SPHERE_DIM, DEV)
        for i in range(B):
            s = int(steps[i])
            emb_i = generate_decode_inputs_gpu(base[i:i + 1], prev[i:i + 1], s, DIM, DEV)
            assert torch.equal(emb_b[i:i + 1], emb_i), f"per-row step embed {i}"
            pick_i = random_pick_indices_gpu(
                base[i:i + 1], prev[i:i + 1], s, DIM, SPHERE_DIM, DEV)
            assert torch.equal(pick_b[i:i + 1], pick_i), f"per-row step pick {i}"

        # And a scalar step must match a constant per-row step tensor.
        c = int(steps[0])
        steps_c = torch.full((B,), c, dtype=torch.int64, device=DEV)
        assert torch.equal(
            generate_decode_inputs_gpu(base, prev, c, DIM, DEV),
            generate_decode_inputs_gpu(base, prev, steps_c, DIM, DEV),
        ), "scalar step != constant step-tensor"


def test_step_seeds_chain_sensitivity():
    """Seeds must change with prev_k AND step (the chain): adjacent steps / prev_k
    values give uncorrelated seeds (avalanche), and batched == per-row."""
    rng = random.Random(5)
    base = torch.tensor([rng.randrange(2**31) for _ in range(8)],
                        dtype=torch.int64, device=DEV)
    prev = torch.tensor([rng.randrange(64) for _ in range(8)],
                        dtype=torch.int64, device=DEV)
    s0 = _step_seeds(base, 0, prev, _SALT_DECODE_EMBED)
    s1 = _step_seeds(base, 1, prev, _SALT_DECODE_EMBED)
    s0b = _step_seeds(base, 0, prev + 1, _SALT_DECODE_EMBED)
    assert not torch.equal(s0, s1), "seed did not change with step"
    assert not torch.equal(s0, s0b), "seed did not change with prev_k"
    # batched == per-row
    for i in range(base.shape[0]):
        si = _step_seeds(base[i:i + 1], 0, prev[i:i + 1], _SALT_DECODE_EMBED)
        assert torch.equal(s0[i:i + 1], si), f"step_seeds row {i}"

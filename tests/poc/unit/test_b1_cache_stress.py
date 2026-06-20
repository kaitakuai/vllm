"""Unit stress test for the B1 reflection-vector cache (PoCNativeState).

B1 skips the per-row reflection-vector rescatter when the row->block_hash map is
unchanged. This fuzzes long sequences of row_hash configs (cache hits, misses, churn,
None rows, varying length) and after EVERY call asserts each row's per-layer buffer
holds exactly that row's block_hash vectors (ground truth computed independently).

Non-circular: compares buffers to independently-generated Householder vectors, not to
self-produced trajectories — so it deterministically catches any stale-cache bug
(hit-when-it-should-miss, miss-when-it-should-hit, wrong length handling) on CPU.
"""
import random

import torch

from vllm.poc.gpu_random import generate_householder_vector
from vllm.poc.native import PoCNativeState

DEV = torch.device("cpu")
NL, HS, MT = 4, 16, 8
BLOCKS = [None, "0xAAA", "0xBBB", "0xCCC"]


def _truth(bh, layer):
    return generate_householder_vector(
        f"{bh}_layer_{layer}_householder", HS, DEV).to(torch.float32)


def _assert_buffers(st, rh):
    for row, bh in enumerate(rh):
        for layer in range(NL):
            got = st.vectors[layer][row]
            if bh is None:
                assert torch.equal(got, torch.zeros(HS, device=DEV)), \
                    f"row {row} layer {layer}: None row not zero ({rh})"
            else:
                assert torch.allclose(got, _truth(bh, layer), atol=1e-6), \
                    f"row {row} layer {layer}: wrong vectors for {bh} ({rh})"


def test_b1_cache_stress_random_churn():
    st = PoCNativeState(NL, HS, MT, DEV, torch.float32)
    rng = random.Random(1234)
    last = None
    hits = misses = 0
    for _ in range(400):
        # 40% repeat last config (exercise cache-HIT/skip path), else new random
        if last is not None and rng.random() < 0.4:
            rh = list(last)
            hits += 1
        else:
            n = rng.randint(1, MT)
            rh = [rng.choice(BLOCKS) for _ in range(n)]
            if rh != last:
                misses += 1
        st.set_row_block_hashes(rh)
        # After a hit (skip) OR a miss (rescatter), buffers[:len(rh)] must be correct.
        _assert_buffers(st, rh)
        last = rh
    assert hits > 20 and misses > 20, f"weak coverage: hits={hits} misses={misses}"


def test_b1_explicit_churn_sequence():
    """The exact A -> B -> A transition the optimization risks: after B churns the
    cache, re-selecting A must restore A's vectors (not leave stale B)."""
    st = PoCNativeState(NL, HS, MT, DEV, torch.float32)
    for rh in ([["0xAAA", "0xAAA"]] * 3       # set A, then 2 cache-hits
               + [["0xBBB", "0xBBB"]]          # churn to B (miss)
               + [["0xAAA", "0xAAA"]]          # back to A (must re-invalidate)
               + [["0xAAA", "0xBBB"]]          # mixed in one batch
               + [[None, "0xAAA"]]):           # masked row + A
        st.set_row_block_hashes(rh)
        _assert_buffers(st, rh)

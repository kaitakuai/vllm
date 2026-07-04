"""Seeded-routing: deterministic, hidden-independent MoE expert selection for PoC.

Removes the MoE top-k routing nondeterminism (the decode-PoC honest-floor amplifier)
by forcing expert selection from a seed instead of the noise-prone hidden. These cover
the seeded selector (bit-exact) and the masked wrapper (PoC rows only; chat untouched).
Pure CPU; no GPU/server.
"""
import torch
import torch.nn as nn

from vllm.poc.gpu_random import (_seed_from_string, expert_logits_from_base,
                                 route_base_seed, seeded_experts, seeded_expert_logits)
from vllm.poc.native import PoCRouterWrapper

DEV = torch.device("cpu")


def test_seeded_logits_deterministic():
    a = seeded_expert_logits("bh_route_layer_0", 256, 8, DEV)
    b = seeded_expert_logits("bh_route_layer_0", 256, 8, DEV)
    assert torch.equal(a, b)                       # same seed -> identical (cross-HW safe: integer murmur3)


def test_exactly_top_k_distinct_chosen():
    a = seeded_expert_logits("x", 256, 8, DEV)
    assert int((a > 0).sum()) == 8                 # exactly top_k experts elevated
    assert torch.topk(a, 8).indices.unique().numel() == 8   # distinct
    assert float(a.min()) == -1.0e4                # floor for the rest


def test_seed_varies_selection_per_layer():
    s0 = set(torch.topk(seeded_expert_logits("bh_route_layer_0", 256, 8, DEV), 8).indices.tolist())
    s1 = set(torch.topk(seeded_expert_logits("bh_route_layer_1", 256, 8, DEV), 8).indices.tolist())
    assert s0 != s1                                # different layer -> different experts


def test_router_wrapper_applies_to_poc_rows_only():
    """The mask is what makes it PoC-only: True rows get the forced (seeded) logits,
    False (chat) rows pass through the real router UNCHANGED."""
    n_exp = 16
    natural = torch.full((3, n_exp), 0.0)
    natural[:, 7] = 9.0                            # chat's real router would pick expert 7

    class FakeGate(nn.Module):
        def forward(self, x):
            return natural.clone(), None           # (logits, bias)

    force = torch.full((3, n_exp), -1.0e4)         # PER-ROW seeded logits buffer [B, n_experts]
    force[:, 3] = 5.0                               # seeded expert = 3 for every PoC row
    mask = torch.tensor([True, False, True])       # rows 0,2 = PoC ; row 1 = chat
    w = PoCRouterWrapper(FakeGate(), force, mask)

    logits, bias = w(torch.randn(3, 8))
    assert bias is None
    assert int(torch.argmax(logits[0])) == 3       # PoC row -> seeded expert 3
    assert int(torch.argmax(logits[2])) == 3       # PoC row -> seeded expert 3
    assert torch.equal(logits[1], natural[1])      # CHAT row untouched (still expert 7)


# --- seed = block_hash + nonce + step + layer (cached base + on-GPU step fold) ---

def _chosen(bh, nonce, step, layer, n=64, k=8):
    return tuple(seeded_experts(bh, nonce, step, layer, n, k, DEV).sort().values.tolist())


def test_route_base_seed_excludes_step():
    # base is the CACHED part (no step) -> hashed once, reused across all steps
    assert route_base_seed("bh", 3, 2) == "bh_n3_route_layer_2"
    bases = {route_base_seed("bh", n, l) for n in range(4) for l in range(4)}
    assert len(bases) == 16                           # distinct per (nonce,layer)


def test_batched_equals_single_row_reference():
    # the live path uses BATCHED expert_logits_from_base; it must equal the per-row
    # seeded_experts reference (so batching never changes the result).
    bases = torch.tensor([_seed_from_string(route_base_seed("bh", n, 0)) for n in range(5)],
                         dtype=torch.int64, device=DEV)
    steps = torch.tensor([3, 3, 3, 3, 3], dtype=torch.int64, device=DEV)
    forced = expert_logits_from_base(bases, steps, 64, 8, DEV)         # [5, 64]
    for r, n in enumerate(range(5)):
        ref = seeded_experts("bh", n, 3, 0, 64, 8, DEV).sort().values
        got = torch.topk(forced[r], 8).indices.sort().values
        assert torch.equal(ref, got)


def test_sensitivity_each_axis_changes_routing():
    b = _chosen("bh", 1, 1, 1)
    assert _chosen("BH", 1, 1, 1) != b               # block_hash
    assert _chosen("bh", 2, 1, 1) != b               # nonce
    assert _chosen("bh", 1, 2, 1) != b               # step
    assert _chosen("bh", 1, 1, 2) != b               # layer


def test_per_step_routing_varies_across_trajectory():
    # within ONE (block,nonce,layer) experts must change step-to-step, so a prover
    # cannot reuse one routing pattern for the whole 256-step trajectory.
    sets = {_chosen("bh", 0, s, 0) for s in range(32)}
    assert len(sets) >= 30                            # ~all distinct


def test_per_nonce_routing_varies():
    sets = {_chosen("bh", n, 0, 0) for n in range(32)}
    assert len(sets) >= 30


def test_uniform_coverage_no_dead_or_dominating_experts():
    n, k = 64, 8
    counts = torch.zeros(n, dtype=torch.long); trials = 0
    for nn_ in range(40):
        for s in range(40):
            for e in _chosen("deadbeef", nn_, s, 0, n, k):
                counts[e] += 1
            trials += 1
    expected = trials * k / n
    assert counts.min() > 0                           # no dead expert
    assert counts.max() < 2.0 * expected              # none dominates


def test_generic_across_contract_expert_counts():
    for n, k in [(64, 8), (128, 8), (256, 8), (384, 8)]:   # a range of expert counts
        chosen = _chosen("bh", 0, 0, 0, n, k)
        assert len(set(chosen)) == k and max(chosen) < n and min(chosen) >= 0


def test_chosen_always_in_range_overflow_safe():
    # Expert ids come from topk over n_experts columns -> structurally in [0, n_experts);
    # there is no hash%n. The murmur SCORE arithmetic overflows int64 internally, but the
    # &0xFFFFFFFF mask keeps the low-32 bits modular-correct (output in [0, 2^32), no
    # inf/nan). Sweep many seeds incl. large n_experts to lock the range guarantee.
    for n, k in [(8, 2), (64, 8), (256, 8), (384, 8), (1024, 8)]:
        for nonce in range(0, 60, 7):
            for step in range(0, 600, 60):
                c = _chosen("deadbeef", nonce, step, 3, n, k)
                assert len(set(c)) == k                    # exactly top_k distinct
                assert min(c) >= 0 and max(c) < n          # never out of range


def test_murmur_low32_correct_despite_int64_overflow():
    from vllm.poc.gpu_random import _batched_murmur3_32
    kmax = (1 << 32) - 1
    assert kmax * 0xCC9E2D51 > (1 << 63) - 1               # intermediate DOES exceed int64
    out = int(_batched_murmur3_32(torch.tensor([[kmax]], dtype=torch.int64),
                                  torch.tensor([[123456789]], dtype=torch.int64))[0, 0])
    assert 0 <= out < (1 << 32)                            # but result is clean 32-bit

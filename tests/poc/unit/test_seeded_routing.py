"""Seeded-routing: deterministic, hidden-independent MoE expert selection for PoC.

Removes the MoE top-k routing nondeterminism (the decode-PoC honest-floor amplifier)
by forcing expert selection from a seed instead of the noise-prone hidden. These cover
the seeded selector (bit-exact) and the masked wrapper (PoC rows only; chat untouched).
Pure CPU; no GPU/server.
"""
import torch
import torch.nn as nn

from vllm.poc.gpu_random import seeded_expert_logits
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

    force = torch.full((n_exp,), -1.0e4)
    force[3] = 5.0                                  # seeded expert = 3
    mask = torch.tensor([True, False, True])       # rows 0,2 = PoC ; row 1 = chat
    w = PoCRouterWrapper(FakeGate(), force, mask)

    logits, bias = w(torch.randn(3, 8))
    assert bias is None
    assert int(torch.argmax(logits[0])) == 3       # PoC row -> seeded expert 3
    assert int(torch.argmax(logits[2])) == 3       # PoC row -> seeded expert 3
    assert torch.equal(logits[1], natural[1])      # CHAT row untouched (still expert 7)

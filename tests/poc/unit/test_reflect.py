"""Unit tests for _reflect — the masked Householder reflection applied to every PoC
row in every decoder layer (vllm/poc/native.py). It shapes every PoC hidden state, so
it is consensus-critical, yet was only covered indirectly. Pure CPU math here:
involution, norm preservation, per-row mask correctness, batch independence.
"""
import torch

from vllm.poc.native import _reflect

DEV = torch.device("cpu")


def _unit(v):
    return v / (v.norm(dim=-1, keepdim=True) + 1e-12)


def test_reflect_is_involution():
    """A Householder reflection is its own inverse: reflect(reflect(x)) == x."""
    torch.manual_seed(0)
    x = torch.randn(8, 32, dtype=torch.float64, device=DEV)
    v = _unit(torch.randn(8, 32, dtype=torch.float64, device=DEV))
    mask = torch.ones(8, 1, dtype=torch.bool, device=DEV)
    once = _reflect(x, v, mask)
    twice = _reflect(once, v, mask)
    assert torch.allclose(twice, x, atol=1e-10), "reflection is not an involution"
    assert not torch.allclose(once, x, atol=1e-6), "reflection was a no-op (v ⟂ x?)"


def test_reflect_preserves_norm():
    """Reflection is orthogonal -> ||reflect(x)|| == ||x|| for unit v."""
    torch.manual_seed(1)
    x = torch.randn(16, 64, dtype=torch.float64, device=DEV)
    v = _unit(torch.randn(16, 64, dtype=torch.float64, device=DEV))
    mask = torch.ones(16, 1, dtype=torch.bool, device=DEV)
    out = _reflect(x, v, mask)
    assert torch.allclose(out.norm(dim=-1), x.norm(dim=-1), atol=1e-10)


def test_reflect_mask_selects_rows():
    """mask=False rows pass through UNCHANGED (chat rows); mask=True rows transform."""
    torch.manual_seed(2)
    x = torch.randn(4, 16, dtype=torch.float64, device=DEV)
    v = _unit(torch.randn(4, 16, dtype=torch.float64, device=DEV))
    mask = torch.tensor([[True], [False], [True], [False]], device=DEV)
    out = _reflect(x, v, mask)
    # chat rows (False) untouched
    assert torch.equal(out[1], x[1]) and torch.equal(out[3], x[3])
    # poc rows (True) changed
    assert not torch.allclose(out[0], x[0]) and not torch.allclose(out[2], x[2])


def test_reflect_row_independent():
    """Each row reflects with its OWN v; a row's result must not depend on other rows
    (the per-row-block_hash isolation guarantee at the math level)."""
    torch.manual_seed(3)
    x = torch.randn(4, 16, dtype=torch.float64, device=DEV)
    v = _unit(torch.randn(4, 16, dtype=torch.float64, device=DEV))
    mask = torch.ones(4, 1, dtype=torch.bool, device=DEV)
    full = _reflect(x, v, mask)
    # reflect row 2 alone -> must equal its slice from the batched result
    solo = _reflect(x[2:3], v[2:3], mask[2:3])
    assert torch.allclose(solo, full[2:3], atol=1e-12)


def test_reflect_orthogonal_vector_is_identity():
    """If x ⟂ v, reflection leaves x unchanged (dot == 0)."""
    x = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64, device=DEV)
    v = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64, device=DEV)
    mask = torch.ones(1, 1, dtype=torch.bool, device=DEV)
    assert torch.allclose(_reflect(x, v, mask), x, atol=1e-12)

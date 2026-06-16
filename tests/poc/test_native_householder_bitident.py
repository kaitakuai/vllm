"""Bit-identity gate: the graphable native Householder wrapper must reproduce the
eager ``register_forward_hook`` reflection EXACTLY, so cross-validator bit-compat is
preserved when we move hook -> native wrapper.

Requires CUDA (the reflection vectors are generated on device). Run on one GPU:

    .venv/bin/python -m pytest tests/poc/test_native_householder_bitident.py -v
"""
import pytest
import torch

from vllm.poc.gpu_random import apply_householder, generate_householder_vector
from vllm.poc.native_householder import (PoCLayerWrapper, _reflect,
                                         attach_native_poc, detach_native_poc)

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@cuda
def test_reflect_op_byte_identical_to_apply_householder():
    dev = torch.device("cuda")
    torch.manual_seed(0)
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        x = torch.randn(8, 13, 4096, device=dev, dtype=dtype)
        v = generate_householder_vector("blkABC_layer_5_householder", 4096, dev)  # fp32
        # hook math, exact .to(x.dtype) ordering (layer_hooks.py:97):
        ref = apply_householder(x, v.to(x.dtype))
        active = torch.ones((), dtype=torch.bool, device=dev)
        got = _reflect(x, v.to(x.dtype), active)
        assert torch.equal(ref, got), f"reflect mismatch dtype={dtype}"
        off = torch.zeros((), dtype=torch.bool, device=dev)
        assert torch.equal(_reflect(x, v.to(x.dtype), off), x), "active=False must be identity"


@cuda
def test_wrapper_tuple_handling_matches_hook():
    """Reflect BOTH hidden and residual with the SAME v (layer_hooks.py:104-106),
    preserve the rest of the tuple untouched."""
    dev = torch.device("cuda")
    h = torch.randn(8, 4096, device=dev, dtype=torch.float16)
    r = torch.randn(8, 4096, device=dev, dtype=torch.float16)
    v = generate_householder_vector("blk_layer_0_householder", 4096, dev)
    active = torch.ones((), dtype=torch.bool, device=dev)

    class Inner(torch.nn.Module):
        def forward(self):
            return (h, r, "extra")

    w = PoCLayerWrapper(Inner(), v.to(torch.float32), active)
    out = w()
    assert torch.equal(out[0], apply_householder(h, v.to(h.dtype)))
    assert torch.equal(out[1], apply_householder(r, v.to(r.dtype)))
    assert out[2] == "extra"


@cuda
def test_wrapper_single_and_bare_tensor():
    dev = torch.device("cuda")
    h = torch.randn(8, 4096, device=dev, dtype=torch.bfloat16)
    v = generate_householder_vector("blk_layer_1_householder", 4096, dev)
    active = torch.ones((), dtype=torch.bool, device=dev)

    class InnerTuple1(torch.nn.Module):
        def forward(self):
            return (h,)

    class InnerBare(torch.nn.Module):
        def forward(self):
            return h

    w1 = PoCLayerWrapper(InnerTuple1(), v.to(torch.float32), active)
    assert torch.equal(w1()[0], apply_householder(h, v.to(h.dtype)))
    wb = PoCLayerWrapper(InnerBare(), v.to(torch.float32), active)
    assert torch.equal(wb(), apply_householder(h, v.to(h.dtype)))


@cuda
def test_attach_detach_roundtrip_and_active_gate():
    """attach wraps layers in place; active=False => identity; detach restores."""
    dev = torch.device("cuda")
    hidden = 256

    class Layer(torch.nn.Module):
        def forward(self, x):
            return (x, x.clone())

    class Inner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Layer() for _ in range(4)])

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()

    m = Model().to(dev)
    state = attach_native_poc(m, hidden, dev)
    assert all(isinstance(l, PoCLayerWrapper) for l in m.model.layers)
    # idempotent
    assert attach_native_poc(m, hidden, dev) is state
    # default inactive -> identity
    x = torch.randn(8, hidden, device=dev, dtype=torch.float16)
    state.set_block_hash("blkZ")
    assert torch.equal(m.model.layers[0](x)[0], x)
    # active -> reflect with layer-0 vector
    state.set_active(True)
    v0 = generate_householder_vector("blkZ_layer_0_householder", hidden, dev)
    assert torch.equal(m.model.layers[0](x)[0], apply_householder(x, v0.to(x.dtype)))
    state.set_active(False)
    detach_native_poc(m)
    assert all(isinstance(l, Layer) for l in m.model.layers)

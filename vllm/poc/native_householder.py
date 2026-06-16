"""Graphable native Householder reflection for decode-PoC.

Replaces the un-capturable ``register_forward_hook`` path in ``layer_hooks.py``
with an ``nn.Module`` wrapper attached to each decoder layer BEFORE
torch.compile / cudagraph capture, so the reflection is traced into the compiled
graph and recorded by ``CUDAGraphWrapper`` instead of running as a Python hook
that dynamo drops and cudagraph cannot replay.

It is kept **bit-identical to our eager hook**
(``layer_hooks.LayerHouseholderHook`` + ``gpu_random.apply_householder``):

  * reflection math: ``x - 2 * (x·v) * v``  (note ``2 *``, exactly matching
    ``apply_householder``; ``v`` down-cast to ``x.dtype`` BEFORE the dot product,
    matching ``layer_hooks.py`` ``v.to(x.dtype)`` ordering).
  * ONE unit vector ``v`` per layer (our single-block, PoC-only-batch semantics),
    broadcast over all rows -- NOT Ilya's per-row ``[max_tokens, hidden]``.
  * ``v`` from the SAME ``generate_householder_vector(seed_str, dim, device)`` with
    ``seed_str = f"{block_hash}_layer_{i}_householder"`` (== layer_hooks.py:77).
  * tuple-output handling matches ``_create_hook``: ``(hidden, residual, *rest)`` ->
    reflect ``hidden`` AND ``residual``, keep ``rest``; ``(hidden,)`` -> reflect it;
    bare tensor -> reflect it.

The only behavioural change vs the hook is graph-enabling:

  * the transform lives INSIDE ``PoCLayerWrapper.forward`` (traced/compiled), not in
    a ``register_forward_hook``.
  * the Python ``if is_poc_forward_active()`` global gate becomes a persistent 0-d
    bool buffer read via ``torch.where`` -> static shape, no host control flow inside
    the graph. The buffer is flipped in place (False = identity for chat, True =
    reflect for PoC); the captured PoC graph is replayed with it True.

Our isolated PoC forward is PoC-only (never mixes chat rows), so a single ``v`` per
layer and a scalar gate are sufficient and exactly reproduce the validated hook.
"""
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .gpu_random import generate_householder_vector


def _reflect(x: torch.Tensor, v: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    """Householder reflection identical to ``apply_householder``, gated by a 0-d bool.

    ``active`` is a persistent 0-d bool buffer: True -> ``x - 2*(x·v)*v``; False -> ``x``.
    ``torch.where`` (not a Python ``if``) keeps the op static-shape and capturable.
    When ``active`` is True the result is bitwise equal to ``apply_householder(x, v)``
    (``torch.where`` selects ``transformed`` element-wise without recomputation).
    ``v`` is broadcast over all leading (token) dims.
    """
    dot = (x * v).sum(dim=-1, keepdim=True)        # == apply_householder
    transformed = x - 2 * dot * v                  # `2 *`, byte-matches the hook
    return torch.where(active, transformed, x)


class PoCLayerWrapper(nn.Module):
    """Wraps one decoder layer; reflects its output hidden (and residual) like the hook."""

    def __init__(self, inner: nn.Module, v: torch.Tensor, active: torch.Tensor):
        super().__init__()
        self.inner = inner
        # persistent=False -> stays out of state_dict (checkpoint untouched).
        # v stored fp32 (generate_householder_vector returns fp32 unit vec); the
        # down-cast to x.dtype happens INSIDE forward, matching layer_hooks.py:97.
        self.register_buffer("poc_v", v, persistent=False)
        self.register_buffer("poc_active", active, persistent=False)

    def forward(self, *args, **kwargs):
        out = self.inner(*args, **kwargs)
        if isinstance(out, tuple):
            if len(out) >= 2:
                hidden = out[0]
                residual = out[1]
                rest = out[2:]
                hidden = _reflect(hidden, self.poc_v.to(hidden.dtype), self.poc_active)
                residual = _reflect(residual, self.poc_v.to(residual.dtype),
                                    self.poc_active)
                return (hidden, residual) + rest
            hidden = out[0]
            return (_reflect(hidden, self.poc_v.to(hidden.dtype), self.poc_active),)
        return _reflect(out, self.poc_v.to(out.dtype), self.poc_active)


class PoCNativeState:
    """Owns the persistent per-layer reflection vectors + the shared 0-d active flag.

    Vectors are (re)generated only when ``block_hash`` changes, written in place
    (``copy_``) into the SAME buffers, so addresses baked into a captured graph stay
    valid -- the graph-safe analogue of the hook regenerating ``reflection_vectors``.
    """

    def __init__(self, num_layers: int, hidden_size: int,
                 device: torch.device):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device
        # fp32 unit vectors (down-cast per-forward inside _reflect, matching the hook).
        self.vectors: List[torch.Tensor] = [
            torch.zeros(hidden_size, device=device, dtype=torch.float32)
            for _ in range(num_layers)
        ]
        # 0-d bool buffer; flipped in place to gate reflection without re-capture.
        self.active: torch.Tensor = torch.zeros((), dtype=torch.bool, device=device)
        self.block_hash: Optional[str] = None

    @torch.inference_mode()
    def set_block_hash(self, block_hash: str) -> None:
        if self.block_hash == block_hash:
            return
        for i in range(self.num_layers):
            seed_str = f"{block_hash}_layer_{i}_householder"   # == layer_hooks.py:77
            v = generate_householder_vector(seed_str, self.hidden_size, self.device)
            self.vectors[i].copy_(v.to(torch.float32))         # in place, addr stable
        self.block_hash = block_hash

    @torch.inference_mode()
    def set_active(self, on: bool) -> None:
        self.active.fill_(bool(on))                            # in place


def _layers_container(model: nn.Module) -> Tuple[nn.Module, str]:
    """Return (parent, attr) of the decoder-layer ModuleList. Mirrors layer_hooks."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model, "layers"
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer, "h"
    return model, "layers"


def _find_layers(model: nn.Module) -> List[nn.Module]:
    """Identical selection order to layer_hooks._find_layers."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    if hasattr(model, "layers"):
        return list(model.layers)
    return []


def attach_native_poc(model: nn.Module, hidden_size: int,
                      device: torch.device) -> PoCNativeState:
    """Wrap each decoder layer with ``PoCLayerWrapper`` IN PLACE, before compile/capture.

    Idempotent: if already attached, returns the cached state. Mutates the existing
    ``layers`` ModuleList in place (``layers[i] = wrapper``) -- the same object the
    model iterates in forward, so the swap is guaranteed visible (the proven approach
    from Ilya's native.py). Wrappers default to ``active=False`` (identity), so normal
    serving is unaffected until ``state.set_active(True)`` is called for a PoC forward.
    """
    existing = getattr(model, "_poc_native_state", None)
    if existing is not None:
        return existing
    parent, attr = _layers_container(model)
    layers = getattr(parent, attr)
    state = PoCNativeState(len(layers), hidden_size, device)
    for i in range(len(layers)):
        layers[i] = PoCLayerWrapper(layers[i], state.vectors[i], state.active)
    model._poc_native_state = state
    return state


def detach_native_poc(model: nn.Module) -> None:
    """Unwrap in place (for tests / fallback)."""
    state = getattr(model, "_poc_native_state", None)
    if state is None:
        return
    parent, attr = _layers_container(model)
    layers = getattr(parent, attr)
    for i in range(len(layers)):
        w = layers[i]
        if isinstance(w, PoCLayerWrapper):
            layers[i] = w.inner
    delattr(model, "_poc_native_state")

"""Native PoC transform.

The per-layer Householder reflection applied as INLINE layer code, so vLLM's
native torch.compile + cudagraph capture it like any model op (prefill AND decode,
dynamic KV, any backend) — replacing the un-capturable Python forward-hook and the
hand-rolled CUDA-graph capture.

Each decoder layer is wrapped by ``PoCLayerWrapper``, which runs the original layer
then reflects the output (hidden AND residual) on PoC rows only, selected by a
shared boolean mask buffer. Chat rows pass through unchanged (mask False →
``where`` is identity), so one compiled model serves chat and PoC. The reflection
vectors (one per layer, seeded by block_hash) and the mask live in stable buffers
updated in-place each round/step, so replay reads the live values.
"""
import os

import torch
from torch import nn

from .gpu_random import generate_householder_vector

# Debug-only TP guard (VLLM_POC_DEBUG_TP=1): PoC reflection vectors / embeds are
# generated per rank from deterministic seeds and MUST be bit-identical across
# tensor-parallel ranks, else rows reflect/inject differently per rank -> corruption.
_DEBUG_TP = os.environ.get("VLLM_POC_DEBUG_TP") == "1"


def _assert_replicated_across_tp(t: torch.Tensor, name: str) -> None:
    """No-op unless VLLM_POC_DEBUG_TP=1 and TP world size > 1. Fingerprints `t`
    (3 moments) and all-gathers across the TP group, asserting bit-equality so a
    per-rank RNG divergence is caught the moment a TP run hits it."""
    if not _DEBUG_TP:
        return
    try:
        import torch.distributed as dist
        from vllm.distributed import (
            get_tensor_model_parallel_group,
            get_tensor_model_parallel_world_size,
        )
    except ImportError:
        return
    if not dist.is_initialized():
        return
    ws = get_tensor_model_parallel_world_size()
    if ws <= 1:
        return
    x = t.detach().to(torch.float64).reshape(-1)
    pos = torch.arange(1, x.numel() + 1, device=x.device, dtype=torch.float64)
    fp = torch.stack([x.sum(), (x * x).sum(), (x * pos).sum()])
    gathered = [torch.empty_like(fp) for _ in range(ws)]
    dist.all_gather(gathered, fp, group=get_tensor_model_parallel_group().device_group)
    for r in range(1, ws):
        if not torch.equal(gathered[0], gathered[r]):
            raise AssertionError(
                f"PoC '{name}' diverged across TP ranks (rank0 vs rank{r}) — "
                "per-rank RNG non-determinism; PoC is not TP-safe in this setup")


def _reflect(x: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked Householder: rows where mask is True -> x - 2*(x·v)*v; else x.
    Per-row independent, static-shape (no data-dependent control flow) -> the
    compiled graph captures it; cudagraph replays it reading live v/mask."""
    dot = (x * v).sum(-1, keepdim=True)
    transformed = x - 2.0 * dot * v
    return torch.where(mask, transformed, x)


class PoCLayerWrapper(nn.Module):
    """Wraps one decoder layer; reflects its output hidden + residual on PoC rows.
    ``v`` is this layer's reflection vector; ``mask`` is the shared per-row PoC mask
    (both stable buffers, updated in place)."""

    def __init__(self, inner: nn.Module, v: torch.Tensor, mask: torch.Tensor):
        super().__init__()
        self.inner = inner
        self.register_buffer("poc_v", v, persistent=False)
        self.register_buffer("poc_mask", mask, persistent=False)

    def forward(self, *args, **kwargs):
        out = self.inner(*args, **kwargs)
        if isinstance(out, tuple):
            hidden = out[0]
            n = hidden.shape[0]
            m = self.poc_mask[:n].unsqueeze(-1)
            v = self.poc_v[:n]  # per-row reflection vectors [n, hidden]
            hidden = _reflect(hidden, v.to(hidden.dtype), m)
            rest = list(out[1:])
            if rest and rest[0] is not None:  # residual
                rest[0] = _reflect(rest[0], v.to(rest[0].dtype), m)
            return (hidden, *rest)
        n = out.shape[0]
        m = self.poc_mask[:n].unsqueeze(-1)
        return _reflect(out, self.poc_v[:n].to(out.dtype), m)


class PoCEmbeddingWrapper(nn.Module):
    """Wraps the token embedding; for PoC rows, replaces the token embeds with the
    deterministic PoC embeds (from a stable buffer). PoC requests carry dummy token
    IDs so the graphed input_ids path runs; this injects the real PoC embeds INSIDE
    the graph. Chat rows keep their token embeds (mask False)."""

    def __init__(self, inner: nn.Module, embeds: torch.Tensor, mask: torch.Tensor):
        super().__init__()
        self.inner = inner
        self.register_buffer("poc_embeds", embeds, persistent=False)
        self.register_buffer("poc_mask", mask, persistent=False)

    def forward(self, input_ids):
        # PoC rows carry a dummy token id whose embedding is overridden below, but
        # under async scheduling that id can be a stale/sentinel value (e.g. -1 from
        # the previous step's sampled-token plumbing) -> out-of-vocab gather crash.
        # Force masked (PoC) rows to a valid in-vocab id (0); their value is unused.
        n = input_ids.shape[0]
        m_rows = self.poc_mask[:n]
        # On-device clamp (no host sync): masked rows -> 0, chat rows unchanged.
        input_ids = torch.where(m_rows, torch.zeros_like(input_ids), input_ids)
        out = self.inner(input_ids)
        m = m_rows.unsqueeze(-1)
        return torch.where(m, self.poc_embeds[:n].to(out.dtype), out)


class PoCNativeState:
    """Per-model PoC transform state: PER-ROW reflection vectors per wrapped layer,
    a shared row mask, and a PoC-embeds buffer. Held on the runner; updated each
    round (block_hash) / step (mask, embeds) in place so the captured graph reads
    live values.

    The reflection vectors are per-row ([max_tokens, hidden]) so requests with
    DIFFERENT block_hashes can share one forward batch without stepping on each
    other (each row reflects with its own block's vectors).
    """

    def __init__(self, num_layers: int, hidden_size: int, max_tokens: int,
                 device, dtype):
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vectors = [
            torch.zeros(max_tokens, hidden_size, device=device, dtype=dtype)
            for _ in range(num_layers)
        ]
        self.mask = torch.zeros(max_tokens, dtype=torch.bool, device=device)
        self.embeds = torch.zeros(max_tokens, hidden_size, device=device, dtype=dtype)
        self._hash_cache: dict[str, list] = {}      # block_hash -> per-layer vectors
        self._last_row_hashes: list | None = None   # skip redundant per-step rescatter

    def set_embeds(self, row_embeds: torch.Tensor) -> None:
        """Write the PoC rows' input embeds into the buffer (in place)."""
        n = row_embeds.shape[0]
        self.embeds[:n].copy_(row_embeds)
        _assert_replicated_across_tp(self.embeds[:n], "embeds")

    def _vectors_for(self, block_hash: str) -> list:
        """Per-layer reflection vectors for a block_hash (cached across forwards)."""
        vs = self._hash_cache.get(block_hash)
        if vs is None:
            vs = [
                generate_householder_vector(
                    f"{block_hash}_layer_{i}_householder",
                    self.hidden_size, self.device)
                for i in range(self.num_layers)
            ]
            self._hash_cache[block_hash] = vs
        return vs

    def set_row_block_hashes(self, row_hashes: list) -> None:
        """Write each row's reflection vectors from ITS OWN block_hash (in place),
        so requests with different block_hashes coexist in one forward. row_hashes[i]
        = block_hash for row i, or None (left zero; masked out). Generation of the
        vectors is cached per block_hash; the scatter is cheap CPU-side setup.

        The reflection vectors depend only on block_hash (not the decode step), so for
        a stable batch the row->hash mapping is unchanged across all decode steps. Skip
        the zero + per-(row,layer) copy_ (num_layers x B kernels) when row_hashes
        matches the last call; the buffers already hold the right values."""
        if row_hashes == self._last_row_hashes:
            return
        for buf in self.vectors:
            buf.zero_()
        for row, bh in enumerate(row_hashes):
            if bh is None:
                continue
            vs = self._vectors_for(bh)
            for i, buf in enumerate(self.vectors):
                buf[row].copy_(vs[i].to(buf.dtype))
        self._last_row_hashes = list(row_hashes)
        _assert_replicated_across_tp(self.vectors[0], "reflection_vectors[0]")

    def set_mask(self, row_mask: torch.Tensor | None) -> None:
        """Set which rows are PoC this forward (in place). None -> all chat."""
        self.mask.zero_()
        if row_mask is not None:
            n = row_mask.shape[0]
            self.mask[:n].copy_(row_mask)


def attach_native_poc(model: nn.Module, layers: list, embed_owner, max_tokens: int,
                      hidden_size: int, device, dtype) -> PoCNativeState:
    """Wrap each decoder layer (Householder) AND the token embedding (PoC-embed
    injection) BEFORE compilation, sharing one mask. Returns the state to drive
    them. Idempotent: skipped if already wrapped."""
    if any(isinstance(layer, PoCLayerWrapper) for layer in layers):
        return getattr(model, "_poc_native_state")
    state = PoCNativeState(len(layers), hidden_size, max_tokens, device, dtype)
    for i, layer in enumerate(layers):
        layers[i] = PoCLayerWrapper(layer, state.vectors[i], state.mask)
    if embed_owner is not None and hasattr(embed_owner, "embed_tokens"):
        embed_owner.embed_tokens = PoCEmbeddingWrapper(
            embed_owner.embed_tokens, state.embeds, state.mask)
    model._poc_native_state = state
    return state

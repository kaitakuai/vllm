"""PoC model runner for vLLM 0.15.x V1 architecture.

Full model forward pass with proper V1 attention metadata.
Uses actual KV cache blocks for attention to work correctly.
Batched forward pass — processes all nonces in a single forward call.
"""
import hashlib
import math
import os
from contextlib import contextmanager
import torch
import torch.distributed as dist
import numpy as np
from typing import List, Optional, Dict, Any

from vllm.distributed import get_pp_group, get_tp_group
from vllm.distributed.communication_op import broadcast_tensor_dict
from vllm.forward_context import set_forward_context, BatchDescriptor
from vllm.config.compilation import CUDAGraphMode
from vllm.sequence import IntermediateTensors
from vllm.logger import init_logger

from .gpu_random import (
    generate_inputs,
    generate_inputs_concat_murmur,
    generate_decode_inputs,
    random_pick_indices,
    apply_haar_rotation,
)
from .layer_hooks import LayerHouseholderHook, poc_forward_context

logger = init_logger(__name__)

DEFAULT_K_DIM = 12

# ---------------------------------------------------------------------------
# decode-PoC (#1135): sphere codebook quantization of decode-step hidden states
# ---------------------------------------------------------------------------
# SPHERE_DIM:    number of hidden-state dimensions sliced and projected onto
#                the unit sphere before nearest-codebook lookup.
# SPHERE_POINTS: number of equidistant reference points on that sphere; each
#                decode step commits log2(SPHERE_POINTS) bits (k-id in [0, N)).
SPHERE_DIM = 256
SPHERE_POINTS = 16


def project_to_sphere(v: torch.Tensor) -> torch.Tensor:
    """Normalize [..., dim] vectors to the unit sphere (L2 norm = 1)."""
    return v / (v.norm(dim=-1, keepdim=True) + 1e-8)


def _halton_on_sphere(n_points: int, dim: int) -> torch.Tensor:
    """Return n_points deterministic, low-discrepancy unit vectors on S^(dim-1).

    Uses the Halton sequence (base-prime per dimension) mapped to the sphere
    via the logit transform.  No randomness — identical output for any call
    with the same (n_points, dim).
    """
    _PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
    coords: List[List[float]] = []
    for d in range(dim):
        base = _PRIMES[d % len(_PRIMES)]
        col: List[float] = []
        for i in range(1, n_points + 1):
            f, r = 1.0, 0.0
            j = i
            while j > 0:
                f /= base
                r += f * (j % base)
                j //= base
            col.append(r)
        coords.append(col)

    # [n_points, dim] in (0, 1)^dim  ->  logit  ->  R^dim  ->  sphere
    raw = torch.tensor(coords, dtype=torch.float32).T.clamp(0.01, 0.99)
    pts = torch.log(raw / (1.0 - raw))  # logit: roughly normal spread
    return project_to_sphere(pts)


def build_equidistant_codebook(
    n_points: int,
    dim: int,
    n_steps: int = 500,
    lr: float = 0.05,
) -> torch.Tensor:
    """Build a codebook of approximately equidistant points on S^(dim-1).

    Solves the Thomson problem: minimize the electrostatic repulsion energy
    (sum of 1/distance^2 for all pairs) so points spread as uniformly as
    possible over the sphere.  Initialisation is deterministic (Halton
    sequence) — no randomness.
    """
    with torch.inference_mode(mode=False):
        pts = _halton_on_sphere(n_points, dim).clone().requires_grad_(True)
        opt = torch.optim.Adam([pts], lr=lr)

        eye = torch.eye(n_points)
        for _ in range(n_steps):
            opt.zero_grad()
            p = project_to_sphere(pts)
            diff = p.unsqueeze(0) - p.unsqueeze(1)
            d2 = (diff * diff).sum(-1)
            energy = ((1.0 - eye) / (d2 + 1e-8)).sum()
            energy.backward()
            opt.step()

        result = project_to_sphere(pts).detach()
    return result


# Built lazily on first decode-PoC use (NOT at import time) so the production
# prefill-only PoC v2 path keeps its import cost and behaviour unchanged.
# Override with an exact, frozen codebook via GONKA_POC_SPHERE_CODEBOOK (path
# to a torch.save'd float32 [SPHERE_POINTS, SPHERE_DIM] tensor) to guarantee
# bit-identical k-ids across torch versions / validators.
_SPHERE_CODEBOOK: Optional[torch.Tensor] = None


def get_sphere_codebook() -> torch.Tensor:
    """Return the (cached) sphere codebook, building or loading it on first use."""
    global _SPHERE_CODEBOOK
    if _SPHERE_CODEBOOK is not None:
        return _SPHERE_CODEBOOK

    path = os.environ.get("GONKA_POC_SPHERE_CODEBOOK")
    if path:
        cb = torch.load(path, map_location="cpu").float()
        if tuple(cb.shape) != (SPHERE_POINTS, SPHERE_DIM):
            raise ValueError(
                f"GONKA_POC_SPHERE_CODEBOOK shape {tuple(cb.shape)} != "
                f"expected ({SPHERE_POINTS}, {SPHERE_DIM})"
            )
        cb = project_to_sphere(cb)
        logger.info("PoC decode: loaded sphere codebook from %s", path)
    else:
        cb = build_equidistant_codebook(SPHERE_POINTS, SPHERE_DIM)

    digest = hashlib.sha256(cb.cpu().numpy().tobytes()).hexdigest()[:16]
    logger.info(
        "PoC decode: sphere codebook ready (points=%d, dim=%d, sha256=%s)",
        SPHERE_POINTS, SPHERE_DIM, digest,
    )
    _SPHERE_CODEBOOK = cb
    return cb


def nearest_sphere_index(
    query: torch.Tensor,
    codebook: torch.Tensor,
) -> torch.Tensor:
    """Return the index of the nearest codebook point for each query vector.

    Args:
        query: unit vectors (one per nonce) [batch, dim]
        codebook: unit vectors from the sphere codebook [SPHERE_POINTS, dim]

    Returns:
        index k in [0, SPHERE_POINTS) per query [batch]
    """
    sims = query.float() @ codebook.float().T   # [batch, SPHERE_POINTS]
    return sims.argmax(dim=-1)                   # [batch]

# NOTE: attention metadata must NOT be cached across PoC calls.
# The metadata builder's internal state (workspace buffers, page-table
# references) is mutated by every inference engine step.  Reusing a
# stale metadata object causes the attention backend to write only a
# fraction of the expected KV entries, producing all-NaN hidden states.
# The cost of rebuilding is <1 ms per call (vs ~15 ms for the model
# forward), so the overhead is negligible.


def _ensure_layer_hooks(worker, block_hash, hidden_size):
    """Ensure per-block Householder reflection is installed for ``block_hash``.

    Default: the eager ``register_forward_hook`` path (LayerHouseholderHook).
    With ``GONKA_POC_NATIVE_HOUSEHOLDER=1``: the graphable native wrapper
    (attached once before compile/capture; reflection vectors refilled in place
    on block_hash change). The native path is bit-identical to the hook on the
    PoC-only batch but is capturable by torch.compile + CUDA-graph.
    """
    model = worker.model_runner.model
    device = worker.device

    if os.environ.get("GONKA_POC_NATIVE_HOUSEHOLDER") == "1":
        from .native_householder import attach_native_poc
        # Reach the real nn.Module under any CUDAGraphWrapper so the wrappers sit
        # inside the capturable region.
        inner = model.unwrap() if hasattr(model, "unwrap") else model
        state = attach_native_poc(inner, hidden_size, device)
        state.set_block_hash(block_hash)   # in-place refill, addresses stable
        worker._poc_native_state = state
        return

    existing_hook = getattr(worker, "_poc_layer_hooks", None)
    if existing_hook is not None:
        if existing_hook.block_hash == block_hash:
            return
        existing_hook.detach()
    hook = LayerHouseholderHook(model, block_hash, device, hidden_size)
    hook._setup(model, block_hash, device, hidden_size)
    worker._poc_layer_hooks = hook


@contextmanager
def _poc_reflection(worker):
    """Activate per-layer reflection around a PoC model() call.

    Native path (GONKA_POC_NATIVE_HOUSEHOLDER=1): flip the persistent ``active``
    buffer True for the duration, then back to False so any later non-PoC forward
    that reaches the wrappers is identity. Hook path: the legacy global flag.
    """
    state = getattr(worker, "_poc_native_state", None)
    if state is not None:
        # GONKA_POC_FORCE_INACTIVE: debug toggle to keep the wrapper inactive
        # (no reflection) under compile, to isolate whether reflection is applied.
        active = os.environ.get("GONKA_POC_FORCE_INACTIVE") != "1"
        state.set_active(active)
        try:
            yield
        finally:
            state.set_active(False)
    else:
        with poc_forward_context():
            yield


def _get_block_size(worker):
    """Get the KV cache block size from the worker config."""
    return worker.cache_config.block_size


def _create_v1_attn_metadata(batch_size, seq_len, block_size, device, worker):
    """Create attention metadata for batch_size sequences.

    Uses the worker's metadata builders to create the correct metadata
    for whatever attention backend is configured (FlashAttention,
    FlashInfer, etc.).
    """
    from vllm.v1.attention.backend import CommonAttentionMetadata

    blocks_per_seq = math.ceil(seq_len / block_size)
    total_tokens = batch_size * seq_len

    # slot_mapping: each sequence gets its own block range
    all_slots = []
    for seq_idx in range(batch_size):
        base_block = seq_idx * blocks_per_seq
        for t in range(seq_len):
            block_idx = base_block + t // block_size
            all_slots.append(block_idx * block_size + t % block_size)
    slot_mapping = torch.tensor(all_slots, dtype=torch.long, device=device)

    # block_table: [batch_size, blocks_per_seq]
    block_table = torch.arange(
        batch_size * blocks_per_seq, dtype=torch.int32, device=device
    ).view(batch_size, blocks_per_seq)

    # query_start_loc: [0, seq_len, 2*seq_len, ..., batch_size*seq_len]
    query_start_loc_gpu = (
        torch.arange(batch_size + 1, dtype=torch.int32, device=device) * seq_len
    )
    query_start_loc_cpu = (
        torch.arange(batch_size + 1, dtype=torch.int32, device="cpu") * seq_len
    )

    seq_lens_gpu = torch.full(
        (batch_size,), seq_len, dtype=torch.int32, device=device
    )
    seq_lens_cpu = torch.full(
        (batch_size,), seq_len, dtype=torch.int32, device="cpu"
    )

    common_attn_metadata = CommonAttentionMetadata(
        query_start_loc=query_start_loc_gpu,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens_gpu,
        num_reqs=batch_size,
        num_actual_tokens=total_tokens,
        max_query_len=seq_len,
        max_seq_len=seq_len,
        block_table_tensor=block_table,
        slot_mapping=slot_mapping,
        causal=True,
        _seq_lens_cpu=seq_lens_cpu,
        seq_lens_cpu_upper_bound=seq_lens_cpu,
        _num_computed_tokens_cpu=torch.zeros(
            batch_size, dtype=torch.int32, device="cpu"
        ),
    )

    model_runner = worker.model_runner
    attn_metadata_dict = {}
    slot_mapping_dict = {}

    for kv_cache_group_attn_groups in model_runner.attn_groups:
        for attn_group in kv_cache_group_attn_groups:
            builder = attn_group.get_metadata_builder(0)
            metadata = builder.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
            )
            for layer_name in attn_group.layer_names:
                attn_metadata_dict[layer_name] = metadata
                slot_mapping_dict[layer_name] = slot_mapping

    return attn_metadata_dict, slot_mapping_dict


def _create_decode_attn_metadata_with_history(
    batch_size,
    prefill_seq_len,
    step,
    block_size,
    device,
    worker,
    prefill_blocks_per_seq,
    max_decode_blocks_per_seq,
    decode_block_start,
    prefill_seq_idx=None,
):
    """Create attention metadata for a single decode step with full context history.

    One new token per sequence (the query) attends to all prefill_seq_len + step
    tokens in the KV cache — real autoregressive decode semantics (decode-PoC
    reference line, bs/poc-context-fix).  Physical block layout must be
    consistent with what _create_v1_attn_metadata wrote during the prefill:
      - seq i prefill blocks: p_i*prefill_blocks_per_seq .. (p_i+1)*prefill_blocks_per_seq - 1,
        where p_i = prefill_seq_idx[i] is the sequence's ORIGINAL batch
        position (NaN filters may have shrunk the batch after the prefill)
      - seq i decode blocks:  decode_block_start + i*max_decode_blocks_per_seq + j
    """
    from vllm.v1.attention.backend import CommonAttentionMetadata

    if prefill_seq_idx is None:
        prefill_seq_idx = list(range(batch_size))

    new_pos = prefill_seq_len + step - 1
    block_in_seq = new_pos // block_size
    slot_in_block = new_pos % block_size
    context_len = prefill_seq_len + step
    total_blocks_for_context = math.ceil(context_len / block_size)

    slot_mapping_list = []
    block_table_rows = []

    for seq_idx in range(batch_size):
        phys_prefill_base = prefill_seq_idx[seq_idx] * prefill_blocks_per_seq
        if block_in_seq < prefill_blocks_per_seq:
            phys_block = phys_prefill_base + block_in_seq
        else:
            decode_blk_idx = block_in_seq - prefill_blocks_per_seq
            phys_block = (
                decode_block_start
                + seq_idx * max_decode_blocks_per_seq
                + decode_blk_idx
            )
        slot_mapping_list.append(phys_block * block_size + slot_in_block)

        row = []
        for blk_in_seq in range(total_blocks_for_context):
            if blk_in_seq < prefill_blocks_per_seq:
                row.append(phys_prefill_base + blk_in_seq)
            else:
                decode_blk_idx = blk_in_seq - prefill_blocks_per_seq
                row.append(
                    decode_block_start
                    + seq_idx * max_decode_blocks_per_seq
                    + decode_blk_idx
                )
        block_table_rows.append(row)

    slot_mapping = torch.tensor(slot_mapping_list, dtype=torch.long, device=device)
    block_table = torch.tensor(block_table_rows, dtype=torch.int32, device=device)

    query_start_loc_gpu = torch.arange(
        batch_size + 1, dtype=torch.int32, device=device
    )
    query_start_loc_cpu = torch.arange(
        batch_size + 1, dtype=torch.int32, device="cpu"
    )
    seq_lens_gpu = torch.full(
        (batch_size,), context_len, dtype=torch.int32, device=device
    )
    seq_lens_cpu = torch.full(
        (batch_size,), context_len, dtype=torch.int32, device="cpu"
    )

    common_attn_metadata = CommonAttentionMetadata(
        query_start_loc=query_start_loc_gpu,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens_gpu,
        num_reqs=batch_size,
        num_actual_tokens=batch_size,
        max_query_len=1,
        max_seq_len=context_len,
        block_table_tensor=block_table,
        slot_mapping=slot_mapping,
        causal=True,
        _seq_lens_cpu=seq_lens_cpu,
        seq_lens_cpu_upper_bound=seq_lens_cpu,
        _num_computed_tokens_cpu=torch.full(
            (batch_size,), prefill_seq_len + step - 1,
            dtype=torch.int32, device="cpu",
        ),
    )

    model_runner = worker.model_runner
    attn_metadata_dict = {}
    slot_mapping_dict = {}

    for kv_cache_group_attn_groups in model_runner.attn_groups:
        for attn_group in kv_cache_group_attn_groups:
            builder = attn_group.get_metadata_builder(0)
            metadata = builder.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
            )
            for layer_name in attn_group.layer_names:
                attn_metadata_dict[layer_name] = metadata
                slot_mapping_dict[layer_name] = slot_mapping

    return attn_metadata_dict, slot_mapping_dict


def _get_or_create_attn_metadata(batch_size, seq_len, block_size, device, worker):
    """Create fresh attention metadata for the given parameters."""
    return _create_v1_attn_metadata(batch_size, seq_len, block_size, device, worker)


class _PocCaptureCtx:
    """Persistent buffers + cached attn metadata for FULL CUDA-graph capture of the
    isolated PoC prefill forward. Built once per (batch, seq_len, hidden, block) and
    reused: the forward fills the buffers in place, so the captured graph replays
    reading stable addresses. Our isolated forward never interleaves with engine
    steps, so caching the attn metadata is safe (unlike the hook/eager path)."""

    def __init__(self, batch, seq_len, hidden_size, block_size, device, dtype, worker):
        n = batch * seq_len
        self.num_tokens = n
        self.inputs_embeds = torch.zeros(n, hidden_size, device=device, dtype=dtype)
        self.positions = torch.arange(seq_len, device=device).repeat(batch)
        self.input_ids = torch.zeros(n, dtype=torch.long, device=device)
        self.attn_metadata, self.slot_mapping_dict = _create_v1_attn_metadata(
            batch, seq_len, block_size, device, worker)
        # num_tokens distinguishes our prefill key (batch*seq_len, large) from the
        # dispatcher's decode keys (num_tokens=batch, small) -> no graph collision.
        self.key = BatchDescriptor(num_tokens=n, num_reqs=batch, uniform=True)
        self.graph_ready = False


def _get_poc_capture_ctx(worker, batch, seq_len, hidden_size, block_size, device, dtype):
    cache = getattr(worker, "_poc_capture_ctxs", None)
    if cache is None:
        cache = {}
        worker._poc_capture_ctxs = cache
    k = (batch, seq_len, hidden_size, block_size)
    ctx = cache.get(k)
    if ctx is None:
        ctx = _PocCaptureCtx(batch, seq_len, hidden_size, block_size, device, dtype, worker)
        cache[k] = ctx
    return ctx


def _capture_poc_prefill(worker, vllm_config, model, ctx, device):
    """One-time warmup + FULL CUDA-graph capture of the isolated PoC prefill.

    vLLM only allows cudagraph capture during a guarded window; our PoC forward
    runs at request time, so we re-open the window explicitly (mirrors
    GPUModelRunner._capture_cudagraphs: set_cudagraph_capturing_enabled(True) +
    graph_capture(device)). ctx buffers must already be filled. The warmup pass
    (NONE mode) triggers compile + sizes workspaces outside the captured region;
    the FULL pass makes the server's CUDAGraphWrapper record the graph for ctx.key.
    """
    from vllm.compilation.monitor import set_cudagraph_capturing_enabled
    from vllm.distributed.parallel_state import graph_capture as _graph_capture

    # warmup: compiled, not captured (sizes workspaces, runs lazy init)
    with set_forward_context(
        ctx.attn_metadata, vllm_config, num_tokens=ctx.num_tokens,
        slot_mapping=ctx.slot_mapping_dict,
        cudagraph_runtime_mode=CUDAGraphMode.NONE, skip_compiled=False,
    ):
        with _poc_reflection(worker):
            model(input_ids=ctx.input_ids, positions=ctx.positions,
                  intermediate_tensors=None, inputs_embeds=ctx.inputs_embeds)
    torch.cuda.synchronize()

    set_cudagraph_capturing_enabled(True)
    try:
        with _graph_capture(device=device):
            with set_forward_context(
                ctx.attn_metadata, vllm_config, num_tokens=ctx.num_tokens,
                slot_mapping=ctx.slot_mapping_dict,
                cudagraph_runtime_mode=CUDAGraphMode.FULL,
                batch_descriptor=ctx.key, skip_compiled=False,
            ):
                with _poc_reflection(worker):
                    model(input_ids=ctx.input_ids, positions=ctx.positions,
                          intermediate_tensors=None, inputs_embeds=ctx.inputs_embeds)
    finally:
        set_cudagraph_capturing_enabled(False)
    torch.cuda.synchronize()
    ctx.graph_ready = True


# TODO: Should we get rid of this apprach?
def _select_poc_kv_scratch(
    kv_caches: list,
    dtype: torch.dtype,
    needed_elems: int,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
) -> Optional[torch.Tensor]:
    """Return a no-copy scratch view into KV cache memory, if safe.

    KV cache storage may use packed dtypes (e.g. ``uint8`` for FP8) or
    backend-specific non-contiguous layouts. Only reuse memory that already
    matches model embedding dtype and is contiguous so ``view(-1)`` does not
    allocate a copy.
    """
    for kv in kv_caches:
        if kv.dtype != dtype:
            continue
        if not kv.is_contiguous():
            continue
        if kv.numel() < needed_elems:
            continue
        return kv.view(-1)[:needed_elems].view(batch_size, seq_len, hidden_size)
    return None


@torch.inference_mode()
def execute_poc_forward(
    worker,
    block_hash: str,
    public_key: str,
    nonces: List[int],
    seq_len: int,
    hidden_size: int,
    k_dim: int = DEFAULT_K_DIM,
    poc_stronger_rng: bool = False,
    max_tokens: int = 0,
    inference_k_points_steps: Optional[Dict[int, List[int]]] = None,
) -> Optional[Dict[str, Any]]:
    """Execute batched PoC forward pass on a V1 worker.

    Processes all nonces in a single forward call for maximum throughput.

    decode-PoC (#1135): when ``max_tokens > 0`` the prefill is followed by
    ``max_tokens`` chained single-token decode steps.  Each decode token
    attends the FULL history (prefill + all previous decode steps) via real
    KV blocks — autoregressive semantics per the reference line
    (bs/poc-context-fix).  Each step quantizes the decode hidden state to the
    nearest sphere-codebook point; the per-step k-id array
    (``k_points_steps``, index 0 = prefill) is returned per nonce.  For
    validation requests, ``inference_k_points_steps`` maps nonce -> the host's
    reference k-ids; the validator counts divergences (``n_sphere_mismatches``)
    while teacher-forcing the trajectory with the reference k (no cascading).
    """
    device = worker.device
    dtype = worker.model_config.dtype
    model = worker.model_runner.model
    vllm_config = worker.vllm_config
    batch_size = len(nonces)

    tp_group = get_tp_group()
    is_tp_driver = tp_group.rank_in_group == 0

    # TP SYNC
    if tp_group.world_size > 1:
        dist.barrier(group=tp_group.cpu_group)
        if is_tp_driver:
            broadcast_tensor_dict({
                "poc_go": True,
                "seq_len": seq_len,
                "hidden_size": hidden_size,
                "nonces": nonces,
                "k_dim": k_dim,
                "poc_stronger_rng": poc_stronger_rng,
                "max_tokens": max_tokens,
            }, src=0)
        else:
            broadcast_data = broadcast_tensor_dict(src=0)
            seq_len = int(broadcast_data["seq_len"])
            hidden_size = int(broadcast_data["hidden_size"])
            nonces = list(broadcast_data["nonces"])
            k_dim = int(broadcast_data["k_dim"])
            batch_size = len(nonces)
            poc_stronger_rng = bool(broadcast_data["poc_stronger_rng"])
            max_tokens = int(broadcast_data["max_tokens"])
            # inference_k_points_steps is intentionally not broadcast: every
            # rank already receives it verbatim as a collective_rpc argument.
            # All TP ranks need the same map — teacher forcing folds it into
            # prev_k, which seeds the decode-step forward inputs.

    pp_group = get_pp_group()

    do_decode = max_tokens > 0
    if do_decode and pp_group.world_size > 1:
        # The decode loop issues per-step model() forwards that require every
        # PP stage to participate, but only the last PP rank reaches the
        # post-prefill code.  Rather than silently skip decode (which would
        # emit a valid-looking prefill-only artifact, as the reference does),
        # refuse explicitly.  The check is deterministic across ranks
        # (max_tokens is broadcast), so all ranks raise together.
        raise RuntimeError(
            "decode-PoC (max_tokens>0) is not supported with pipeline "
            f"parallelism (pp_world_size={pp_group.world_size}); use TP only."
        )

    # Pre-forward sync
    if tp_group.world_size > 1:
        dist.barrier(group=tp_group.cpu_group)
    torch.cuda.synchronize()

    _ensure_layer_hooks(worker, block_hash, hidden_size)

    # Get block_size and prepare attention metadata (cached, reused)
    block_size = _get_block_size(worker)
    attn_metadata, slot_mapping_dict = _get_or_create_attn_metadata(
        batch_size, seq_len, block_size, device, worker
    )

    # Positions for the batch
    positions = torch.arange(seq_len, device=device).repeat(batch_size)

    # Generate inputs for all nonces at once
    intermediate_tensors = None
    inputs_embeds = None

    if pp_group.is_first_rank:
        kv_caches = getattr(worker.model_runner, "kv_caches", [])
        needed_elems = batch_size * seq_len * hidden_size
        kv_scratch = _select_poc_kv_scratch(
            kv_caches, dtype, needed_elems, batch_size, seq_len, hidden_size,
        )
        if kv_scratch is not None:
            from .gpu_random import _seed_from_string, _normal
            for i, nonce in enumerate(nonces):
                seed = _seed_from_string(
                    f"{block_hash}_{public_key}_nonce{nonce}")
                vals = _normal(seed, seq_len * hidden_size, device)
                kv_scratch[i].copy_(vals.view(seq_len, hidden_size).to(dtype))
                del vals
            inputs_embeds = kv_scratch
        else:
            _gen_fn = generate_inputs_concat_murmur if poc_stronger_rng else generate_inputs
            inputs_embeds = _gen_fn(
                block_hash, public_key, nonces,
                dim=hidden_size, seq_len=seq_len,
                device=device, dtype=dtype,
            )
    else:
        intermediate_tensors = IntermediateTensors(
            pp_group.recv_tensor_dict(all_gather_group=get_tp_group())
        )

    prefill_eager = (vllm_config.model_config.enforce_eager
                     or os.environ.get("GONKA_POC_PREFILL_EAGER") == "1")
    # FULL CUDA-graph capture of the compiled prefill (Step 2). Requires the server
    # to run with a full-graph cudagraph_mode (model wrapped in CUDAGraphWrapper) and
    # the native wrapper (graphable reflection). First call captures the graph for our
    # BatchDescriptor; subsequent calls replay it. Only on the first PP rank (where
    # inputs_embeds exist) and not in eager mode.
    capture_on = (os.environ.get("GONKA_POC_CAPTURE") == "1"
                  and not prefill_eager
                  and pp_group.is_first_rank
                  and inputs_embeds is not None)

    if capture_on:
        ctx = _get_poc_capture_ctx(
            worker, batch_size, seq_len, hidden_size, block_size, device, dtype)
        ctx.inputs_embeds.copy_(inputs_embeds.reshape(-1, hidden_size))
        if not ctx.graph_ready:
            _capture_poc_prefill(worker, vllm_config, model, ctx, device)
        # replay: the CUDAGraphWrapper finds the captured graph for ctx.key and
        # replays it reading the (in-place filled) persistent buffers.
        with set_forward_context(
            ctx.attn_metadata, vllm_config,
            num_tokens=ctx.num_tokens,
            slot_mapping=ctx.slot_mapping_dict,
            cudagraph_runtime_mode=CUDAGraphMode.FULL,
            batch_descriptor=ctx.key,
            skip_compiled=False,
        ):
            with _poc_reflection(worker):
                hidden_states = model(
                    input_ids=ctx.input_ids,
                    positions=ctx.positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=ctx.inputs_embeds,
                )
    else:
        with set_forward_context(
            attn_metadata, vllm_config,
            num_tokens=batch_size * seq_len,
            slot_mapping=slot_mapping_dict,
            # Follow the server's compilation setting: compiled when not --enforce-eager,
            # eager otherwise. Removes the PoC eager-jail (decode-PoC compiled mode).
            skip_compiled=prefill_eager,
        ):
            with _poc_reflection(worker):
                hidden_states = model(
                    input_ids=(None if prefill_eager
                               else torch.zeros(batch_size * seq_len, dtype=torch.long, device=device)),
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds.view(-1, hidden_size) if inputs_embeds is not None else None,
                )

    # PP: send to next rank if not last
    if not pp_group.is_last_rank:
        if isinstance(hidden_states, IntermediateTensors):
            pp_group.send_tensor_dict(
                hidden_states.tensors, all_gather_group=get_tp_group()
            )
        return None

    # Handle tuple return
    if isinstance(hidden_states, tuple):
        hidden_states = hidden_states[0]

    # Extract last hidden per sequence
    hidden_states = hidden_states.view(batch_size, seq_len, -1)
    last_hidden = hidden_states[:, -1, :].float()  # [batch_size, hidden_size]

    # The prefill wrote each sequence's KV into blocks indexed by its ORIGINAL
    # batch position.  The NaN filters below shrink the batch, so the decode
    # loop must map each surviving row back to its original position to read
    # the right prefill blocks (the reference line misses this and reads
    # shifted KV after filtering).
    prefill_batch_size = batch_size
    prefill_seq_idx: List[int] = list(range(batch_size))

    # NaN detection
    nan_mask = torch.isnan(last_hidden).any(dim=-1)  # [batch_size]
    if nan_mask.any():
        clean_idx = (~nan_mask).nonzero(as_tuple=True)[0]
        nan_count = nan_mask.sum().item()
        logger.warning("NaN in %d/%d hidden states (GPU fault?)", nan_count, batch_size)

        if clean_idx.numel() == 0:
            logger.error("All %d nonces produced NaN — batch rejected", batch_size)
            return {"nonces": [], "vectors": np.empty((0, k_dim), dtype=np.float16)}

        last_hidden = last_hidden[clean_idx]
        nonces = [nonces[i] for i in clean_idx.tolist()]
        prefill_seq_idx = [prefill_seq_idx[i] for i in clean_idx.tolist()]
        batch_size = len(nonces)

    # Normalize to unit sphere
    last_hidden = last_hidden / (last_hidden.norm(dim=-1, keepdim=True) + 1e-8)

    # Batched k-dim pick + Haar rotation
    indices = random_pick_indices(block_hash, public_key, nonces, hidden_size, k_dim, device)
    xk = torch.gather(last_hidden, 1, indices)
    yk = apply_haar_rotation(block_hash, public_key, nonces, xk, device)

    # Normalize output vectors
    yk = yk / (yk.norm(dim=-1, keepdim=True) + 1e-8)

    # Convert to FP16
    vectors_f16 = yk.half().cpu().numpy()  # [batch_size, k_dim]

    # Late NaN check after FP16 conversion
    nan_out = np.isnan(vectors_f16).any(axis=1)
    if nan_out.any():
        clean = ~nan_out
        vectors_f16 = vectors_f16[clean]
        nonces = [n for n, c in zip(nonces, clean) if c]
        # Keep last_hidden aligned with the surviving nonces so the decode
        # loop below indexes the same rows as the returned artifacts.
        keep_idx = torch.tensor(
            [i for i, c in enumerate(clean) if c], device=device, dtype=torch.long
        )
        last_hidden = last_hidden[keep_idx]
        prefill_seq_idx = [s for s, c in zip(prefill_seq_idx, clean) if c]
        batch_size = len(nonces)
        logger.warning("NaN in FP16 output — %d nonces filtered", nan_out.sum())

    result: Dict[str, Any] = {
        "nonces": nonces,
        "vectors": vectors_f16,
    }

    # -------------------------------------------------------------------------
    # decode-PoC (#1135): sphere-quantized chained decode steps.
    # Prefill-only PoC v2 (max_tokens == 0) returns above without touching this.
    # -------------------------------------------------------------------------
    if do_decode and batch_size > 0:
        codebook = get_sphere_codebook().to(device=device, dtype=last_hidden.dtype)

        # Prefill k (step 0): pick SPHERE_DIM dims, project, nearest codebook.
        # step=0 salts the seed (..._pick_256_decode0, reference line format);
        # the PoC v2 artifact pick above stays on the legacy seed (step=None).
        sph_idx0 = random_pick_indices(
            block_hash, public_key, nonces, hidden_size, SPHERE_DIM, device,
            step=0,
        )
        xk_sph0 = project_to_sphere(torch.gather(last_hidden, 1, sph_idx0))
        k0_list: List[int] = nearest_sphere_index(xk_sph0, codebook).cpu().tolist()

        # Per-nonce reference k-ids for validation requests (None for generation).
        inf_steps_per_nonce: List[Optional[List[int]]] = [
            (inference_k_points_steps.get(n) if inference_k_points_steps else None)
            for n in nonces
        ]
        # n_sphere_mismatches: -1 = generation (non-validation) request.
        mismatch_count: List[int] = [
            0 if s is not None else -1 for s in inf_steps_per_nonce
        ]
        k_points_steps_per_nonce: List[List[int]] = [[k] for k in k0_list]

        # Compare prefill k against the reference (step 0).
        for i, inf in enumerate(inf_steps_per_nonce):
            if inf is not None and len(inf) > 0 and k0_list[i] != inf[0]:
                mismatch_count[i] += 1

        # Initial prev_k: validation nonces seed with the reference k so both
        # servers run the same decode trajectory.
        prev_k: List[int] = [
            (inf_steps_per_nonce[i][0]
             if inf_steps_per_nonce[i] is not None and len(inf_steps_per_nonce[i]) > 0
             else k0_list[i])
            for i in range(batch_size)
        ]

        # Decode KV block layout: decode blocks live AFTER the prefill blocks
        # (reference line layout).  Decode steps attend the full history, so
        # prefill KV must stay valid for the whole loop and each step's new
        # token KV needs a slot that does not collide with prefill blocks.
        # decode_block_start clears the ORIGINAL prefill region — the batch
        # may have shrunk in the NaN filters above.
        prefill_blocks_per_seq = math.ceil(seq_len / block_size)
        decode_block_start = prefill_batch_size * prefill_blocks_per_seq
        max_decode_blocks_per_seq = (
            math.ceil((seq_len + max_tokens) / block_size)
            - prefill_blocks_per_seq
            + 1
        )
        required_blocks = (
            decode_block_start + batch_size * max_decode_blocks_per_seq
        )
        num_gpu_blocks = getattr(
            worker.vllm_config.cache_config, "num_gpu_blocks", None
        )
        if num_gpu_blocks is not None and required_blocks > num_gpu_blocks:
            # Out-of-range slot writes corrupt memory silently; refuse instead.
            raise RuntimeError(
                f"decode-PoC needs {required_blocks} KV blocks "
                f"(batch={batch_size}, seq_len={seq_len}, "
                f"max_tokens={max_tokens}) but only {num_gpu_blocks} exist; "
                "reduce the PoC batch size or max_tokens."
            )

        for step in range(1, max_tokens + 1):
            if tp_group.world_size > 1:
                # Rendezvous + pin the trajectory: all TP ranks must seed the
                # step's forward with the same prev_k (reference line parity).
                dist.barrier(group=tp_group.cpu_group)
                if is_tp_driver:
                    broadcast_tensor_dict({"prev_k": prev_k}, src=0)
                else:
                    prev_k = list(broadcast_tensor_dict(src=0)["prev_k"])

            # Decode embedding seeded by prev_k (one token per nonce).
            decode_embeds = generate_decode_inputs(
                block_hash, public_key, nonces, prev_k,
                step=step, dim=hidden_size, device=device, dtype=dtype,
            )  # [batch_size, 1, hidden_size]
            decode_pos = torch.full(
                (batch_size,), seq_len + step - 1, device=device, dtype=torch.long
            )
            # Fresh full-history metadata every step (stale metadata -> all-NaN):
            # the new token attends all seq_len + step context tokens.
            dec_attn_metadata, dec_slot_mapping = (
                _create_decode_attn_metadata_with_history(
                    batch_size, seq_len, step, block_size, device, worker,
                    prefill_blocks_per_seq, max_decode_blocks_per_seq,
                    decode_block_start,
                    prefill_seq_idx=prefill_seq_idx,
                )
            )

            with set_forward_context(
                dec_attn_metadata, vllm_config,
                num_tokens=batch_size,
                slot_mapping=dec_slot_mapping,
                # decode-PoC compiled mode: follow server compilation setting.
                skip_compiled=(vllm_config.model_config.enforce_eager or os.environ.get("GONKA_POC_DECODE_EAGER") == "1"),
            ):
                with _poc_reflection(worker):
                    hs_dec = model(
                        input_ids=(None if (vllm_config.model_config.enforce_eager or os.environ.get("GONKA_POC_DECODE_EAGER") == "1")
                                   else torch.zeros(batch_size, dtype=torch.long, device=device)),
                        positions=decode_pos,
                        intermediate_tensors=None,
                        inputs_embeds=decode_embeds.view(-1, hidden_size),
                    )

            if isinstance(hs_dec, tuple):
                hs_dec = hs_dec[0]
            last_hidden_dec = hs_dec.view(batch_size, 1, -1)[:, 0, :].float()
            last_hidden_dec = last_hidden_dec / (
                last_hidden_dec.norm(dim=-1, keepdim=True) + 1e-8
            )

            # Dimension subset is chained on prev_k AND salted with the step
            # index (without the step there are only SPHERE_POINTS subsets).
            sph_idx_dec = random_pick_indices(
                block_hash, public_key, nonces, hidden_size, SPHERE_DIM, device,
                prev_point_ids=prev_k, step=step,
            )
            xk_sph_dec = project_to_sphere(
                torch.gather(last_hidden_dec, 1, sph_idx_dec)
            )
            step_k_list: List[int] = nearest_sphere_index(
                xk_sph_dec, codebook
            ).cpu().tolist()

            for i, computed_k in enumerate(step_k_list):
                k_points_steps_per_nonce[i].append(computed_k)

            # Teacher forcing: validation nonces advance on the reference k and
            # count divergences; generation nonces advance on their own k.
            new_prev_k: List[int] = []
            for i, computed_k in enumerate(step_k_list):
                inf = inf_steps_per_nonce[i]
                if inf is not None and step < len(inf):
                    if computed_k != inf[step]:
                        mismatch_count[i] += 1
                    new_prev_k.append(inf[step])
                else:
                    new_prev_k.append(computed_k)
            prev_k = new_prev_k

        result["k_points_steps"] = k_points_steps_per_nonce
        result["n_sphere_mismatches"] = mismatch_count

    return result

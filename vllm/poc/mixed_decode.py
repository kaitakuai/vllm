"""Phase 2: step-driven mixed decode-PoC support.

A decode-PoC request runs ONE decode token per scheduler step, mixed with chat
in the same forward (instead of running its whole decode loop inside one
pure-batch call). Its KV is allocated on demand by the KV manager exactly like
chat (pure dynamic KV, no reserved blocks) so its prefill KV persists and each
decode step reads it. ``sphere_k`` is chained across steps via the per-request
state held here.

Each PoC request carries a single nonce (routes fan out per-nonce via
``generate(poc_params)``), so the decode state is per-(request, nonce).

The slot/layout helpers at the top are pure (no torch) and unit-tested. The
model-runner helpers at the bottom (moved out of gpu_model_runner.py to keep the
core vLLM footprint minimal) take the GPUModelRunner as ``runner``. Decode-PoC is
always step-driven and mixed with chat (one PoC decode token per scheduler step,
fused into the chat forward — chat is never frozen); validation runs pure.
"""
from dataclasses import dataclass, field

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

# Bound on consecutive chat-prefill defers before a decoding PoC is forced an
# exclusive step (fairness valve — keeps PoC from starving under chat churn).
POC_DEFER_LIMIT = 4


def slice_sampling_metadata(sm, rows, device):
    """Restrict a SamplingMetadata to `rows` (input_batch indices), so the sampler
    runs on chat rows only. PoC rows have no sampling semantics; keeping them out
    avoids stale/oversized penalty tensors and the per-row param mismatch."""
    import dataclasses

    idx = torch.tensor(rows, device=device, dtype=torch.long)
    keep = set(rows)
    remap = {old: new for new, old in enumerate(rows)}

    def take_t(t):
        return None if t is None else t[idx]

    def take_list(lst):
        return [lst[i] for i in rows] if lst else lst

    def remap_dict(d):
        return None if d is None else {remap[k]: v for k, v in d.items() if k in keep}

    return dataclasses.replace(
        sm,
        temperature=take_t(sm.temperature),
        top_p=take_t(sm.top_p),
        top_k=take_t(sm.top_k),
        generators=remap_dict(sm.generators) or {},
        logprob_token_ids=remap_dict(sm.logprob_token_ids),
        prompt_token_ids=take_t(sm.prompt_token_ids),
        frequency_penalties=take_t(sm.frequency_penalties),
        presence_penalties=take_t(sm.presence_penalties),
        repetition_penalties=take_t(sm.repetition_penalties),
        output_token_ids=take_list(sm.output_token_ids),
        spec_token_ids=take_list(sm.spec_token_ids),
        allowed_token_ids_mask=take_t(sm.allowed_token_ids_mask),
        bad_words_token_ids=remap_dict(sm.bad_words_token_ids),
        enforced_next_token_ids=take_t(sm.enforced_next_token_ids),
    )

def decode_only_mixing_gate(
    *,
    mixed_cudagraph: bool,
    poc_decode_pending: bool,
    poc_will_prefill: bool,
    chat_will_prefill: bool,
    consecutive_defers: int,
    defer_limit: int = POC_DEFER_LIMIT,
) -> tuple[bool, bool, int]:
    """Decide (defer_chat, defer_poc, consecutive_defers) so chat and PoC share a
    forward only when both decode; prefills run isolated. Mutually exclusive defers.
    Pure (unit-testable). With mixed_cudagraph=False reduces to the original
    behaviour (defer_chat=poc_decode_pending, defer_poc=False). The valve bounds
    consecutive chat-prefill defers so chat churn can't starve a decoding PoC.
    """
    defer_chat = poc_decode_pending or (mixed_cudagraph and poc_will_prefill)
    defer_poc = mixed_cudagraph and (not defer_chat) and chat_will_prefill
    if defer_poc:
        consecutive_defers += 1
        if consecutive_defers > defer_limit:
            # Give the decoding PoC one exclusive (pure-decode, graphable) step.
            defer_poc, defer_chat, consecutive_defers = False, True, 0
    else:
        consecutive_defers = 0
    return defer_chat, defer_poc, consecutive_defers


def poc_is_pure_path(poc_params) -> bool:
    """True for prefill-only PoC (max_tokens == 0), which has no decode loop. All
    decode — generation and validation — runs step-driven. Pure (unit-testable)."""
    return poc_params.max_tokens == 0


def poc_step_num_tokens(poc_params, num_computed_tokens: int) -> int:
    """Tokens to schedule for a PoC request this step: mixed decode generation
    prefills seq_len once then 1 token/step; the pure / prefill-only path is a
    single seq_len step. Pure (unit-testable)."""
    if not poc_is_pure_path(poc_params):
        return poc_params.seq_len if num_computed_tokens == 0 else 1
    return poc_params.seq_len


def aligned_step(own_k: int, reference_k):
    """One validation comparison step, shared by both decode call-sites.
    reference_k = the reference trajectory's sphere_k for this step, or None when
    generating. Returns (mismatch_delta, next_prev_k): when validating, count a
    mismatch if own_k differs and seed the next step from the reference k (aligned,
    no cascade); when generating, seed from own_k."""
    if reference_k is None:
        return 0, own_k
    return (1 if own_k != reference_k else 0), reference_k


def poc_share_budget(poc_share: float, token_budget: int) -> int:
    """PoC's slice of a step's compute (token) budget. poc_share=0 -> PoC blocked
    this step; 1.0 -> PoC may use the whole budget. Pure (unit-testable)."""
    return int(poc_share * token_budget)


def poc_alloc_footprint(poc_params, num_new_tokens: int) -> int:
    """Dynamic-KV blocks to allocate: the pure path runs the whole decode loop in
    one step so it allocates seq_len+max_tokens upfront; the mixed path allocates
    one step's tokens. Pure (unit-testable)."""
    if poc_is_pure_path(poc_params):
        return poc_params.seq_len + poc_params.max_tokens
    return num_new_tokens


@dataclass
class PoCDecodeState:
    """Per-request decode state, carried across scheduler steps."""
    nonce: int
    slot: int
    seq_len: int
    max_tokens: int
    # number of decode steps completed so far (0 == only prefill done).
    step: int = 0
    # previous step's sphere_k; seeds the next step's input embedding + the
    # per-step random dimension selection. -1 before the prefill sphere_k is set.
    prev_k: int = -1
    # full sphere_k trajectory: prefill k, then one per decode step.
    k_points_steps: list[int] = field(default_factory=list)
    # the prefill artifact vector (base64), set at the prefill step.
    vector_b64: str = ""
    n_sphere_mismatches: int = 0
    # validation reference trajectory (enforced_k_steps), or None for
    # generation. index 0 = prefill k, 1..N = decode-step k. Drives aligned_step.
    reference: list | None = None
    # --- GPU-native chaining: prev_k stays on device so the per-step host sync
    # disappears (-> async scheduling works). The trajectory accumulates on device
    # and is copied to host ONCE at end-of-sequence (emit-once), so no per-step
    # delta crosses the IPC boundary. ---
    base_seeds: "torch.Tensor | None" = None     # [1] int64 per-nonce base (set once)
    prev_k_t: "torch.Tensor | None" = None        # [1] int64, chained on device
    reference_t: "torch.Tensor | None" = None     # [R] int64 uploaded reference
    mismatch_t: "torch.Tensor | None" = None      # [1] int64 on-device accumulator
    n_nan_t: "torch.Tensor | None" = None         # [1] int64 non-finite-step counter (device)
    k_steps_t: list = field(default_factory=list)  # list of [1] int64; cat+tolist at end
    # debug only (PoCParams.debug): the per-step pre-snap sphere slices — the same
    # q whose argmax is sphere_k — kept so the documented PoCOutput.sph_values_steps
    # contract (v1/outputs.py) can be emitted. list of [1, SPHERE_DIM] float tensors,
    # index 0 = prefill, 1..N = decode; device-accumulated like k_steps_t (no
    # per-step host sync), encoded once at emit.
    q_steps_t: list = field(default_factory=list)


class PoCMixedDecodeManager:
    """Per-request decode-state pool for step-driven mixed decode-PoC.

    One instance per model runner (lazily created). A finite pool of
    ``poc_max_batch_size`` state slots (sphere_k chaining + step counter); the
    scheduler caps concurrent decode-PoC requests to that many, so ``allocate``
    never starves in a correct configuration (returns ``None`` defensively if it
    would). KV itself is paged/dynamic via the manager — slots hold no blocks.
    """

    def __init__(self, poc_max_batch_size: int):
        self._free_slots: list[int] = list(range(poc_max_batch_size))
        self._state: dict[str, PoCDecodeState] = {}

    def get(self, req_id: str) -> PoCDecodeState | None:
        return self._state.get(req_id)

    def allocate(self, req_id: str, nonce: int, seq_len: int,
                 max_tokens: int) -> PoCDecodeState | None:
        existing = self._state.get(req_id)
        if existing is not None:
            return existing
        if not self._free_slots:
            return None
        slot = self._free_slots.pop(0)
        st = PoCDecodeState(
            nonce=nonce, slot=slot, seq_len=seq_len, max_tokens=max_tokens
        )
        self._state[req_id] = st
        return st

    def free(self, req_id: str) -> None:
        st = self._state.pop(req_id, None)
        if st is not None:
            self._free_slots.append(st.slot)


def get_decode_manager(runner) -> "PoCMixedDecodeManager":
    """Lazily get/create the per-runner mixed-decode manager."""
    mgr = getattr(runner, "_poc_mixed_decode_mgr", None)
    if mgr is None:
        mgr = PoCMixedDecodeManager(runner.cache_config.poc_max_batch_size)
        runner._poc_mixed_decode_mgr = mgr
    return mgr


def setup_decode_poc(runner, poc_requests) -> bool:
    """Entry hook (called from gpu_model_runner before _prepare_inputs).

    For each decode-PoC request (max_tokens>0): grab a state slot + refresh its
    per-request decode step counter. KV is pure dynamic: the scheduler-allocated
    paged block-table row already drives decode (see below).

    Returns True if any decode-PoC is active this step, signalling the caller to
    route the batch through the unified step-driven path. Returns False when there
    are no decode-PoC requests. Generation and validation both run here; validation
    carries its reference trajectory in PoCDecodeState.reference (aligned compare).
    """
    decode_reqs = [r for r in poc_requests if r.poc_params.max_tokens > 0]
    if not decode_reqs:
        return False
    mgr = get_decode_manager(runner)
    # Pure dynamic KV: the scheduler allocated real (paged) blocks via the
    # manager, so _prepare_inputs already built the correct block-table row +
    # slot_mapping. We only track decode state (step + reference trajectory).
    for r in decode_reqs:
        pp = r.poc_params
        st = mgr.allocate(r.req_id, pp.nonce, pp.seq_len, pp.max_tokens)
        if st is None:
            # Pool exhausted (scheduler caps to poc_max_batch_size; defensive).
            logger.warning("PoC mixed-decode slot pool exhausted for %s",
                           r.req_id)
            continue
        # decode step = tokens computed beyond prefill (0 during the prefill step)
        st.step = max(0, r.num_computed_tokens - pp.seq_len)
        st.reference = pp.enforced_k_steps  # None for generation
    return True


# ---------------------------------------------------------------------------
# Mixed-batch model-runner helpers (moved out of gpu_model_runner.py to keep the
# core vLLM footprint minimal). Each takes the GPUModelRunner as `runner`.
# ---------------------------------------------------------------------------

def build_unified_mixed_batch_inputs(
    runner,
    scheduler_output: "SchedulerOutput",
    chat_input_ids: torch.Tensor | None,
    chat_inputs_embeds: torch.Tensor | None,
    chat_positions: torch.Tensor,
    poc_req_ids: set,
    num_total_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict]]:
    """Build unified inputs for mixed batch (chat + PoC in same forward).

    CRITICAL: Preserves scheduler's token order to match slot_mapping.
    Tokens are built in the exact order of runner.input_batch.req_ids.

    Args:
        scheduler_output: The scheduler output with token counts
        chat_input_ids: Chat token IDs [num_total_tokens] or None
        chat_inputs_embeds: Chat embeddings [num_total_tokens, hidden] or None
        chat_positions: Chat positions [num_total_tokens]
        poc_req_ids: Set of PoC request IDs
        num_total_tokens: Total scheduled tokens (chat + PoC)

    Returns:
        Tuple of:
        - unified_embeds: [num_total_tokens, hidden_size]
        - unified_positions: [num_total_tokens]
        - poc_position_mask: [num_total_tokens] bool tensor (True = PoC)
        - poc_metadata: List of dicts with PoC request info
    """
    from vllm.poc.gpu_random import generate_inputs

    hidden_size = runner.model_config.get_hidden_size()
    num_reqs = runner.input_batch.num_reqs
    req_ids = runner.input_batch.req_ids

    tokens_per_req = [scheduler_output.num_scheduled_tokens[req_id]
                      for req_id in req_ids]

    unified_embeds = torch.empty(
        (num_total_tokens, hidden_size),
        dtype=runner.dtype,
        device=runner.device,
    )
    unified_positions = torch.empty(
        num_total_tokens,
        dtype=chat_positions.dtype,
        device=runner.device,
    )
    poc_position_mask = torch.zeros(
        num_total_tokens,
        dtype=torch.bool,
        device=runner.device,
    )
    poc_metadata = []

    offset = 0
    # Decode-step embeddings are generated in ONE batched call after this loop
    # (was per-nonce). Each entry: (decode_state, decode_step, offset).
    decode_embed_jobs = []

    for req_idx in range(num_reqs):
        req_id = req_ids[req_idx]
        num_tokens = tokens_per_req[req_idx]

        if num_tokens <= 0:
            continue

        if req_id in poc_req_ids:
            req_state = runner.requests[req_id]
            poc_params = req_state.poc_params
            seq_len = poc_params.seq_len
            mgr = getattr(runner, "_poc_mixed_decode_mgr", None)
            st = mgr.get(req_id) if mgr is not None else None

            if st is not None and req_state.num_computed_tokens >= seq_len:
                # Phase 2 decode step: one token, embed chained from prev sphere_k.
                # GPU-native: prev_k is a device tensor (set by the previous step's
                # output processing) -> no host sync -> async-scheduling safe. The
                # embedding itself is generated in ONE batched call after the loop.
                from vllm.poc.gpu_random import decode_base_seeds
                decode_step = req_state.num_computed_tokens - seq_len + 1
                if st.base_seeds is None:
                    st.base_seeds = decode_base_seeds(
                        poc_params.block_hash, poc_params.public_key,
                        [poc_params.nonce], runner.device)
                decode_embed_jobs.append((st, decode_step, offset))
                unified_positions[offset] = req_state.num_computed_tokens
                poc_position_mask[offset] = True
                poc_metadata.append({
                    'type': 'poc', 'req_id': req_id, 'start_idx': offset,
                    'length': 1, 'poc_params': poc_params,
                    'decode_state': st, 'decode_step': decode_step,
                })
                offset += 1
            else:
                # Prefill (prefill-only PoC, or the prefill step of a decode-PoC).
                poc_len = num_tokens
                poc_embeds = generate_inputs(
                    poc_params.block_hash,
                    poc_params.public_key,
                    [poc_params.nonce],
                    dim=hidden_size,
                    seq_len=poc_len,
                    device=runner.device,
                    dtype=runner.dtype,
                ).squeeze(0)  # [poc_len, hidden]
                unified_embeds[offset:offset + poc_len] = poc_embeds
                unified_positions[offset:offset + poc_len] = torch.arange(
                    poc_len, device=runner.device, dtype=chat_positions.dtype
                )
                poc_position_mask[offset:offset + poc_len] = True
                poc_metadata.append({
                    'type': 'poc', 'req_id': req_id, 'start_idx': offset,
                    'length': poc_len, 'poc_params': poc_params,
                    'decode_state': st,
                })
                offset += poc_len

        else:
            if chat_inputs_embeds is not None:
                unified_embeds[offset:offset + num_tokens] = (
                    chat_inputs_embeds[offset:offset + num_tokens]
                )
            elif chat_input_ids is not None:
                token_ids = chat_input_ids[offset:offset + num_tokens]
                # v15 renamed get_input_embeddings -> embed_input_ids (and
                # the model is the cudagraph wrapper, which forwards it).
                chat_embeds = runner.model.embed_input_ids(input_ids=token_ids)
                unified_embeds[offset:offset + num_tokens] = chat_embeds

            unified_positions[offset:offset + num_tokens] = (
                chat_positions[offset:offset + num_tokens]
            )
            offset += num_tokens

    # Batched decode-step embeddings: one generate_decode_inputs_gpu call for the
    # whole nonce-batch (per-row identical to the old per-nonce calls).
    if decode_embed_jobs:
        from vllm.poc.gpu_random import generate_decode_inputs_gpu
        base_seeds = torch.cat([j[0].base_seeds for j in decode_embed_jobs])  # [B]
        prev_k = torch.cat([j[0].prev_k_t for j in decode_embed_jobs])        # [B]
        steps = torch.tensor([j[1] for j in decode_embed_jobs],
                             dtype=torch.int64, device=runner.device)
        embeds = generate_decode_inputs_gpu(
            base_seeds, prev_k, steps,
            dim=hidden_size, device=runner.device, dtype=runner.dtype)  # [B, 1, H]
        offs = torch.tensor([j[2] for j in decode_embed_jobs],
                            dtype=torch.long, device=runner.device)
        unified_embeds.index_copy_(0, offs, embeds[:, 0])   # [B, H] -> rows offs

    return unified_embeds, unified_positions, poc_position_mask, poc_metadata


def process_poc_outputs_from_hidden(
    runner,
    hidden_states: torch.Tensor,
    poc_metadata: list[dict],
) -> dict[str, "PoCOutput"]:
    from vllm.v1.outputs import PoCOutput
    from vllm.poc.gpu_random import (
        random_pick_indices, apply_haar_rotation,
        decode_base_seeds, random_pick_indices_gpu,
    )
    from vllm.poc.data import encode_vector
    from vllm.poc.sphere import (
        SPHERE_DIM, get_sphere_codebook, project_to_sphere, snap_with_guard,
    )

    poc_outputs = {}
    # Codebook is constant (device-only); cache on the runner instead of re-copying
    # it every step. nearest_sphere_index casts to float, so dtype here is moot.
    codebook = getattr(runner, "_poc_codebook", None)
    if codebook is None:
        codebook = get_sphere_codebook().to(device=runner.device)
        runner._poc_codebook = codebook

    # Decode steps are the hot path; collect them and run ONE batched set of GPU
    # ops for the whole nonce-batch below (was a per-nonce Python loop = B× the
    # kernel launches). Prefill-only / prefill-step PoCs (rare, once per request)
    # stay inline.
    decode_metas = []

    for meta in poc_metadata:
        st = meta.get('decode_state')
        if st is not None and 'decode_step' in meta:
            decode_metas.append(meta)
            continue

        end = meta['start_idx'] + meta['length']
        poc_params = meta['poc_params']
        nonce = poc_params.nonce

        last_hidden = hidden_states[end - 1].float()
        last_hidden = last_hidden / (last_hidden.norm() + 1e-8)
        hidden_size = last_hidden.shape[-1]

        def _vector_b64():
            idx = random_pick_indices(
                poc_params.block_hash, poc_params.public_key, [nonce],
                hidden_size, poc_params.k_dim, runner.device)
            xk = last_hidden[idx[0]]
            yk = apply_haar_rotation(
                poc_params.block_hash, poc_params.public_key, [nonce],
                xk.unsqueeze(0), runner.device)[0]
            yk = yk / (yk.norm() + 1e-8)
            return encode_vector(yk.half().cpu().numpy())

        def _sphere_from_idx(sph):
            """hidden -> (sphere index, non-finite mask, pre-snap slice) as [1]/[1]/
            [1, SPHERE_DIM] TENSORs (no .item(), so the chain stays on GPU and async
            scheduling works)."""
            xk_sphere = project_to_sphere(torch.gather(last_hidden.unsqueeze(0), 1, sph))
            k_, bad_ = snap_with_guard(xk_sphere, codebook)  # (k[1] int64, bad[1] bool)
            return k_, bad_, xk_sphere

        if st is None:
            # Prefill-only PoC: just the vector_b64 artifact.
            poc_outputs[meta['req_id']] = PoCOutput(
                nonce=nonce, vector_b64=_vector_b64())
            continue

        # Prefill step of a decode-PoC: compute the prefill sphere_k (k0) and start
        # the on-device trajectory. Decode is scored on k_points_steps; the chain
        # (prev_k_t) stays on device until end-of-sequence (emit-once).
        if st.base_seeds is None:
            st.base_seeds = decode_base_seeds(
                poc_params.block_hash, poc_params.public_key, [nonce], runner.device)
        sph0 = random_pick_indices(
            poc_params.block_hash, poc_params.public_key, [nonce],
            hidden_size, SPHERE_DIM, runner.device)
        k0_t, bad0, q0 = _sphere_from_idx(sph0)             # [1]/[1]/[1,SPHERE_DIM]
        st.k_steps_t = [k0_t]
        # debug: keep the pre-snap slice so sph_values_steps can be emitted
        st.q_steps_t = [q0.detach()] if poc_params.debug else []
        st.mismatch_t = torch.zeros(1, dtype=torch.int64, device=runner.device)
        st.n_nan_t = bad0.to(torch.int64)                   # [1] non-finite-step counter
        if st.reference is not None:
            st.reference_t = torch.tensor(
                st.reference, dtype=torch.int64, device=runner.device)
            ref0 = st.reference_t[0:1]
            # a non-finite step is a compute fault, not a mismatch -> exclude it
            st.mismatch_t += ((k0_t != ref0) & (k0_t >= 0)).to(torch.int64)
            st.prev_k_t = ref0                              # aligned (teacher-forced)
        else:
            st.prev_k_t = k0_t

    # Batched decode step: one set of GPU ops (seed -> pick -> sphere) for the whole
    # nonce-batch. Per-row results are identical to the old per-nonce calls (same
    # seeds, murmur, topk, gather, argmax), so artifacts are unchanged.
    if decode_metas:
        device = runner.device
        H = hidden_states.shape[-1]
        idxs = [m['start_idx'] + m['length'] - 1 for m in decode_metas]
        lh = hidden_states[idxs].float()                       # [B, H]
        lh = lh / (lh.norm(dim=-1, keepdim=True) + 1e-8)
        base_seeds = torch.cat([m['decode_state'].base_seeds for m in decode_metas])
        prev_k = torch.cat([m['decode_state'].prev_k_t for m in decode_metas])
        steps = torch.tensor([m['decode_step'] for m in decode_metas],
                             dtype=torch.int64, device=device)
        sph = random_pick_indices_gpu(base_seeds, prev_k, steps, H, SPHERE_DIM, device)
        # snap_with_guard: argmax(NaN) is garbage -> non-finite rows return k=-1
        # (compute fault, NOT fraud). bad_all stays on device (no per-step sync).
        q_all = project_to_sphere(torch.gather(lh, 1, sph))          # [B, SPHERE_DIM]
        k_all, bad_all = snap_with_guard(q_all, codebook)            # [B] int64, [B] bool

        for i, meta in enumerate(decode_metas):
            st = meta['decode_state']
            step = meta['decode_step']
            k_t = k_all[i:i + 1]                               # [1] tensor (view)
            st.k_steps_t.append(k_t)
            if meta['poc_params'].debug:
                # debug: keep the pre-snap slice (device, like k_steps_t — no sync)
                st.q_steps_t.append(q_all[i:i + 1].detach())
            st.n_nan_t += bad_all[i:i + 1].to(torch.int64)    # device accumulate (no sync)
            if st.reference_t is not None and step < st.reference_t.shape[0]:
                ref = st.reference_t[step:step + 1]
                # exclude a non-finite step from the mismatch count (fault != fraud)
                st.mismatch_t += ((k_t != ref) & (k_t >= 0)).to(torch.int64)
                st.prev_k_t = ref                             # aligned (teacher-forced)
            else:
                st.prev_k_t = k_t
            if step >= st.max_tokens:
                # End-of-sequence: ONE host copy of the whole trajectory + count
                # (emit-once). This single terminal PoCOutput is what the engine drains.
                k_points = torch.cat(st.k_steps_t).tolist()
                n_mismatches = int(st.mismatch_t.item())
                n_nan = int(st.n_nan_t.item())
                if n_nan:
                    logger.warning(
                        "PoC decode nonce %s: %d/%d non-finite hidden step(s) "
                        "(compute fault, NOT fraud; excluded from mismatch rate) — "
                        "trajectory suspect, re-run on a clean GPU",
                        meta['poc_params'].nonce, n_nan, len(k_points))
                # debug: emit the documented sph_values_steps contract (v1/outputs.py)
                # — one fp16-LE base64 slice per trajectory step, same wire format as
                # vector_b64 (data.encode_vector). Single host copy, emit-once.
                sph_vals = []
                if meta['poc_params'].debug and st.q_steps_t:
                    sph_vals = [
                        encode_vector(q.squeeze(0).cpu().numpy())
                        for q in st.q_steps_t]
                poc_outputs[meta['req_id']] = PoCOutput(
                    nonce=meta['poc_params'].nonce,
                    vector_b64="",
                    k_points_steps=k_points,
                    n_sphere_mismatches=(
                        n_mismatches if st.reference is not None else -1),
                    n_nan_steps=n_nan,
                    sph_values_steps=sph_vals,
                )
                get_decode_manager(runner).free(meta['req_id'])

    return poc_outputs


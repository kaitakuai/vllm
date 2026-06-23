"""Deterministic GPU-based random generation for PoC.

Core primitives for generating reproducible random tensors seeded by
(block_hash, public_key, nonce). Used by the production inference pipeline.

OPTIMIZED: Serial Python loops replaced with batched GPU operations.
"""
import hashlib
import math
from typing import List, Optional

import torch


def _seed_from_string(seed_string: str) -> int:
    h = hashlib.sha256(seed_string.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _murmur3_32(keys: torch.Tensor, seed: int) -> torch.Tensor:
    """Murmur3 hash for int32 keys. Returns int64 to preserve full uint32 range."""
    c1, c2 = 0xCC9E2D51, 0x1B873593

    h = torch.full_like(keys, seed & 0xFFFFFFFF, dtype=torch.int64)
    k = keys.to(torch.int64) & 0xFFFFFFFF

    k = (k * c1) & 0xFFFFFFFF
    k = ((k << 15) | (k >> 17)) & 0xFFFFFFFF
    k = (k * c2) & 0xFFFFFFFF

    h = h ^ k
    h = ((h << 13) | (h >> 19)) & 0xFFFFFFFF
    h = (h * 5 + 0xE6546B64) & 0xFFFFFFFF

    h = h ^ (h >> 16)
    h = (h * 0x85EBCA6B) & 0xFFFFFFFF
    h = h ^ (h >> 13)
    h = (h * 0xC2B2AE35) & 0xFFFFFFFF
    h = h ^ (h >> 16)
    return h


def _batched_murmur3_32(keys: torch.Tensor, seeds: torch.Tensor) -> torch.Tensor:
    """Batched Murmur3 hash with per-row seeds.

    Args:
        keys: [batch_size, n] int32 tensor
        seeds: [batch_size, 1] int64 tensor
    Returns:
        [batch_size, n] int64 tensor
    """
    c1, c2 = 0xCC9E2D51, 0x1B873593

    h = (seeds & 0xFFFFFFFF).expand_as(keys.to(torch.int64))
    k = keys.to(torch.int64) & 0xFFFFFFFF

    k = (k * c1) & 0xFFFFFFFF
    k = ((k << 15) | (k >> 17)) & 0xFFFFFFFF
    k = (k * c2) & 0xFFFFFFFF

    h = h ^ k
    h = ((h << 13) | (h >> 19)) & 0xFFFFFFFF
    h = (h * 5 + 0xE6546B64) & 0xFFFFFFFF

    h = h ^ (h >> 16)
    h = (h * 0x85EBCA6B) & 0xFFFFFFFF
    h = h ^ (h >> 13)
    h = (h * 0xC2B2AE35) & 0xFFFFFFFF
    h = h ^ (h >> 16)
    return h


def _batched_normal(seeds: list, n: int, device: torch.device) -> torch.Tensor:
    """Generate batched normal random numbers for multiple seeds.

    Args:
        seeds: List of integer seeds
        n: Number of random numbers per seed
        device: Target device

    Returns:
        Tensor of shape [len(seeds), n]
    """
    batch_size = len(seeds)
    n_pairs = (n + 1) // 2
    total = n_pairs * 2

    indices = torch.arange(total, device=device, dtype=torch.int32).unsqueeze(0).expand(batch_size, -1)
    seed_tensor = torch.tensor(seeds, dtype=torch.int64, device=device).unsqueeze(1)

    h = _batched_murmur3_32(indices, seed_tensor)
    u = h.to(torch.float32) / 4294967296.0

    u1 = u[:, :n_pairs]
    u2 = u[:, n_pairs:]
    u1 = torch.clamp(u1, min=1e-10)

    z0 = torch.sqrt(-2.0 * torch.log(u1)) * torch.cos(2.0 * math.pi * u2)
    z1 = torch.sqrt(-2.0 * torch.log(u1)) * torch.sin(2.0 * math.pi * u2)
    return torch.cat([z0, z1], dim=1)[:, :n]


def _uniform(seed: int, n: int, device: torch.device) -> torch.Tensor:
    indices = torch.arange(n, device=device, dtype=torch.int32)
    hashes = _murmur3_32(indices, seed)
    return hashes.to(torch.float32) / 4294967296.0


def _normal(seed: int, n: int, device: torch.device) -> torch.Tensor:
    n_pairs = (n + 1) // 2
    u = _uniform(seed, n_pairs * 2, device)
    u1, u2 = u[:n_pairs], u[n_pairs:]
    u1 = torch.clamp(u1, min=1e-10)
    z0 = torch.sqrt(-2.0 * torch.log(u1)) * torch.cos(2.0 * math.pi * u2)
    z1 = torch.sqrt(-2.0 * torch.log(u1)) * torch.sin(2.0 * math.pi * u2)
    return torch.cat([z0, z1])[:n]


def generate_inputs(
    block_hash: str,
    public_key: str,
    nonces: List[int],
    dim: int,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Generate deterministic input embeddings for PoC."""
    batch_size = len(nonces)
    result = torch.empty(batch_size, seq_len, dim, device=device, dtype=dtype)
    for i, nonce in enumerate(nonces):
        seed_str = f"{block_hash}_{public_key}_nonce{nonce}"
        seed = _seed_from_string(seed_str)
        normal = _normal(seed, seq_len * dim, device)
        result[i] = normal.view(seq_len, dim).to(dtype)
    return result


def generate_inputs_concat_murmur(
    block_hash: str,
    public_key: str,
    nonces: List[int],
    dim: int,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Generate deterministic input embeddings using concat-murmur (stronger RNG).

    Uses all 256 bits of SHA256 by splitting into 8 × 32-bit sub-seeds.
    Each sub-seed generates one segment of length ceil(n/8) via the existing
    murmur3 pipeline; segments are concatenated.
    """
    batch_size = len(nonces)
    result = torch.empty(batch_size, seq_len, dim, device=device, dtype=dtype)
    n = seq_len * dim
    seg_len = (n + 7) // 8  # ceil(n/8); last segment may be shorter

    for i, nonce in enumerate(nonces):
        h = hashlib.sha256(
            f"{block_hash}_{public_key}_nonce{nonce}".encode()
        ).digest()
        sub_seeds = [int.from_bytes(h[j:j + 4], 'big') for j in range(0, 32, 4)]

        segments = [
            _normal(s, min(seg_len, n - k * seg_len), device)
            for k, s in enumerate(sub_seeds)
            if k * seg_len < n
        ]
        flat = torch.cat(segments)[:n]
        result[i] = flat.view(seq_len, dim).to(dtype)

    return result


def generate_decode_inputs(
    block_hash: str,
    public_key: str,
    nonces: List[int],
    prev_k: List[int],
    step: int,
    dim: int,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Generate deterministic decode-step input embedding chained to previous sphere_k point.

    The seed incorporates the nearest sphere_k point chosen in the previous
    step so that each decode step is deterministically linked to its predecessor.

    Seed format is byte-identical to the decode-PoC reference (#1135):
    ``{block_hash}_{public_key}_nonce{nonce}_decode{step}_k{k}`` (note: no
    underscore after ``nonce``, matching ``generate_inputs``).

    Args:
        block_hash: Block hash for seeding
        public_key: Public key for seeding
        nonces: List of nonce values
        prev_k: Nearest sphere index from the previous step (one per nonce)
        step: Decode step index (1-based; step 0 is the prefill)
        dim: Hidden dimension size
        device: Target device
        dtype: Output dtype (default float16)

    Returns:
        Tensor of shape [batch_size, 1, dim]
    """
    batch_size = len(nonces)
    result = torch.empty(batch_size, 1, dim, device=device, dtype=dtype)

    for i, (nonce, k) in enumerate(zip(nonces, prev_k)):
        seed_str = f"{block_hash}_{public_key}_nonce{nonce}_decode{step}_k{k}"
        seed = _seed_from_string(seed_str)
        normal = _normal(seed, dim, device)
        result[i, 0] = normal.to(dtype)

    return result


def generate_target(
    block_hash: str,
    public_key: str,
    dim: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate deterministic target unit vector."""
    seed_str = f"{block_hash}_{public_key}_target"
    seed = _seed_from_string(seed_str)
    normal = _normal(seed, dim, device)
    target = normal.to(dtype)
    target = target / target.norm()
    return target


def generate_householder_vector(
    seed_str: str,
    dim: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate a single unit vector for Householder reflection."""
    seed = _seed_from_string(seed_str)
    v = _normal(seed, dim, device)
    return v / v.norm()


def apply_householder(
    x: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Apply Householder reflection: H @ x = x - 2*(v.x)*v"""
    dot = (x * v).sum(dim=-1, keepdim=True)
    return x - 2 * dot * v


def random_pick_indices(
    block_hash: str,
    public_key: str,
    nonces: List[int],
    dim: int,
    k: int,
    device: torch.device,
    prev_point_ids: Optional[List[int]] = None,
    step: Optional[int] = None,
) -> torch.Tensor:
    """Pick k dimensions per nonce deterministically (vectorized).

    When ``step`` is None the seed format and result are byte-identical to
    the production PoC v2 path
    (``{block_hash}_{public_key}_nonce_{nonce}_pick_{k}``); chaining is not
    allowed on this path.

    decode-PoC (#1135) callers pass ``step`` (0 = prefill sphere pick,
    1..N = decode steps), which salts the seed
    (``..._pick_{k}_decode{step}``), plus ``prev_point_ids`` on decode steps
    (``..._pick_{k}_decode{step}_k_{prev}``).  Folding the step index in
    keeps the dimension subset unique per step: chained on prev_k alone
    there are only SPHERE_POINTS possible subsets, which is predictable.
    Formats match the decode-PoC reference line (bs/poc-context-fix); the
    only change vs. the reference is vectorization of the murmur3 scoring.
    """
    if k <= 0 or k > dim:
        raise ValueError(f"k must be in [1, dim], got k={k}, dim={dim}")
    if step is None and prev_point_ids is not None:
        raise ValueError(
            "prev_point_ids requires step (decode-PoC path); "
            "the legacy PoC v2 path (step=None) does not chain"
        )

    batch_size = len(nonces)

    seeds = []
    for i, nonce in enumerate(nonces):
        if step is None:
            seeds.append(_seed_from_string(
                f"{block_hash}_{public_key}_nonce_{nonce}_pick_{k}"
            ))
        elif prev_point_ids is None:
            seeds.append(_seed_from_string(
                f"{block_hash}_{public_key}_nonce_{nonce}_pick_{k}_decode{step}"
            ))
        else:
            seeds.append(_seed_from_string(
                f"{block_hash}_{public_key}_nonce_{nonce}_pick_{k}_decode{step}_k_{prev_point_ids[i]}"
            ))

    all_idx = torch.arange(dim, device=device, dtype=torch.int32).unsqueeze(0).expand(batch_size, -1)
    seed_tensor = torch.tensor(seeds, dtype=torch.int64, device=device).unsqueeze(1)
    scores = _batched_murmur3_32(all_idx, seed_tensor)

    _, chosen = torch.topk(-scores, k=k, largest=True, sorted=False, dim=1)
    return chosen.to(torch.int64)


def apply_haar_rotation(
    block_hash: str,
    public_key: str,
    nonces: List[int],
    x: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Apply Haar-random rotation via k-1 Householder reflections (vectorized)."""
    batch_size, k = x.shape
    if k <= 0:
        raise ValueError(f"k must be positive, got k={k}")

    y = x.clone()

    all_seeds_by_step = []
    for j in range(k - 1):
        step_seeds = []
        for nonce in nonces:
            step_seeds.append(_seed_from_string(
                f"{block_hash}_{public_key}_nonce_{nonce}_haar_hh_{k}_{j}"
            ))
        all_seeds_by_step.append(step_seeds)

    for j in range(k - 1):
        v_batch = _batched_normal(all_seeds_by_step[j], k, device)
        v_batch = v_batch / (v_batch.norm(dim=-1, keepdim=True) + 1e-30)
        v_batch = v_batch.to(y.dtype)

        dot = (y * v_batch).sum(dim=-1, keepdim=True)
        y = y - 2 * dot * v_batch

    return y


# ---------------------------------------------------------------------------
# GPU-native decode chaining  (ported from axeltec poc-v0.20-decode-poc-cg)
# ---------------------------------------------------------------------------
# The legacy decode chain (generate_decode_inputs / random_pick_indices) builds
# each step's seed with SHA256 of a string containing prev_k -> prev_k must be a
# Python int -> a GPU->CPU sync every step -> forces --no-async-scheduling.
#
# These functions keep prev_k on the GPU as a tensor and mix it in with on-GPU
# murmur3. block_hash/public_key/nonce are CONSTANT per request, so their SHA256
# base seed is computed ONCE (no per-step sync); only step + prev_k are mixed per
# step, fully on device. This is a NEW seed scheme: VALUES DIFFER from the
# SHA256-string path. Selected at runtime via GONKA_POC_SEED_SCHEME=gpu_native
# (legacy = default, for A-parity). Distinct salts separate the decode-embed
# stream from the dim-pick stream.

_SALT_DECODE_EMBED = 0x0D
_SALT_DECODE_PICK = 0x91
_MIX_A = 0x9E3779B1  # golden-ratio odd constant
_MIX_B = 0x85EBCA77


def decode_base_seeds(
    block_hash: str,
    public_key: str,
    nonces: List[int],
    device: torch.device,
) -> torch.Tensor:
    """Per-nonce base seed (constant for the whole request) -> [B] int64 on device.
    Computed once; carries no per-step dependency, so the host SHA256 here is fine."""
    seeds = [_seed_from_string(f"{block_hash}_{public_key}_nonce{n}") for n in nonces]
    return torch.tensor(seeds, dtype=torch.int64, device=device)


def _step_seeds(
    base_seeds: torch.Tensor, step, prev_k: torch.Tensor, salt: int
) -> torch.Tensor:
    """Per-step seed = on-GPU murmur3 mixing base (per nonce) with step + prev_k.

    base_seeds [B] int64 (constant), prev_k [B] int64 (chained on device, NEVER
    .item()'d). step is a host int (same for the whole batch) OR a [B] int64 tensor
    (per-row step). Returns [B] int64 fully on device, so the decode chain has no
    GPU->CPU sync. The per-row result is identical to calling this once per row."""
    if torch.is_tensor(step):
        step_term = step.to(torch.int64).view(-1) * _MIX_B + salt
    else:
        step_term = int(step) * _MIX_B + salt
    key = ((prev_k.to(torch.int64).view(-1) & 0xFFFFFFFF) * _MIX_A
           + step_term) & 0xFFFFFFFF
    return _batched_murmur3_32(key.view(-1, 1), base_seeds.view(-1, 1)).view(-1)


def _batched_normal_t(seeds: torch.Tensor, n: int, device: torch.device) -> torch.Tensor:
    """Like _batched_normal but `seeds` is already an int64 tensor [B] (no host
    list). Returns [B, n] standard normals, fully on device."""
    batch_size = seeds.shape[0]
    n_pairs = (n + 1) // 2
    total = n_pairs * 2
    indices = torch.arange(total, device=device, dtype=torch.int32).unsqueeze(0).expand(batch_size, -1)
    h = _batched_murmur3_32(indices, seeds.view(-1, 1))
    u = h.to(torch.float32) / 4294967296.0
    u1 = torch.clamp(u[:, :n_pairs], min=1e-10)
    u2 = u[:, n_pairs:]
    z0 = torch.sqrt(-2.0 * torch.log(u1)) * torch.cos(2.0 * math.pi * u2)
    z1 = torch.sqrt(-2.0 * torch.log(u1)) * torch.sin(2.0 * math.pi * u2)
    return torch.cat([z0, z1], dim=1)[:, :n]


def generate_decode_inputs_gpu(
    base_seeds: torch.Tensor,
    prev_k: torch.Tensor,
    step: int,
    dim: int,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """GPU-native counterpart of generate_decode_inputs: next decode-step input
    embedding chained to prev_k (tensor). Returns [B, 1, dim]."""
    seeds = _step_seeds(base_seeds, step, prev_k, _SALT_DECODE_EMBED)
    return _batched_normal_t(seeds, dim, device).to(dtype).unsqueeze(1)


def random_pick_indices_gpu(
    base_seeds: torch.Tensor,
    prev_k: torch.Tensor,
    step: int,
    dim: int,
    k: int,
    device: torch.device,
) -> torch.Tensor:
    """GPU-native counterpart of random_pick_indices (decode): k dims per row,
    seed chained to prev_k (tensor). Returns [B, k] int64."""
    if k <= 0 or k > dim:
        raise ValueError(f"k must be in [1, dim], got k={k}, dim={dim}")
    seeds = _step_seeds(base_seeds, step, prev_k, _SALT_DECODE_PICK)
    all_idx = torch.arange(dim, device=device, dtype=torch.int32).unsqueeze(0).expand(seeds.shape[0], -1)
    scores = _batched_murmur3_32(all_idx, seeds.view(-1, 1))
    _, chosen = torch.topk(-scores, k=k, largest=True, sorted=False, dim=1)
    return chosen.to(torch.int64)

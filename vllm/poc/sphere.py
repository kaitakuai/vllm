"""PoC sphere codebook: project a hidden-state slice onto an equidistant
codebook on the unit sphere and snap it to the nearest point (sphere_k).

This is the discrete PoC fingerprint of the forward pass — pure math, no model
execution (PoC runs natively through vLLM; see native.py + mixed_decode.py).
"""
import hashlib
import os
from typing import Optional

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# SPHERE_DIM: dimension of the hidden-state slice projected onto the sphere.
# SPHERE_POINTS: number of equidistant codebook points on that sphere.
SPHERE_DIM = 256
SPHERE_POINTS = 16


def project_to_sphere(v: torch.Tensor) -> torch.Tensor:
    """Normalize [..., dim] vectors to the unit sphere (L2 norm = 1)."""
    return v / (v.norm(dim=-1, keepdim=True) + 1e-8)


def _halton_on_sphere(n_points: int, dim: int) -> torch.Tensor:
    """Return n_points deterministic, low-discrepancy unit vectors on S^(dim-1).

    Uses the Halton sequence (base-prime per dimension) mapped to the sphere
    via the logit transform. Identical output for any call with the same args.
    """
    _PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
    coords: list[list[float]] = []
    for d in range(dim):
        base = _PRIMES[d % len(_PRIMES)]
        col: list[float] = []
        for i in range(1, n_points + 1):
            f, r = 1.0, 0.0
            j = i
            while j > 0:
                f /= base
                r += f * (j % base)
                j //= base
            col.append(r)
        coords.append(col)
    raw = torch.tensor(coords, dtype=torch.float32).T.clamp(0.01, 0.99)
    pts = torch.log(raw / (1.0 - raw))
    return project_to_sphere(pts)


def build_equidistant_codebook(
    n_points: int,
    dim: int,
    n_steps: int = 500,
    lr: float = 0.05,
) -> torch.Tensor:
    """Build approximately equidistant points on S^(dim-1) via Thomson problem.

    Minimizes electrostatic repulsion energy so points spread uniformly.
    Deterministic initialisation (Halton sequence). Result is cached in
    _SPHERE_CODEBOOK at module load time.
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


# Frozen, checked-in codebook (audit #1). Built ONCE on CPU (deterministic
# Halton init + Adam), normalized once, saved here. Every node loads these exact
# bytes so prover and validator share a bit-identical codebook — the only
# cross-node consensus guarantee. Adam's backward is NOT bit-reproducible across
# torch/BLAS/HW, so a rebuild is unsafe: load the file, assert the hash.
_CODEBOOK_FILE = os.path.join(os.path.dirname(__file__), "sphere_codebook.pt")
EXPECTED_CODEBOOK_SHA256 = (
    "2201d1093583f43f328dd4643ddf1f34737f6da8c1956395c43fd093e357e34f")

_SPHERE_CODEBOOK: Optional[torch.Tensor] = None


def _codebook_sha256(cb: torch.Tensor) -> str:
    return hashlib.sha256(
        cb.detach().cpu().float().contiguous().numpy().tobytes()).hexdigest()


def get_sphere_codebook() -> torch.Tensor:
    """Return the cached sphere codebook, resolving (and verifying) it on first use.

    Resolution order:
      1. ``GONKA_POC_SPHERE_CODEBOOK=<path>`` — load that frozen file (explicit
         cross-node pin; e.g. a single file shared across validators).
      2. the checked-in ``sphere_codebook.pt`` — load and assert it matches
         ``EXPECTED_CODEBOOK_SHA256`` (catches a drifted/corrupt file).
      3. fallback: rebuild via Adam — NOT consensus-safe; logged as a warning.

    The codebook is stored pre-normalized float32 and used as-is (no GPU
    re-projection, which would add per-HW ULP drift and silently re-open #1).
    """
    global _SPHERE_CODEBOOK
    if _SPHERE_CODEBOOK is not None:
        return _SPHERE_CODEBOOK

    path = os.environ.get("GONKA_POC_SPHERE_CODEBOOK")
    if path:
        cb = torch.load(path, map_location="cpu").float().contiguous()
        if tuple(cb.shape) != (SPHERE_POINTS, SPHERE_DIM):
            raise ValueError(
                f"GONKA_POC_SPHERE_CODEBOOK shape {tuple(cb.shape)} != "
                f"expected ({SPHERE_POINTS}, {SPHERE_DIM})")
        logger.info("PoC sphere codebook: loaded override %s (sha256=%s)",
                    path, _codebook_sha256(cb)[:16])
    elif os.path.exists(_CODEBOOK_FILE):
        cb = torch.load(_CODEBOOK_FILE, map_location="cpu").float().contiguous()
        digest = _codebook_sha256(cb)
        if digest != EXPECTED_CODEBOOK_SHA256:
            raise ValueError(
                f"sphere codebook {digest} != frozen reference "
                f"{EXPECTED_CODEBOOK_SHA256}; refusing to run (consensus risk)")
        logger.info("PoC sphere codebook: loaded frozen reference (sha256=%s)",
                    digest[:16])
    else:
        cb = build_equidistant_codebook(SPHERE_POINTS, SPHERE_DIM)
        logger.warning(
            "PoC sphere codebook: %s missing, REBUILT via Adam (sha256=%s) — "
            "NOT consensus-safe across nodes", _CODEBOOK_FILE,
            _codebook_sha256(cb)[:16])

    _SPHERE_CODEBOOK = cb
    return cb


def nearest_sphere_index(query: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """Return the index of the nearest codebook point for each query vector.

    Args:
        query: unit vectors [batch, dim]
        codebook: unit vectors [SPHERE_POINTS, dim]
    Returns:
        index tensor [batch] with values in [0, SPHERE_POINTS)
    """
    sims = query.float() @ codebook.float().T
    return sims.argmax(dim=-1)


def snap_with_guard(
    query: torch.Tensor, codebook: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """`nearest_sphere_index` with a non-finite guard.

    A non-finite query row is a COMPUTE FAULT (GPU contention / kernel fault /
    stale attention metadata), NOT fraud — but `argmax(NaN)` silently returns a
    garbage index that would count as a mismatch and read as fraud. So we detect
    it and return sentinel ``-1`` for those rows, plus the boolean fault mask, so
    the caller can log it and EXCLUDE those steps from the mismatch rate.

    Returns ``(k, bad)``: ``k`` [batch] int64 (``-1`` where non-finite),
    ``bad`` [batch] bool (True where the query row was non-finite).
    """
    bad = ~torch.isfinite(query).all(dim=-1)             # [batch]
    k = nearest_sphere_index(query, codebook)            # [batch]
    k = torch.where(bad, torch.full_like(k, -1), k)
    return k, bad

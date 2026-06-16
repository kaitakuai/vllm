"""PoC sphere codebook: project a hidden-state slice onto an equidistant
codebook on the unit sphere and snap it to the nearest point (sphere_k).

This is the discrete PoC fingerprint of the forward pass — pure math, no model
execution (PoC runs natively through vLLM; see native.py + mixed_decode.py).
"""
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


_SPHERE_CODEBOOK: torch.Tensor = build_equidistant_codebook(SPHERE_POINTS, SPHERE_DIM)


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

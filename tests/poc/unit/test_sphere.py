"""Unit tests for the PoC sphere codebook (vllm/poc/sphere.py).

The sphere_k fingerprint = project a hidden-state slice to the unit sphere and
snap to the nearest codebook point. Pure, deterministic math — no GPU, no model.
"""
import torch

from vllm.poc.sphere import (
    SPHERE_DIM,
    SPHERE_POINTS,
    _SPHERE_CODEBOOK,
    project_to_sphere,
    nearest_sphere_index,
    build_equidistant_codebook,
)


def test_project_to_sphere_unit_norm():
    x = torch.randn(32, SPHERE_DIM) * 7.0  # arbitrary scale
    u = project_to_sphere(x)
    norms = u.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_project_to_sphere_handles_zero():
    # the +1e-8 guard means a zero vector must not produce NaN/inf
    u = project_to_sphere(torch.zeros(1, SPHERE_DIM))
    assert torch.isfinite(u).all()


def test_codebook_shape():
    assert _SPHERE_CODEBOOK.shape == (SPHERE_POINTS, SPHERE_DIM)


def test_codebook_points_are_unit_norm():
    norms = _SPHERE_CODEBOOK.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_codebook_points_are_distinct():
    # Thomson spread: no two points coincide. Min pairwise cosine < 1.
    cb = _SPHERE_CODEBOOK.float()
    sims = cb @ cb.T
    off_diag = sims - torch.eye(SPHERE_POINTS) * 2.0  # push diagonal out of the way
    assert off_diag.max().item() < 0.999, "two codebook points are (near) identical"


def test_nearest_index_maps_codebook_point_to_itself():
    # each codebook point's nearest neighbour is itself
    idx = nearest_sphere_index(_SPHERE_CODEBOOK, _SPHERE_CODEBOOK)
    assert torch.equal(idx, torch.arange(SPHERE_POINTS))


def test_nearest_index_in_range():
    q = project_to_sphere(torch.randn(64, SPHERE_DIM))
    idx = nearest_sphere_index(q, _SPHERE_CODEBOOK)
    assert idx.shape == (64,)
    assert int(idx.min()) >= 0 and int(idx.max()) < SPHERE_POINTS


def test_codebook_build_is_deterministic():
    # same args -> identical codebook (deterministic Halton init + Adam, no RNG).
    a = build_equidistant_codebook(SPHERE_POINTS, SPHERE_DIM, n_steps=20)
    b = build_equidistant_codebook(SPHERE_POINTS, SPHERE_DIM, n_steps=20)
    assert torch.equal(a, b)

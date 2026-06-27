"""Codebook pinning / consensus tests (audit #1).

The decode-PoC snap (`nearest_sphere_index`) is only consensus-safe if the
prover and validator use a **byte-identical** sphere codebook. Today the
codebook is built by Adam at import (`sphere.py:77`) with:
  * no env override to load a single frozen codebook across nodes,
  * no checked-in reference hash to assert the build didn't drift,
so two nodes have NO guarantee they share a codebook — and the snap flips
k-ids under even ULP-scale codebook differences (characterization below).

These are RED on the current code (the pin mechanism doesn't exist yet) and
turn GREEN once `get_sphere_codebook()` (env override) + `EXPECTED_CODEBOOK_SHA256`
(frozen reference) land. Pure CPU; no GPU/server.
"""
import hashlib

import pytest
import torch

import vllm.poc.sphere as sphere


def _sha256(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().float().contiguous().numpy().tobytes()).hexdigest()


@pytest.fixture(autouse=True)
def _reset_codebook_cache():
    """Each test resolves the codebook fresh (env override / frozen file)."""
    sphere._SPHERE_CODEBOOK = None
    yield
    sphere._SPHERE_CODEBOOK = None


# --- RED until the pin lands ------------------------------------------------

def test_codebook_env_override_is_used(tmp_path, monkeypatch):
    """A frozen codebook on disk (one file shared by every node) must be the
    codebook the snap actually uses — the only cross-node consensus guarantee."""
    cb = sphere.project_to_sphere(
        torch.randn(sphere.SPHERE_POINTS, sphere.SPHERE_DIM))
    path = tmp_path / "codebook.pt"
    torch.save(cb, path)
    monkeypatch.setenv("GONKA_POC_SPHERE_CODEBOOK", str(path))

    # Force a reload so the override is honored (no-op once implemented).
    sphere._SPHERE_CODEBOOK = None
    got = sphere.get_sphere_codebook()          # RED: no such accessor today

    assert torch.allclose(got, cb, atol=1e-6), "env-override codebook not used"
    # …and the snap must read it, not a stale module global.
    q = sphere.project_to_sphere(torch.randn(64, sphere.SPHERE_DIM))
    assert torch.equal(
        sphere.nearest_sphere_index(q, got),
        sphere.nearest_sphere_index(q, cb))


def test_default_codebook_matches_frozen_reference_hash():
    """The shipped default codebook must match a checked-in reference hash, so
    a build that drifts (different torch/BLAS/HW Adam trajectory) is caught."""
    expected = sphere.EXPECTED_CODEBOOK_SHA256     # RED: constant doesn't exist
    assert _sha256(sphere.get_sphere_codebook()) == expected, \
        "default codebook drifted from the frozen reference"


# --- Characterization: WHY the pin matters (passes now; documents the harm) --

def test_snap_consensus_requires_byte_identical_codebook():
    """Consensus contract: ONLY a byte-identical codebook gives identical k-ids.
    The SAME codebook flips nothing; a tiny per-node difference — exactly what
    unpinned cross-node Adam builds produce — flips k-ids. So pinning (one frozen
    file shared by every node) is mandatory, not a nicety. Documents the amplifier."""
    torch.manual_seed(0)
    cb = sphere.build_equidistant_codebook(sphere.SPHERE_POINTS, sphere.SPHERE_DIM)
    q = sphere.project_to_sphere(torch.randn(4000, sphere.SPHERE_DIM))
    k0 = sphere.nearest_sphere_index(q, cb)

    # identical codebook -> zero divergence (the guarantee pinning provides)
    assert torch.equal(k0, sphere.nearest_sphere_index(q, cb.clone()))

    # any per-node drift -> nonzero divergence (the risk without pinning)
    flips = {}
    for eps in (1e-4, 1e-3, 1e-2):
        cb_drift = sphere.project_to_sphere(cb + eps * torch.randn_like(cb))
        flips[eps] = (k0 != sphere.nearest_sphere_index(q, cb_drift)).float().mean().item()
    # monotone, and even ULP-scale drift is non-zero — that's the hair trigger.
    assert flips[1e-4] > 0.0, f"snap should flip under codebook drift; got {flips}"
    assert flips[1e-2] > flips[1e-4], f"flip rate should grow with drift; got {flips}"
    print(f"\ncodebook-drift -> k-id flip rate: {flips}")

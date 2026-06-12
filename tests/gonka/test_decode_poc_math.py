"""Unit tests for decode-PoC (#1135) deterministic math.

CPU-only; no GPU/server required.  Covers:
1. Seed derivation byte-identity (reference values frozen against donor).
2. generate_decode_inputs: shape, determinism, prev_k / step chaining.
3. random_pick_indices: prev=None == production PoC v2; prev changes result.
4. Sphere codebook: nearest_sphere_index argmax, build determinism, env override.
"""
import pytest

torch = pytest.importorskip("torch")

from vllm.poc.gpu_random import (  # noqa: E402
    _seed_from_string,
    generate_decode_inputs,
    random_pick_indices,
)
from vllm.poc import poc_model_runner as pmr  # noqa: E402

CPU = torch.device("cpu")
BH = "deadbeef" * 8
PK = "cafebabe" * 8


# ---------------------------------------------------------------------------
# 1. Seed derivation — frozen reference values (must match donor 71573d91c)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed_str,expected", [
    (f"{BH}_{PK}_nonce42", 2840397398),
    (f"{BH}_{PK}_nonce42_decode1_k7", 3049228576),
    (f"{BH}_{PK}_nonce42_decode2_k3", 2004860272),
    (f"{BH}_{PK}_nonce_42_pick_12", 178240286),
    (f"{BH}_{PK}_nonce_42_pick_256", 3115550570),
    (f"{BH}_{PK}_nonce_42_pick_256_k_7", 2414532222),
])
def test_seed_reference_values(seed_str, expected):
    assert _seed_from_string(seed_str) == expected


# ---------------------------------------------------------------------------
# 2. generate_decode_inputs
# ---------------------------------------------------------------------------

def test_decode_inputs_shape_and_determinism():
    nonces, prev_k, dim = [42, 99], [7, 3], 16
    a = generate_decode_inputs(BH, PK, nonces, prev_k, step=1, dim=dim, device=CPU)
    b = generate_decode_inputs(BH, PK, nonces, prev_k, step=1, dim=dim, device=CPU)
    assert a.shape == (2, 1, dim)
    assert torch.equal(a, b)  # deterministic


def test_decode_inputs_chained_on_prev_k_and_step():
    nonces, dim = [42], 16
    base = generate_decode_inputs(BH, PK, nonces, [7], step=1, dim=dim, device=CPU)
    diff_k = generate_decode_inputs(BH, PK, nonces, [8], step=1, dim=dim, device=CPU)
    diff_step = generate_decode_inputs(BH, PK, nonces, [7], step=2, dim=dim, device=CPU)
    assert not torch.equal(base, diff_k)    # prev_k folds into the seed
    assert not torch.equal(base, diff_step)  # step folds into the seed


# ---------------------------------------------------------------------------
# 3. random_pick_indices — backward-compat + chaining
# ---------------------------------------------------------------------------

def test_pick_prev_none_is_stable_and_within_range():
    nonces, dim, k = [42, 99, 7], 256, 12
    out1 = random_pick_indices(BH, PK, nonces, dim, k, CPU)
    out2 = random_pick_indices(BH, PK, nonces, dim, k, CPU)
    assert out1.shape == (3, k)
    assert torch.equal(out1, out2)            # deterministic (prod PoC v2 path)
    assert int(out1.min()) >= 0 and int(out1.max()) < dim
    # k distinct dims per nonce
    for row in out1:
        assert len(set(row.tolist())) == k


def test_pick_prev_changes_result():
    nonces, dim, k = [42], 256, 256
    no_prev = random_pick_indices(BH, PK, nonces, dim, k, CPU)
    with_prev = random_pick_indices(BH, PK, nonces, dim, k, CPU, prev_point_ids=[7])
    # same multiset of indices (it's a permutation of all dims for k==dim) but a
    # different seed must produce a different *ordering*.
    assert not torch.equal(no_prev, with_prev)
    # chaining on a different prev yields yet another ordering
    other_prev = random_pick_indices(BH, PK, nonces, dim, k, CPU, prev_point_ids=[8])
    assert not torch.equal(with_prev, other_prev)


# ---------------------------------------------------------------------------
# 4. Sphere codebook
# ---------------------------------------------------------------------------

def test_nearest_sphere_index_is_argmax_cosine():
    cb = pmr.build_equidistant_codebook(pmr.SPHERE_POINTS, pmr.SPHERE_DIM)
    # A query exactly on codebook point j must return j.
    for j in (0, 3, pmr.SPHERE_POINTS - 1):
        q = cb[j:j + 1].clone()
        idx = pmr.nearest_sphere_index(q, cb)
        assert int(idx[0]) == j


def test_codebook_build_is_deterministic_and_unit_norm():
    a = pmr.build_equidistant_codebook(pmr.SPHERE_POINTS, pmr.SPHERE_DIM)
    b = pmr.build_equidistant_codebook(pmr.SPHERE_POINTS, pmr.SPHERE_DIM)
    assert a.shape == (pmr.SPHERE_POINTS, pmr.SPHERE_DIM)
    assert torch.allclose(a, b)  # same machine/torch -> identical
    norms = a.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_get_sphere_codebook_caches(monkeypatch):
    monkeypatch.setattr(pmr, "_SPHERE_CODEBOOK", None)
    monkeypatch.delenv("GONKA_POC_SPHERE_CODEBOOK", raising=False)
    cb1 = pmr.get_sphere_codebook()
    cb2 = pmr.get_sphere_codebook()
    assert cb1 is cb2  # cached singleton


def test_get_sphere_codebook_env_override(tmp_path, monkeypatch):
    frozen = torch.nn.functional.normalize(
        torch.randn(pmr.SPHERE_POINTS, pmr.SPHERE_DIM), dim=-1
    )
    path = tmp_path / "codebook.pt"
    torch.save(frozen, path)
    monkeypatch.setattr(pmr, "_SPHERE_CODEBOOK", None)
    monkeypatch.setenv("GONKA_POC_SPHERE_CODEBOOK", str(path))
    cb = pmr.get_sphere_codebook()
    assert torch.allclose(cb, frozen, atol=1e-5)


def test_get_sphere_codebook_env_shape_mismatch(tmp_path, monkeypatch):
    bad = torch.randn(pmr.SPHERE_POINTS + 1, pmr.SPHERE_DIM)
    path = tmp_path / "bad.pt"
    torch.save(bad, path)
    monkeypatch.setattr(pmr, "_SPHERE_CODEBOOK", None)
    monkeypatch.setenv("GONKA_POC_SPHERE_CODEBOOK", str(path))
    with pytest.raises(ValueError):
        pmr.get_sphere_codebook()

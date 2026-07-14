"""Unit tests for the continuous vector-channel score (vllm/poc/validation.py
score_vector_channel + its run_validation plumbing).

The channel compares the prover's per-step pre-snap sphere slices
(sph_values_steps, fp16-LE base64 — same wire codec as vector_b64) against the
validator's own teacher-forced recompute: d_t = 1 - <q_p, q_v> over decode
steps (index 0 = prefill is excluded), mean per nonce. It is EVIDENCE next to
the k-based verdict, never a verdict change. Pure CPU math.
"""
import numpy as np

from vllm.poc.data import decode_vector, encode_vector
from vllm.poc.validation import run_validation, score_vector_channel


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def _steps(*vecs):
    """Encode a trajectory: prefill slice + the given decode-step slices."""
    prefill = _unit(np.ones(4))
    return [encode_vector(prefill)] + [encode_vector(_unit(v)) for v in vecs]


def test_wire_codec_roundtrip_keeps_unit_norm():
    rng = np.random.default_rng(7)
    q = _unit(rng.standard_normal(256))
    back = decode_vector(encode_vector(q))
    assert back.shape == (256,)
    # fp16 quantization noise only — far below any honest floor of interest
    assert abs(float(np.linalg.norm(back)) - 1.0) < 1e-3
    assert 1.0 - float(np.dot(_unit(back), q)) < 1e-6


def test_identical_slices_score_zero_and_orthogonal_score_one():
    e0, e1 = [1, 0, 0, 0], [0, 1, 0, 0]
    computed = [{"nonce": 0, "sph_values_steps": _steps(e0, e0)},
                {"nonce": 1, "sph_values_steps": _steps(e0, e0)}]
    # nonce 0: reference identical -> 0; nonce 1: both steps orthogonal -> 1
    ref = {0: _steps(e0, e0), 1: _steps(e1, e1)}
    score = score_vector_channel(computed, ref)
    by = {e["nonce"]: e for e in score["per_nonce"]}
    assert by[0]["mean_dist"] < 1e-6
    assert abs(by[1]["mean_dist"] - 1.0) < 1e-3
    assert score["n_nonces_scored"] == 2
    assert abs(score["max_nonce_dist"] - by[1]["mean_dist"]) < 1e-9


def test_prefill_slice_is_excluded_from_the_score():
    e0, e1 = [1, 0, 0, 0], [0, 1, 0, 0]
    # trajectories whose PREFILL slices disagree maximally but decode steps agree
    computed = [{"nonce": 0,
                 "sph_values_steps": [encode_vector(_unit(e0)),
                                      encode_vector(_unit(e0))]}]
    ref = {0: [encode_vector(_unit(e1)), encode_vector(_unit(e0))]}
    score = score_vector_channel(computed, ref)
    assert score["per_nonce"][0]["mean_dist"] < 1e-6   # prefill mismatch ignored


def test_non_finite_slice_is_a_fault_not_a_distance():
    e0 = [1, 0, 0, 0]
    bad = [float("nan"), 0, 0, 0]
    computed = [{"nonce": 0, "sph_values_steps": _steps(e0, bad)}]
    ref = {0: _steps(e0, e0)}
    score = score_vector_channel(computed, ref)
    e = score["per_nonce"][0]
    assert e["n_steps_scored"] == 1 and e["n_bad_steps"] == 1
    assert e["mean_dist"] < 1e-6                       # only the finite step counts


def test_score_is_none_without_vectors_on_both_sides():
    assert score_vector_channel([{"nonce": 0}], {0: _steps([1, 0, 0, 0])}) is None
    assert score_vector_channel(
        [{"nonce": 0, "sph_values_steps": _steps([1, 0, 0, 0])}], {}) is None


def test_run_validation_attaches_vector_score_but_verdict_stays_k_based():
    e0, e1 = [1, 0, 0, 0], [0, 1, 0, 0]
    arts = [{"nonce": 0, "k_points_steps": [1, 2, 3], "n_sphere_mismatches": 0,
             "sph_values_steps": _steps(e0, e0)}]
    ref = {0: _steps(e1, e1)}   # vector channel screams (dist ~1)...
    res = run_validation(arts, {}, n_total=1, use_trajectory=True,
                         ref_vectors=ref)
    assert "vector_score" in res
    assert abs(res["vector_score"]["mean_dist"] - 1.0) < 1e-3
    assert res["fraud_detected"] is False   # ...but the k-verdict is unchanged
    # and without ref vectors the response shape is exactly the old one
    res2 = run_validation(arts, {}, n_total=1, use_trajectory=True)
    assert "vector_score" not in res2
    assert {k for k in res2} == {"n_total", "n_mismatch", "mismatch_nonces",
                                 "per_nonce", "p_value", "fraud_detected"}

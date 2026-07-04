"""Per-nonce evidence persistence (decode-PoC validation).

`run_validation` computes a per-nonce `n_sphere_mismatches` for every artifact but
historically summed them and dropped the per-nonce array (val JSON `artifacts: []`),
so the report could only show an average — not the per-nonce honest/fraud separation
the anti-cheat evidence needs. These tests pin the contract that the per-nonce counts
are now RETURNED (and thus persistable by collect.py). Pure CPU; no GPU/server.
"""
import vllm.poc.validation as V


def _arts(counts, n_steps=17):
    """Fake validator computed_artifacts: nonce i has counts[i] sphere_k mismatches."""
    return [{"nonce": i, "k_points_steps": [0] * n_steps, "n_sphere_mismatches": c}
            for i, c in enumerate(counts)]


def test_per_nonce_returned_and_sums_to_total():
    counts = [0, 3, 57, 111]
    res = V.run_validation(_arts(counts), validation_map={}, n_total=len(counts),
                           use_trajectory=True, p_mismatch=0.1)
    pn = res["per_nonce"]
    assert [e["nonce"] for e in pn] == [0, 1, 2, 3]
    assert [e["n_sphere_mismatches"] for e in pn] == counts        # per-nonce preserved
    assert all(e["n_steps"] == 17 for e in pn)
    assert sum(e["n_sphere_mismatches"] for e in pn) == res["n_mismatch"]  # consistent w/ aggregate


def test_generation_marker_minus_one_counts_as_zero():
    # -1 == "generation, no reference" — must not corrupt per-nonce counts or the sum
    res = V.run_validation(_arts([-1, 5, -1]), validation_map={}, n_total=3,
                           use_trajectory=True, p_mismatch=0.1)
    assert [e["n_sphere_mismatches"] for e in res["per_nonce"]] == [0, 5, 0]
    assert res["n_mismatch"] == 5


def test_per_nonce_length_matches_nonce_count():
    res = V.run_validation(_arts([1] * 128), validation_map={}, n_total=128,
                           use_trajectory=True, p_mismatch=0.1)
    assert len(res["per_nonce"]) == 128          # every nonce represented (the chart's x-axis)


def test_codebook_hash_is_a_frozen_sha256():
    # the codebook hash stamped into report meta must be a real 64-hex frozen reference
    import vllm.poc.sphere as sphere
    h = sphere.EXPECTED_CODEBOOK_SHA256
    assert isinstance(h, str) and len(h) == 64 and all(c in "0123456789abcdef" for c in h)

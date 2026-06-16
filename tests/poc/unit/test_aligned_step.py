"""Unit tests for aligned_step — the single shared compare-and-seed used by every
PoC decode call-site (so generation and validation can't diverge).

Contract:
- generation (reference None): never a mismatch; seed next from own k.
- validation: mismatch only when own != reference; ALWAYS seed next from the
  reference k (aligned — a divergence does not cascade into later steps).
"""
from vllm.poc.mixed_decode import aligned_step


def test_generation_no_reference_seeds_from_own():
    assert aligned_step(7, None) == (0, 7)
    assert aligned_step(0, None) == (0, 0)


def test_validation_match_no_mismatch_seeds_from_reference():
    # own == reference → 0 mismatch, next seeded from reference (== own here)
    assert aligned_step(5, 5) == (0, 5)


def test_validation_divergence_counts_and_reseeds_from_reference():
    # own != reference → 1 mismatch, but next prev_k is the REFERENCE (not own),
    # so the trajectories stay aligned and the divergence does not cascade.
    assert aligned_step(9, 4) == (1, 4)


def test_no_cascade_property():
    # A wrong step is counted once; because we reseed from the reference, the
    # NEXT step starts from the same place the reference did.
    delta1, prev1 = aligned_step(9, 4)      # diverged
    assert (delta1, prev1) == (1, 4)
    delta2, prev2 = aligned_step(4, 4)      # back in step with reference
    assert (delta2, prev2) == (0, 4)        # not counted again

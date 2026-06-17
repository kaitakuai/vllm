"""Unit tests for the pure PoC scheduling-policy helpers extracted from the
scheduler into vllm.poc.mixed_decode (keeps the core vLLM footprint thin).

These encode the contract the scheduler relies on:
  - validation recompute + prefill-only run the PURE (exclusive) path;
  - decode GENERATION runs step-driven mixed (seq_len prefill, then 1 token/step);
  - dynamic-KV allocation reserves the whole seq_len+max_tokens upfront for the
    pure path (it runs the whole decode loop in one step) but only one step's
    tokens for the mixed path.
"""
from vllm.poc.mixed_decode import (
    poc_is_pure_path, poc_step_num_tokens, poc_alloc_footprint, poc_share_budget,
)
from vllm.poc.poc_params import PoCParams


def _params(*, max_tokens, validation, seq_len=256):
    return PoCParams(
        block_hash="0xabc", public_key="0xpub", block_height=1, nonce=0,
        seq_len=seq_len, max_tokens=max_tokens,
        enforced_k_steps=[1, 2, 3] if validation else None,
    )


def test_pure_path_classification():
    # pure == prefill-only (no decode loop); all decode is step-driven now.
    assert poc_is_pure_path(_params(max_tokens=0, validation=False))
    # validation now runs step-driven (unified), not pure
    assert not poc_is_pure_path(_params(max_tokens=8, validation=True))
    # decode generation -> step-driven (not pure)
    assert not poc_is_pure_path(_params(max_tokens=8, validation=False))


def test_step_num_tokens_mixed_generation():
    pp = _params(max_tokens=8, validation=False, seq_len=256)
    # first step (nothing computed) = prefill of seq_len
    assert poc_step_num_tokens(pp, 0) == 256
    # every later step = a single decode token
    assert poc_step_num_tokens(pp, 256) == 1
    assert poc_step_num_tokens(pp, 300) == 1


def test_step_num_tokens_validation_is_step_driven():
    # validation is unified onto the step-driven path: seq_len prefill, then 1/step.
    val = _params(max_tokens=8, validation=True, seq_len=128)
    assert poc_step_num_tokens(val, 0) == 128
    assert poc_step_num_tokens(val, 128) == 1
    # prefill-only is a single seq_len step.
    prefill_only = _params(max_tokens=0, validation=False, seq_len=128)
    assert poc_step_num_tokens(prefill_only, 0) == 128


def test_alloc_footprint_prefill_only_reserves_full():
    pp = _params(max_tokens=0, validation=False, seq_len=256)
    # prefill-only (pure) runs in one step -> reserve seq_len (+max_tokens=0)
    assert poc_alloc_footprint(pp, num_new_tokens=256) == 256


def test_alloc_footprint_decode_reserves_one_step():
    # both generation and validation are step-driven -> reserve this step's tokens
    for validation in (False, True):
        pp = _params(max_tokens=8, validation=validation, seq_len=256)
        assert poc_alloc_footprint(pp, num_new_tokens=256) == 256
        assert poc_alloc_footprint(pp, num_new_tokens=1) == 1


def test_share_budget_extremes_and_rounding():
    # 0.0 blocks PoC entirely this step; 1.0 grants the whole step budget.
    assert poc_share_budget(0.0, 32768) == 0
    assert poc_share_budget(1.0, 32768) == 32768
    # fractional share is floored (int truncation).
    assert poc_share_budget(0.5, 32768) == 16384
    assert poc_share_budget(0.3, 1000) == 300
    assert poc_share_budget(0.5, 33) == 16

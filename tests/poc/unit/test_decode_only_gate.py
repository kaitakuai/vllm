"""Unit tests for the decode-only mixing gate (scheduler contract) — pure logic, no GPU.

The gate decides, per scheduler step, whether to defer chat or PoC so chat and PoC
share a forward ONLY when BOTH decode (uniform -> CUDA-graphable). The 4 mixed shapes:
  (a) chat-dec  + poc-dec  -> MIX  (no defer)          graphable, allowed
  (b) chat-pref + poc-dec  -> defer PoC  (chat pure)   excluded
  (c) chat-dec  + poc-pref -> defer chat (PoC pure)    excluded
  (d) chat-pref + poc-pref -> defer chat (PoC pure)    excluded (PoC prefill priority)
"""
import pytest

from vllm.poc.mixed_decode import decode_only_mixing_gate as gate, POC_DEFER_LIMIT


def _g(**kw):
    base = dict(mixed_cudagraph=True, poc_decode_pending=False,
                poc_will_prefill=False, chat_will_prefill=False, consecutive_defers=0)
    base.update(kw)
    return gate(**base)


def test_a_both_decode_mixes():
    defer_chat, defer_poc, _ = _g()                       # neither side prefilling
    assert not defer_chat and not defer_poc               # -> MIX (graphable)


def test_b_chat_prefill_defers_poc():
    defer_chat, defer_poc, _ = _g(chat_will_prefill=True)
    assert defer_poc and not defer_chat                   # chat prefill runs pure


def test_c_poc_prefill_defers_chat():
    defer_chat, defer_poc, _ = _g(poc_will_prefill=True)
    assert defer_chat and not defer_poc                   # PoC prefill runs pure


def test_d_both_prefill_poc_priority():
    defer_chat, defer_poc, _ = _g(poc_will_prefill=True, chat_will_prefill=True)
    assert defer_chat and not defer_poc                   # PoC prefill wins


def test_validation_pure_decode_always_defers_chat():
    # poc_decode_pending (validation / flag-off pure decode) -> exclusive PoC batch,
    # independent of the cudagraph flag.
    for flag in (True, False):
        defer_chat, defer_poc, _ = _g(mixed_cudagraph=flag, poc_decode_pending=True,
                                      chat_will_prefill=True)
        assert defer_chat and not defer_poc


@pytest.mark.parametrize("poc_pref,chat_pref",
                         [(False, False), (True, False), (False, True), (True, True)])
def test_flag_off_reduces_to_original(poc_pref, chat_pref):
    # Flag off: defer_poc never set; defer_chat == poc_decode_pending only.
    defer_chat, defer_poc, _ = _g(mixed_cudagraph=False,
                                  poc_will_prefill=poc_pref, chat_will_prefill=chat_pref)
    assert defer_poc is False and defer_chat is False
    defer_chat2, defer_poc2, _ = _g(mixed_cudagraph=False, poc_decode_pending=True,
                                    poc_will_prefill=poc_pref, chat_will_prefill=chat_pref)
    assert defer_chat2 is True and defer_poc2 is False


def test_mutual_exclusivity_all_combinations():
    for pdp in (True, False):
        for pwp in (True, False):
            for cwp in (True, False):
                for flag in (True, False):
                    dc, dp, _ = _g(mixed_cudagraph=flag, poc_decode_pending=pdp,
                                   poc_will_prefill=pwp, chat_will_prefill=cwp)
                    assert not (dc and dp), (pdp, pwp, cwp, flag)


def test_valve_bounds_poc_starvation():
    # Continuous chat prefill: after POC_DEFER_LIMIT defers, PoC gets one exclusive step.
    n, defers = 0, 0
    for _ in range(POC_DEFER_LIMIT + 1):
        defer_chat, defer_poc, n = gate(
            mixed_cudagraph=True, poc_decode_pending=False,
            poc_will_prefill=False, chat_will_prefill=True, consecutive_defers=n)
        defers += int(defer_poc)
    assert defer_chat and not defer_poc          # the (LIMIT+1)-th step flips to PoC-exclusive
    assert n == 0                                # counter reset after the forced step
    assert defers == POC_DEFER_LIMIT             # exactly LIMIT defers before the flip


def test_valve_resets_on_non_defer_step():
    _, defer_poc, n = _g(chat_will_prefill=True, consecutive_defers=3)
    assert defer_poc and n == 4
    _, defer_poc2, n2 = _g(chat_will_prefill=False, consecutive_defers=n)  # mix step
    assert not defer_poc2 and n2 == 0

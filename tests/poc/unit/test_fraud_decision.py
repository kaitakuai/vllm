"""Unit tests for fraud_test — the binomial fraud/honest decision (vllm/poc/data.py).
This is the literal consensus verdict (mismatch count -> fraud bool) yet had no unit
coverage. Pure math: monotonicity, the honest/fraud extremes, the empty case, and the
decision boundary around fraud_threshold.
"""
from vllm.poc.data import fraud_test

P = 0.1          # honest baseline mismatch rate (default p_mismatch)
THR = 0.05       # default fraud_threshold (p-value cutoff)


def test_zero_mismatch_is_honest():
    p_value, fraud = fraud_test(0, 100, p_mismatch=P, fraud_threshold=THR)
    assert fraud is False
    assert p_value > THR


def test_all_mismatch_is_fraud():
    p_value, fraud = fraud_test(100, 100, p_mismatch=P, fraud_threshold=THR)
    assert fraud is True
    assert p_value < THR


def test_empty_is_not_fraud():
    """No samples -> cannot accuse; p_value=1, fraud=False (avoids div-by-zero)."""
    p_value, fraud = fraud_test(0, 0, p_mismatch=P, fraud_threshold=THR)
    assert p_value == 1.0 and fraud is False


def test_at_baseline_rate_is_honest():
    """Mismatch rate ≈ p_mismatch -> consistent with honest -> not fraud."""
    _, fraud = fraud_test(10, 100, p_mismatch=P, fraud_threshold=THR)
    assert fraud is False


def test_pvalue_monotonic_in_mismatch():
    """More mismatches (same n) -> smaller p-value (more suspicious), monotonically."""
    pvals = [fraud_test(k, 100, p_mismatch=P, fraud_threshold=THR)[0]
             for k in range(0, 101, 10)]
    for a, b in zip(pvals, pvals[1:]):
        assert b <= a + 1e-12, f"p-value not monotonically non-increasing: {pvals}"


def test_decision_boundary_crossing():
    """There is a single crossover count: below it honest, at/above it fraud (for
    n=100, p=0.1, thr=0.05). Find it and assert the flip is clean."""
    n = 100
    flips = [fraud_test(k, n, p_mismatch=P, fraud_threshold=THR)[1] for k in range(n + 1)]
    # once True, stays True (no oscillation) — monotone decision
    first_true = flips.index(True)
    assert all(flips[k] for k in range(first_true, n + 1)), \
        f"fraud decision oscillates: {flips}"
    assert not any(flips[k] for k in range(first_true)), \
        f"fraud flagged below the crossover: {flips}"
    # sanity: crossover is above the honest baseline (10) and below all-mismatch
    assert P * n < first_true < n

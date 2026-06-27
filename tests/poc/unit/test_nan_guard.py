"""Non-finite (NaN/Inf) guard for the sphere snap (audit #7 / task #55).

A non-finite hidden is a COMPUTE FAULT (GPU contention / kernel fault / stale
attention metadata), NOT fraud. `argmax(NaN)` silently returns a garbage index
that would count as a mismatch and read as fraud. `snap_with_guard` must instead
return sentinel -1 + a fault mask so the caller can log it and EXCLUDE the step
from the mismatch rate. Pure CPU; no GPU/server.
"""
import torch

from vllm.poc.sphere import (
    SPHERE_DIM, SPHERE_POINTS, get_sphere_codebook, project_to_sphere,
    nearest_sphere_index, snap_with_guard,
)


def _cb():
    return get_sphere_codebook()


def test_finite_queries_match_plain_snap():
    """On finite input the guard is a no-op: same k as nearest_sphere_index, no faults."""
    cb = _cb()
    q = project_to_sphere(torch.randn(128, SPHERE_DIM))
    k, bad = snap_with_guard(q, cb)
    assert not bad.any()
    assert torch.equal(k, nearest_sphere_index(q, cb))
    assert int(k.min()) >= 0 and int(k.max()) < SPHERE_POINTS


def test_nan_and_inf_rows_return_sentinel():
    """Non-finite rows -> k = -1 + bad=True; finite rows unaffected."""
    cb = _cb()
    q = project_to_sphere(torch.randn(6, SPHERE_DIM))
    good = nearest_sphere_index(q, cb).clone()
    q[1] = float("nan")
    q[3, 0] = float("inf")
    q[4] = float("-inf")
    k, bad = snap_with_guard(q, cb)

    faulted = {1, 3, 4}
    assert set(bad.nonzero(as_tuple=True)[0].tolist()) == faulted
    for i in faulted:
        assert int(k[i]) == -1                      # sentinel, never a valid index
    for i in (0, 2, 5):                              # finite rows: unchanged
        assert not bool(bad[i]) and int(k[i]) == int(good[i])


def test_sentinel_is_distinguishable_from_valid_index():
    """-1 is < 0 so callers can exclude faults via (k >= 0); valid k never is."""
    cb = _cb()
    q = project_to_sphere(torch.randn(32, SPHERE_DIM))
    q[0] = float("nan")
    k, _ = snap_with_guard(q, cb)
    valid = k >= 0
    assert not bool(valid[0])
    assert bool(valid[1:].all())
    # the exclusion idiom used in scoring: a fault is neither match nor mismatch
    ref = torch.zeros_like(k)
    mismatch = ((k != ref) & (k >= 0))
    assert not bool(mismatch[0])                     # faulted step excluded

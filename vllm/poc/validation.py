"""PoC artifact validation logic."""
from typing import Dict, List, Optional, Tuple

import numpy as np

from .data import decode_vector, fraud_test, DEFAULT_DIST_THRESHOLD, DEFAULT_P_MISMATCH, DEFAULT_FRAUD_THRESHOLD


def score_vector_channel(
    computed_artifacts: List[Dict],
    ref_vectors: Dict[int, List[str]],
) -> Optional[Dict]:
    """Continuous vector-channel score: per-step cosine distance between the
    prover's pre-snap sphere slices (``sph_values_steps`` from the reference
    artifacts) and the validator's own teacher-forced recompute.

    Rationale: the sphere_k snap keeps ~4 bits/step — a mismatch fires only when
    a deviation crosses a Voronoi boundary, so a subtle fraud (e.g. a close
    quant) sits only a few pp above the honest cross-HW flip floor. The pre-snap
    slices carry the full displacement field: honest cross-config distance is
    ~1e-4 while a quant-fraud sits ~5e-3 on every step (A100 pilot: fraud/floor
    60x, no per-nonce overlap), so the averaged distance separates where the
    flip rate cannot. Consensus is untouched — the k-id chain still seeds the
    next step; the distance is computed by ONE validator, whose threshold
    absorbs cross-HW float drift instead of a bit-exact agreement.

    Scores decode steps only (index 1..N; index 0 = prefill, which has its own
    legacy vector_b64 path). Non-finite slices are skipped like the NaN guard.
    Returns None when no (computed, reference) vector pair exists — e.g. debug
    off on either side — so callers can attach it as optional evidence.
    """
    per_nonce: List[Dict] = []
    for a in computed_artifacts:
        ref_b64 = ref_vectors.get(a["nonce"]) or []
        own_b64 = a.get("sph_values_steps") or []
        n = min(len(ref_b64), len(own_b64))
        if n < 2:      # need at least one decode step beyond the prefill slice
            continue
        dists = []
        n_bad = 0
        for t in range(1, n):
            vp = decode_vector(ref_b64[t])
            vv = decode_vector(own_b64[t])
            if vp.shape != vv.shape or not (
                    np.all(np.isfinite(vp)) and np.all(np.isfinite(vv))):
                n_bad += 1
                continue
            dists.append(1.0 - float(np.dot(vp, vv)))
        if dists:
            per_nonce.append({
                "nonce": a["nonce"],
                "mean_dist": float(np.mean(dists)),
                "n_steps_scored": len(dists),
                "n_bad_steps": n_bad,
            })
    if not per_nonce:
        return None
    return {
        "mean_dist": float(np.mean([e["mean_dist"] for e in per_nonce])),
        "max_nonce_dist": float(max(e["mean_dist"] for e in per_nonce)),
        "n_nonces_scored": len(per_nonce),
        "per_nonce": per_nonce,
    }


def validate_artifacts(
    computed_artifacts: List[Dict],
    validation_map: Dict[int, str],
    dist_threshold: float = DEFAULT_DIST_THRESHOLD,
    k_dim: int = 12,
) -> Tuple[int, List[int]]:
    """Compare computed artifacts against validation artifacts.

    Args:
        computed_artifacts: List of {"nonce": int, "vector_b64": str}
        validation_map: Dict mapping nonce -> vector_b64
        dist_threshold: L2 distance threshold for mismatch
        k_dim: Expected vector dimension

    Returns:
        (n_mismatch, mismatch_nonces)
    """
    n_mismatch = 0
    mismatch_nonces = []

    for artifact in computed_artifacts:
        nonce = artifact["nonce"]
        received_b64 = validation_map.get(nonce)
        if not received_b64:
            continue

        computed_vec = decode_vector(artifact["vector_b64"])
        received_vec = decode_vector(received_b64)

        if received_vec.shape != (k_dim,):
            n_mismatch += 1
            mismatch_nonces.append(nonce)
            continue

        if not np.all(np.isfinite(received_vec)):
            n_mismatch += 1
            mismatch_nonces.append(nonce)
            continue

        distance = float(np.linalg.norm(computed_vec - received_vec))

        if distance > dist_threshold:
            n_mismatch += 1
            mismatch_nonces.append(nonce)
    
    return n_mismatch, mismatch_nonces


def run_validation(
    computed_artifacts: List[Dict],
    validation_map: Dict[int, str],
    n_total: int,
    dist_threshold: float = DEFAULT_DIST_THRESHOLD,
    p_mismatch: float = DEFAULT_P_MISMATCH,
    fraud_threshold: float = DEFAULT_FRAUD_THRESHOLD,
    k_dim: int = 12,
    use_trajectory: bool = False,
    ref_vectors: Optional[Dict[int, List[str]]] = None,
) -> Dict:
    """Run full validation with fraud test. Same response shape for both flows.

    - prefill (use_trajectory=False): vector-L2 per nonce + binomial fraud_test
      (uses p_mismatch + fraud_threshold). Unchanged.
    - decode (use_trajectory=True, max_tokens>0): count sphere_k mismatches over
      all steps; fraud when the mismatch rate exceeds p_mismatch (reused as the max
      allowed fraction). No binomial; p_value carries the rate but the decision
      ignores it.
    - ref_vectors (optional, decode): prover-side sph_values_steps per nonce.
      When both sides carry pre-snap slices, the continuous vector-channel score
      (score_vector_channel) is attached as ``vector_score`` EVIDENCE — the
      verdict stays k-based so the two channels can be A/B'd on the same run.
    """
    per_nonce: List[Dict] = []   # per-nonce evidence: [{nonce, n_sphere_mismatches, n_steps}]
    if use_trajectory:
        n_mismatch = 0
        n_steps = 0
        mismatch_nonces = []
        for a in computed_artifacts:
            traj = a.get("k_points_steps") or []
            if not traj:
                continue
            m = a.get("n_sphere_mismatches", 0) or 0
            if m < 0:  # -1 == generation (no reference); treat as no mismatch
                m = 0
            n_mismatch += m
            n_steps += len(traj)
            per_nonce.append({"nonce": a["nonce"], "n_sphere_mismatches": m, "n_steps": len(traj)})
            if m > 0:
                mismatch_nonces.append(a["nonce"])
        rate = (n_mismatch / n_steps) if n_steps else 0.0
        fraud_detected = rate > p_mismatch   # decision; p_value not used
        p_value = rate                       # kept for shape; ignored by the decision
    else:
        n_mismatch, mismatch_nonces = validate_artifacts(
            computed_artifacts, validation_map, dist_threshold, k_dim
        )
        p_value, fraud_detected = fraud_test(n_mismatch, n_total, p_mismatch, fraud_threshold)

    result = {
        "n_total": n_total,
        "n_mismatch": n_mismatch,
        "mismatch_nonces": mismatch_nonces,
        "per_nonce": per_nonce,
        "p_value": p_value,
        "fraud_detected": fraud_detected,
    }
    if use_trajectory and ref_vectors:
        vector_score = score_vector_channel(computed_artifacts, ref_vectors)
        if vector_score is not None:
            result["vector_score"] = vector_score
    return result

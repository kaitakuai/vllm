"""PoC artifact validation logic."""
from typing import Dict, List, Tuple

import numpy as np

from .data import decode_vector, fraud_test, DEFAULT_DIST_THRESHOLD, DEFAULT_P_MISMATCH, DEFAULT_FRAUD_THRESHOLD


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
) -> Dict:
    """Run full validation with fraud test. Same response shape for both flows.

    - prefill (use_trajectory=False): vector-L2 per nonce + binomial fraud_test
      (uses p_mismatch + fraud_threshold). Unchanged.
    - decode (use_trajectory=True, max_tokens>0): count sphere_k mismatches over
      all steps; fraud when the mismatch rate exceeds p_mismatch (reused as the max
      allowed fraction). No binomial; p_value carries the rate but the decision
      ignores it.
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

    return {
        "n_total": n_total,
        "n_mismatch": n_mismatch,
        "mismatch_nonces": mismatch_nonces,
        "per_nonce": per_nonce,
        "p_value": p_value,
        "fraud_detected": fraud_detected,
    }

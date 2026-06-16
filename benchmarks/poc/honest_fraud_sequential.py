#!/usr/bin/env python3
"""Single-GPU honest/fraud PoC separation — one model loaded at a time.

The canonical flow (tests_poc/scripts/collect_validation.sh on bs/poc-context-fix)
needs an inference server AND a validation server up at once (two machines/IPs).
On a single GPU two 7B models don't co-fit, so this splits inference and
validation across time:

  Phase 1 (fraud model up):   --mode infer  --url <fraud>   --out fraud_traj.json
  Phase 2 (honest model up):  --mode infer  --url <honest>  --out honest_traj.json
                              --mode validate --url <honest> --traj honest_traj.json --label HONEST
                              --mode validate --url <honest> --traj fraud_traj.json  --label FRAUD

Validation seeds the server with the saved inference sphere_k trajectory
(``inference_k_points_steps``) so steps are ALIGNED (a disagreement does not
cascade); the server returns ``n_sphere_mismatches`` = #steps where this server's
sphere_k differs from the seeded one. Honest (same model) ~ 0; fraud (different
model) is high. See wiki: testing/honest-fraud-separation.

Reuses the request helpers from ``poc_validation.py`` (same dir).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from poc_validation import (  # noqa: E402
    send_inference_request, send_validation_request, extract_artifact,
    get_server_model, wait_for_server,
)

DEFAULT_BLOCK_HASH = "deadbeef" * 8
DEFAULT_PUBLIC_KEY = "cafebabe" * 8


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["infer", "validate"], required=True)
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", help="infer: where to save the trajectory JSON")
    ap.add_argument("--traj", help="validate: trajectory JSON to validate against")
    ap.add_argument("--label", default="", help="validate: label for the report")
    ap.add_argument("--nonces", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--block-hash", default=DEFAULT_BLOCK_HASH)
    ap.add_argument("--public-key", default=DEFAULT_PUBLIC_KEY)
    a = ap.parse_args()
    nonces = list(range(a.nonces))

    wait_for_server(a.url)
    model = get_server_model(a.url)
    print(f"server {a.url} model={model}")

    if a.mode == "infer":
        assert a.out, "--out required for --mode infer"
        resp = send_inference_request(a.url, a.block_hash, a.public_key, nonces,
                                      a.seq_len, a.max_tokens, model)
        traj = extract_artifact(resp, type="inference")  # {nonce: [sphere_k...]}
        json.dump({str(k): v for k, v in traj.items()}, open(a.out, "w"))
        any_len = len(next(iter(traj.values()))) if traj else 0
        print(f"saved {len(traj)} trajectories (len {any_len}) -> {a.out}")
        return

    assert a.traj, "--traj required for --mode validate"
    inf_steps = {int(k): v for k, v in json.load(open(a.traj)).items()}
    resp = send_validation_request(a.url, a.block_hash, a.public_key, nonces,
                                   a.seq_len, a.max_tokens, inf_steps, model)
    val = extract_artifact(resp, type="validation")
    per = a.max_tokens + 1
    print(f"\n=== {a.label or 'VALIDATE'} (validated on {model}) ===")
    tot = 0
    for n in nonces:
        m = val.get(n, {}).get("n_sphere_mismatches", -1)
        tot += m if m > 0 else 0
        print(f"  nonce {n}: n_sphere_mismatches = {m} / {per}")
    denom = len(nonces) * per
    print(f"  TOTAL mismatches = {tot} / {denom}  ({100 * tot / denom:.1f}%)")


if __name__ == "__main__":
    main()

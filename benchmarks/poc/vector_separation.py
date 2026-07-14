#!/usr/bin/env python3
"""vector_separation.py — OFFLINE vector-channel separation over collected runs.

Takes GEN:VAL file pairs produced by collect.py **--debug** (the gen file carries
the prover's per-step pre-snap sphere slices, the val file the validator's own
teacher-forced recompute) and scores the continuous channel next to the discrete
one, on the SAME trajectories:

  per step   d_t = 1 - <q_prover, q_validator>      (decode steps 1..N)
  per nonce  D_i = mean_t d_t
  per pair   mean/sd/min/max of D_i  +  the discrete k-mismatch rate for contrast

Honest/fraud grouping comes from the val file's results.honest flag (same rule as
analyze.py: validator_model == prover_model). Across groups it reports AUC, d',
the worst-honest vs weakest-fraud gap, and K — the number of nonces at which a
mean-of-K test separates at FPR 1e-4 / power 95% (gaussian approximation).

  vector_separation.py gen_honest.json:val_honest.json gen_fraud.json:val_fraud.json ...

No server, no GPU — pure file analysis (numpy).
"""
import argparse
import base64
import json
import math
import sys

import numpy as np


def _slices(artifact):
    sv = artifact.get("sph_values_steps") or []
    if not sv:
        return None
    return np.stack([
        np.frombuffer(base64.b64decode(s), dtype="<f2").astype(np.float32)
        for s in sv])


def pair_dists(gen_file, val_file):
    """Per-nonce mean cosine distance over decode steps (index 1..)."""
    gen = json.load(open(gen_file))
    val = json.load(open(val_file))
    g = {a["nonce"]: a for a in gen.get("artifacts", [])}
    v = {a["nonce"]: a for a in val.get("artifacts", [])}
    out = []
    for nonce in sorted(set(g) & set(v)):
        qp, qv = _slices(g[nonce]), _slices(v[nonce])
        if qp is None or qv is None:
            continue
        n = min(len(qp), len(qv))
        if n < 2:
            continue
        cos = np.sum(qp[1:n] * qv[1:n], axis=1)
        ok = np.isfinite(cos)
        if ok.any():
            out.append(float(np.mean(1.0 - cos[ok])))
    honest = bool(val.get("results", {}).get("honest", True))
    rate = val.get("results", {}).get("rate")
    return np.array(out), honest, rate


def auc(honest, fraud):
    wins = sum((1.0 if f > h else 0.5 if f == h else 0.0)
               for h in honest for f in fraud)
    return wins / (len(honest) * len(fraud))


def k_to_separate(honest, fraud, z=2.8 + 1.645):
    """Nonces needed so a mean-of-K test separates at FPR 1e-4 / power 95%."""
    mh, sh = honest.mean(), honest.std(ddof=1)
    mf, sf = fraud.mean(), fraud.std(ddof=1)
    if mf <= mh:
        return math.inf
    return max(1, math.ceil(((z * math.sqrt((sh**2 + sf**2) / 2)) / (mf - mh))**2))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pairs", nargs="+", metavar="GEN:VAL",
                    help="gen/val file pair, colon-separated (both from collect.py --debug)")
    a = ap.parse_args()

    rows, hon_all, fr_all = [], [], []
    for spec in a.pairs:
        try:
            gen_file, val_file = spec.split(":", 1)
        except ValueError:
            sys.exit(f"bad pair (want GEN:VAL): {spec}")
        d, honest, rate = pair_dists(gen_file, val_file)
        if not len(d):
            print(f"SKIP {spec}: no sph_values_steps on both sides "
                  f"(generate AND validate need --debug)")
            continue
        rows.append((gen_file, honest, rate, d))
        (hon_all if honest else fr_all).append(d)

    print(f"{'pair (gen)':44s} {'class':7s} {'k-rate':>8s} "
          f"{'vec mean':>10s} {'sd':>9s} {'min..max':>21s}")
    for gen_file, honest, rate, d in rows:
        print(f"{gen_file[-44:]:44s} {'honest' if honest else 'FRAUD':7s} "
              f"{(f'{rate*100:.2f}%' if rate is not None else '—'):>8s} "
              f"{d.mean():>10.2e} {d.std(ddof=1):>9.1e} "
              f"{d.min():>10.2e}..{d.max():<9.2e}")

    if hon_all and fr_all:
        h, f = np.concatenate(hon_all), np.concatenate(fr_all)
        gap = f.min() / h.max() if h.max() > 0 else math.inf
        print(f"\nhonest n={len(h)}  fraud n={len(f)}")
        print(f"AUC={auc(h, f):.4f}   worst-honest={h.max():.2e}  "
              f"weakest-fraud={f.min():.2e}  gap={gap:.1f}x"
              f"{'  (NO overlap)' if gap > 1 else '  (OVERLAP)'}")
        print(f"K (nonces to separate @ FPR 1e-4 / power .95) = {k_to_separate(h, f)}")


if __name__ == "__main__":
    main()

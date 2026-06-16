"""Cross-validate sphere_k assignment across two vLLM PoC servers.

The server now computes the nearest equidistant codebook point internally
and returns ``sphere_k`` directly in each artifact.  This script simply
sends the same (block_hash, nonce) pairs to a PRIMARY and a VALIDATION
server, reads ``sphere_k`` from both responses, and checks that they agree.

No vector decoding or client-side geometry is needed.

Usage
-----
python sphere_cross_validation_v2.py --num-hashes 2 --nonces 0:16
python sphere_cross_validation_v2.py --block-hashes abc def --nonces 0 1 2 --batch-size 4
python sphere_cross_validation_v2.py --primary http://host-a:8001 --validation http://host-b:8001
"""

import argparse
import datetime
import hashlib
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import requests

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
# 65.108.33.83
PRIMARY_URL        = "http://65.108.33.83:8001"
PRIMARY_MODEL      = "RedHatAI/Qwen2.5-7B-Instruct-quantized.w8a16"

VALIDATION_URL     = "http://0.0.0.0:8001"
VALIDATION_MODEL   = "RedHatAI/Qwen2.5-7B-Instruct-quantized.w8a16"

PUBLIC_KEY   = "default_public_key"
SEQ_LEN      = 256
K_DIM        = 12
BATCH_SIZE   = 8
TIMEOUT_SEC  = 300
OUTPUT_DIR   = "results"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_block_hashes(n: int) -> List[str]:
    return [hashlib.sha256(os.urandom(32)).hexdigest() for _ in range(n)]


def parse_nonces(specs: List[str]) -> List[int]:
    """Parse nonce specs: plain ints, 'start:end', or 'start:end:step'."""
    nonces: List[int] = []
    for token in specs:
        if ":" in token:
            parts = [int(p) for p in token.split(":")]
            if len(parts) == 2:
                nonces.extend(range(parts[0], parts[1]))
            elif len(parts) == 3:
                nonces.extend(range(parts[0], parts[1], parts[2]))
            else:
                raise ValueError(f"Invalid nonce range: {token!r}")
        else:
            nonces.append(int(token))
    return nonces


def send_batch(
    url: str,
    model: str,
    block_hash: str,
    nonces: List[int],
) -> dict:
    endpoint = f"{url.rstrip('/')}/api/v1/pow/generate"
    payload = {
        "block_hash":   block_hash,
        "block_height": 0,
        "public_key":   PUBLIC_KEY,
        "node_id":      0,
        "node_count":   1,
        "nonces":       nonces,
        "params":       {"model": model, "seq_len": SEQ_LEN, "k_dim": K_DIM},
        "batch_size":   len(nonces),
        "wait":         True,
    }
    resp = requests.post(endpoint, json=payload, timeout=TIMEOUT_SEC)
    resp.raise_for_status()
    return resp.json()


def fetch_both(
    block_hash: str,
    nonces: List[int],
    primary_url: str,
    primary_model: str,
    validation_url: str,
    validation_model: str,
) -> Tuple[Optional[dict], Optional[dict], Optional[str], Optional[str]]:
    """Send the same batch to both servers concurrently."""
    primary_resp = validation_resp = None
    primary_err  = validation_err  = None

    def _fetch(label, url, model):
        return label, send_batch(url, model, block_hash, nonces)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            pool.submit(_fetch, "primary",    primary_url,    primary_model):    "primary",
            pool.submit(_fetch, "validation", validation_url, validation_model): "validation",
        }
        for fut in as_completed(futures):
            try:
                label, resp = fut.result()
                if label == "primary":
                    primary_resp = resp
                else:
                    validation_resp = resp
            except Exception as exc:
                label = futures[fut]
                if label == "primary":
                    primary_err = str(exc)
                else:
                    validation_err = str(exc)

    return primary_resp, validation_resp, primary_err, validation_err


def parse_artifacts(resp: dict) -> Dict[int, int]:
    """Extract {nonce: prefill_k} from a server response.

    The prefill k is k_points_steps[0] (the scalar sphere_k field was dropped).
    """
    items = resp if isinstance(resp, list) else resp.get("artifacts", resp.get("results", []))
    out: Dict[int, int] = {}
    for item in items:
        nonce = item.get("nonce")
        steps = item.get("k_points_steps") or []
        if nonce is None:
            continue
        if not steps:
            print(f"  WARNING: nonce={nonce} has no k_points_steps",
                  file=sys.stderr)
        out[nonce] = steps[0] if steps else -1
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-validate sphere_k across two vLLM PoC servers.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    hash_group = parser.add_mutually_exclusive_group(required=True)
    hash_group.add_argument("--num-hashes", type=int, metavar="N",
                            help="Generate N random block hashes")
    hash_group.add_argument("--block-hashes", nargs="+", metavar="HASH",
                            help="Explicit block hash(es)")

    parser.add_argument("--nonces", nargs="+", default=["0:16"], metavar="SPEC",
                        help='Nonce specs: ints, "start:end", "start:end:step" (default: 0:16)')
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Nonces per HTTP request (default: {BATCH_SIZE})")
    parser.add_argument("--primary",          default=PRIMARY_URL,        metavar="URL")
    parser.add_argument("--primary-model",    default=PRIMARY_MODEL,      metavar="MODEL")
    parser.add_argument("--validation",       default=VALIDATION_URL,     metavar="URL")
    parser.add_argument("--validation-model", default=VALIDATION_MODEL,   metavar="MODEL")
    parser.add_argument("--num-requests", type=int, default=1, metavar="N",
                        help="Repeat each (hash, nonce) batch N times and check "
                             "consistency within each server as well as across them "
                             "(default: 1)")
    parser.add_argument("--output", default=None,
                        help="Output JSON file (default: auto-generated in results/)")
    args = parser.parse_args()

    # ── block hashes ──────────────────────────────────────────────────────
    if args.num_hashes is not None:
        block_hashes = generate_block_hashes(args.num_hashes)
        print(f"Generated {args.num_hashes} random block hashes:")
        for h in block_hashes:
            print(f"  {h}")
    else:
        block_hashes = args.block_hashes

    nonces = parse_nonces(args.nonces)
    if not nonces:
        print("No nonces specified – nothing to do.", file=sys.stderr)
        sys.exit(1)

    batches = [nonces[i : i + args.batch_size]
               for i in range(0, len(nonces), args.batch_size)]

    if args.output:
        out_path = args.output
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, f"sphere_xval_v2_{ts}.json")

    print(f"\nPrimary    : {args.primary}  model={args.primary_model}")
    print(f"Validation : {args.validation}  model={args.validation_model}")
    print(f"Hashes     : {len(block_hashes)}")
    print(f"Nonces     : {len(nonces)}  ({len(batches)} batches × <={args.batch_size})")
    print(f"Requests   : {args.num_requests}x per (hash, batch)")
    print(f"Output     : {out_path}\n")

    all_records: List[dict] = []
    n_match = n_mismatch = n_error = 0
    # consistency counters: same server, repeated requests
    n_p_consistent = n_p_inconsistent = 0
    n_v_consistent = n_v_inconsistent = 0

    for block_hash in block_hashes:
        print(f"block_hash={block_hash[:12]}...")
        for batch_idx, batch in enumerate(batches):
            tag = f"  batch {batch_idx + 1}/{len(batches)} nonces={batch[0]}..{batch[-1]}"

            # Collect sphere_k for each repetition: {nonce: [k_req0, k_req1, ...]}
            p_runs: Dict[int, List[int]] = {n: [] for n in batch}
            v_runs: Dict[int, List[int]] = {n: [] for n in batch}
            had_error = False

            for req_idx in range(args.num_requests):
                p_resp, v_resp, p_err, v_err = fetch_both(
                    block_hash, batch,
                    args.primary,    args.primary_model,
                    args.validation, args.validation_model,
                )

                if p_err or v_err:
                    if p_err:
                        print(f"{tag} req={req_idx+1}  PRIMARY ERROR: {p_err}",
                              file=sys.stderr)
                    if v_err:
                        print(f"{tag} req={req_idx+1}  VALIDATION ERROR: {v_err}",
                              file=sys.stderr)
                    n_error += len(batch)
                    had_error = True
                    continue

                p_arts = parse_artifacts(p_resp)
                v_arts = parse_artifacts(v_resp)
                for nonce in batch:
                    if nonce in p_arts:
                        p_runs[nonce].append(p_arts[nonce])
                    if nonce in v_arts:
                        v_runs[nonce].append(v_arts[nonce])

            if had_error and args.num_requests == 1:
                continue

            batch_match = batch_mismatch = 0
            for nonce in batch:
                p_vals = p_runs[nonce]
                v_vals = v_runs[nonce]

                if not p_vals or not v_vals:
                    missing = "primary" if not p_vals else "validation"
                    print(f"  WARNING: nonce={nonce} missing in {missing} response",
                          file=sys.stderr)
                    n_error += 1
                    continue

                # Cross-server comparison: use first successful request
                p_k = p_vals[0]
                v_k = v_vals[0]
                cross_match = (p_k == v_k)

                # Within-server consistency across repetitions
                p_consistent = len(set(p_vals)) == 1
                v_consistent = len(set(v_vals)) == 1

                if p_consistent:
                    n_p_consistent += 1
                else:
                    n_p_inconsistent += 1
                if v_consistent:
                    n_v_consistent += 1
                else:
                    n_v_inconsistent += 1

                record = {
                    "block_hash":            block_hash,
                    "nonce":                 nonce,
                    "primary_sphere_k":      p_k,
                    "validation_sphere_k":   v_k,
                    "match":                 cross_match,
                    "primary_all_k":         p_vals,
                    "validation_all_k":      v_vals,
                    "primary_consistent":    p_consistent,
                    "validation_consistent": v_consistent,
                }
                all_records.append(record)

                if cross_match:
                    n_match      += 1
                    batch_match  += 1
                else:
                    n_mismatch     += 1
                    batch_mismatch += 1

            status = "OK" if batch_mismatch == 0 else f"MISMATCH x{batch_mismatch}"
            extra = ""
            if args.num_requests > 1:
                p_inc = sum(1 for n in batch if not all(
                    v == p_runs[n][0] for v in p_runs[n]) if p_runs[n])
                v_inc = sum(1 for n in batch if not all(
                    v == v_runs[n][0] for v in v_runs[n]) if v_runs[n])
                extra = f"  inconsistent: P={p_inc} V={v_inc}"
            print(f"{tag}  -> {status}  (match={batch_match}, mismatch={batch_mismatch}){extra}")

    # ── output document ───────────────────────────────────────────────────
    n_total    = n_match + n_mismatch
    match_rate = n_match / n_total if n_total > 0 else 0.0
    mismatches = [r for r in all_records if not r["match"]]

    output_doc = {
        "run_timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "config": {
            "primary_url":        args.primary,
            "primary_model":      args.primary_model,
            "validation_url":     args.validation,
            "validation_model":   args.validation_model,
            "k_dim":              K_DIM,
            "seq_len":            SEQ_LEN,
            "batch_size":         args.batch_size,
            "num_requests":       args.num_requests,
            "block_hashes":       block_hashes,
            "nonces":             nonces,
        },
        "stats": {
            "total":                    n_total,
            "match":                    n_match,
            "mismatch":                 n_mismatch,
            "error":                    n_error,
            "match_rate":               round(match_rate, 6),
            "primary_consistent":       n_p_consistent,
            "primary_inconsistent":     n_p_inconsistent,
            "validation_consistent":    n_v_consistent,
            "validation_inconsistent":  n_v_inconsistent,
        },
        "mismatches": mismatches,
        "records":    all_records,
    }

    with open(out_path, "w") as fh:
        json.dump(output_doc, fh, indent=2)

    # ── summary ───────────────────────────────────────────────────────────
    bar_w  = 40
    filled = int(bar_w * match_rate)
    bar    = "#" * filled + "-" * (bar_w - filled)

    print("\n" + "=" * 60)
    print("SPHERE CROSS-VALIDATION SUMMARY  (v2 — server-side sphere_k)")
    print("=" * 60)
    print(f"  Total compared  : {n_total}")
    print(f"  Match (P vs V)  : {n_match}  ({match_rate * 100:.2f} %)")
    print(f"  Mismatch        : {n_mismatch}  ({(1 - match_rate) * 100:.2f} %)")
    print(f"  Errors / skipped: {n_error}")
    print(f"  [{bar}]  {match_rate * 100:.1f} %")
    if args.num_requests > 1:
        print(f"\n  Intra-server consistency  ({args.num_requests} requests per pair):")
        p_tot = n_p_consistent + n_p_inconsistent
        v_tot = n_v_consistent + n_v_inconsistent
        p_rate = n_p_consistent / p_tot if p_tot else 0.0
        v_rate = n_v_consistent / v_tot if v_tot else 0.0
        print(f"  Primary    consistent: {n_p_consistent}/{p_tot}  ({p_rate*100:.2f} %)")
        print(f"  Validation consistent: {n_v_consistent}/{v_tot}  ({v_rate*100:.2f} %)")
    print(f"\n  Output          : {out_path}")

    if mismatches:
        print(f"\nMISMATCHED RECORDS ({len(mismatches)}):")
        hdr = f"  {'hash':12s}  {'nonce':>6s}  {'primary_k':>9s}  {'valid_k':>7s}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for r in mismatches:
            print(
                f"  {r['block_hash'][:12]}  "
                f"{r['nonce']:>6d}  "
                f"{r['primary_sphere_k']:>9d}  "
                f"{r['validation_sphere_k']:>7d}"
            )

        if len(block_hashes) > 1:
            per_hash: Dict[str, int] = defaultdict(int)
            for r in mismatches:
                per_hash[r["block_hash"]] += 1
            print("\n  Mismatches per block_hash:")
            for h, cnt in sorted(per_hash.items(), key=lambda x: -x[1]):
                print(f"    {h[:16]}...  {cnt}")

    print("=" * 60)
    sys.exit(0 if n_mismatch == 0 else 1)


if __name__ == "__main__":
    main()

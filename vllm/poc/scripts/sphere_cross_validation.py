"""Cross-validate sphere k-point assignment across two vLLM PoC servers.

For every (block_hash, nonce) pair the same request is sent to a PRIMARY and a
VALIDATION server concurrently.  The `reduced_hidden_state_b64` (3-D FP16
vector) returned by each server is:

    1. decoded from base64 FP16
    2. L2-normalised onto the unit sphere
    3. matched to the closest of the k=12 icosahedron vertices

The two closest-k-point indices are compared.  Results are written to a JSON
file and a summary table is printed at the end.

Usage
-----
python sphere_cross_validation.py                          # all defaults
python sphere_cross_validation.py --num-hashes 2 --nonces 0:64
python sphere_cross_validation.py --block-hashes abc def --nonces 0 1 2 --batch-size 4
"""

import argparse
import base64
import datetime
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests

PRIMARY_URL = "http://65.108.33.83:8001"
PRIMARY_MODEL = "RedHatAI/Qwen2.5-7B-Instruct-quantized.w8a16"

VALIDATION_URL = "http://65.108.33.83:8003"
VALIDATION_MODEL = "RedHatAI/Qwen2.5-7B-Instruct-quantized.w8a16"

PUBLIC_KEY = "default_public_key"
SEQ_LEN = 256
K_DIM = 12      
BATCH_SIZE = 8 
TIMEOUT_SEC = 300

OUTPUT_DIR  = "results"

def _build_sphere_points(k: int = 12) -> np.ndarray:
    """Return the 12 unit-sphere vertices of a regular icosahedron."""
    if k != 12:
        raise ValueError("Only k=12 is currently supported.")
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    raw = np.array([
        [ 0,  1,  phi], [ 0,  1, -phi],
        [ 0, -1,  phi], [ 0, -1, -phi],
        [ 1,  phi,  0], [ 1, -phi,  0],
        [-1,  phi,  0], [-1, -phi,  0],
        [ phi,  0,  1], [ phi,  0, -1],
        [-phi,  0,  1], [-phi,  0, -1],
    ], dtype=np.float64)
    return raw / np.linalg.norm(raw[0])

SPHERE_POINTS = _build_sphere_points(K_DIM)   # shape [12, 3]

def decode_vector(b64: str) -> np.ndarray:
    """Decode base64 FP16 little-endian bytes to float64 numpy array."""
    return np.frombuffer(base64.b64decode(b64), dtype="<f2").astype(np.float64)


def project_to_sphere(vec: np.ndarray) -> np.ndarray:
    """L2-normalise a 3-D vector onto the unit sphere."""
    n = np.linalg.norm(vec)
    if n < 1e-12:
        raise ValueError(f"Near-zero vector cannot be projected: {vec}")
    return vec / n


def closest_k_point(unit_vec: np.ndarray) -> int:
    """Return the index of the closest icosahedron vertex (cosine similarity)."""
    sims = SPHERE_POINTS @ unit_vec
    return int(np.argmax(sims))


def arc_distance_deg(unit_a: np.ndarray, unit_b: np.ndarray) -> float:
    """Great-circle distance in degrees between two unit vectors."""
    return float(np.degrees(np.arccos(np.clip(unit_a @ unit_b, -1.0, 1.0))))


def generate_block_hashes(n: int) -> List[str]:
    return [hashlib.sha256(os.urandom(32)).hexdigest() for _ in range(n)]


def parse_nonces(specs: List[str]) -> List[int]:
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
) -> Tuple[Optional[dict], Optional[dict], Optional[str], Optional[str]]:
    """Send the same batch to both servers concurrently.

    Returns (primary_resp, validation_resp, primary_err, validation_err).
    """
    primary_resp = validation_resp = None
    primary_err  = validation_err  = None

    def _fetch(label, url, model):
        return label, send_batch(url, model, block_hash, nonces)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            pool.submit(_fetch, "primary",    PRIMARY_URL,    PRIMARY_MODEL):    "primary",
            pool.submit(_fetch, "validation", VALIDATION_URL, VALIDATION_MODEL): "validation",
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


def parse_artifacts(resp: dict) -> Dict[int, dict]:
    """Extract nonce → {reduced_vec, unit_vec, k_point} from a server response."""
    items = resp if isinstance(resp, list) else resp.get("artifacts", resp.get("results", []))
    out: Dict[int, dict] = {}
    for item in items:
        nonce = item.get("nonce")
        rhs_b64 = item.get("reduced_hidden_state_b64")
        if nonce is None or rhs_b64 is None:
            continue
        try:
            raw_vec = decode_vector(rhs_b64)
            unit_vec = project_to_sphere(raw_vec)
            k_pt = closest_k_point(unit_vec)
        except Exception as exc:
            print(f"  WARNING: nonce={nonce} projection failed: {exc}", file=sys.stderr)
            continue
        out[nonce] = {
            "raw_vec": raw_vec.tolist(),
            "unit_vec": unit_vec.tolist(),
            "k_point": k_pt,
        }
    return out

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-validate sphere k-point assignment across two vLLM PoC servers.",
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
    parser.add_argument("--output", default=None,
                        help="Output JSON file (default: auto-generated in results/)")
    args = parser.parse_args()

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

    batches = [nonces[i : i + args.batch_size] for i in range(0, len(nonces), args.batch_size)]

    if args.output:
        out_path = args.output
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, f"sphere_xval_{ts}.json")

    print(f"\nPrimary server    : {PRIMARY_URL}  model={PRIMARY_MODEL}")
    print(f"Validation server : {VALIDATION_URL}  model={VALIDATION_MODEL}")
    print(f"Block hashes      : {len(block_hashes)}")
    print(f"Nonces            : {len(nonces)}  ({len(batches)} batches x <={args.batch_size})")
    print(f"k-points          : {K_DIM} (icosahedron)")
    print(f"Output            : {out_path}\n")
    
    all_records: List[dict] = []
    n_match = n_mismatch = n_error = 0

    for block_hash in block_hashes:
        print(f"block_hash={block_hash[:12]}...")
        for batch_idx, batch in enumerate(batches):
            tag = f"  batch {batch_idx+1}/{len(batches)} nonces={batch[0]}..{batch[-1]}"

            p_resp, v_resp, p_err, v_err = fetch_both(block_hash, batch)

            if p_err or v_err:
                if p_err:
                    print(f"{tag}  PRIMARY ERROR: {p_err}", file=sys.stderr)
                if v_err:
                    print(f"{tag}  VALIDATION ERROR: {v_err}", file=sys.stderr)
                n_error += len(batch)
                continue

            p_arts = parse_artifacts(p_resp)
            v_arts = parse_artifacts(v_resp)

            batch_match = batch_mismatch = 0
            for nonce in batch:
                p_item = p_arts.get(nonce)
                v_item = v_arts.get(nonce)

                if p_item is None or v_item is None:
                    print(f"  WARNING: nonce={nonce} missing in "
                          f"{'primary' if p_item is None else 'validation'} response",
                          file=sys.stderr)
                    n_error += 1
                    continue

                match = p_item["k_point"] == v_item["k_point"]
                arc   = arc_distance_deg(
                    np.array(p_item["unit_vec"]),
                    np.array(v_item["unit_vec"]),
                )

                record = {
                    "block_hash":        block_hash,
                    "nonce":             nonce,
                    "primary_k_point":   p_item["k_point"],
                    "validation_k_point": v_item["k_point"],
                    "primary_unit_vec":  p_item["unit_vec"],
                    "validation_unit_vec": v_item["unit_vec"],
                    "arc_distance_deg":  round(arc, 4),
                    "match":             match,
                }
                all_records.append(record)

                if match:
                    n_match += 1
                    batch_match += 1
                else:
                    n_mismatch += 1
                    batch_mismatch += 1

            status = "OK" if batch_mismatch == 0 else f"MISMATCH x{batch_mismatch}"
            print(f"{tag}  -> {status}  (match={batch_match}, mismatch={batch_mismatch})")

    # ── build output document ─────────────────────────────────────────────────
    n_total = n_match + n_mismatch
    match_rate = n_match / n_total if n_total > 0 else 0.0

    mismatches = [r for r in all_records if not r["match"]]

    output_doc = {
        "run_timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "config": {
            "primary_url":      PRIMARY_URL,
            "primary_model":    PRIMARY_MODEL,
            "validation_url":   VALIDATION_URL,
            "validation_model": VALIDATION_MODEL,
            "k_dim":            K_DIM,
            "seq_len":          SEQ_LEN,
            "batch_size":       args.batch_size,
            "block_hashes":     block_hashes,
            "nonces":           nonces,
        },
        "stats": {
            "total":       n_total,
            "match":       n_match,
            "mismatch":    n_mismatch,
            "error":       n_error,
            "match_rate":  round(match_rate, 6),
        },
        "mismatches": mismatches,
        "records":    all_records,
    }

    with open(out_path, "w") as fh:
        json.dump(output_doc, fh, indent=2)

    # ── print summary ─────────────────────────────────────────────────────────
    bar_w = 40
    filled = int(bar_w * match_rate)
    bar = "#" * filled + "-" * (bar_w - filled)

    print("\n" + "=" * 60)
    print("SPHERE CROSS-VALIDATION SUMMARY")
    print("=" * 60)
    print(f"  Total compared  : {n_total}")
    print(f"  Match           : {n_match}  ({match_rate*100:.2f} %)")
    print(f"  Mismatch        : {n_mismatch}  ({(1-match_rate)*100:.2f} %)")
    print(f"  Errors / skipped: {n_error}")
    print(f"  [{bar}]  {match_rate*100:.1f} %")
    print(f"  Output          : {out_path}")

    if mismatches:
        print(f"\nMISMATCHED RECORDS ({len(mismatches)}):")
        hdr = f"  {'hash':12s}  {'nonce':>6s}  {'primary_k':>9s}  {'valid_k':>7s}  {'arc_deg':>8s}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for r in mismatches:
            print(
                f"  {r['block_hash'][:12]}  "
                f"{r['nonce']:>6d}  "
                f"{r['primary_k_point']:>9d}  "
                f"{r['validation_k_point']:>7d}  "
                f"{r['arc_distance_deg']:>8.2f}"
            )

        # per-hash breakdown
        from collections import defaultdict
        per_hash: Dict[str, int] = defaultdict(int)
        for r in mismatches:
            per_hash[r["block_hash"]] += 1
        if len(per_hash) > 1:
            print("\n  Mismatches per block_hash:")
            for h, cnt in sorted(per_hash.items(), key=lambda x: -x[1]):
                print(f"    {h[:16]}...  {cnt}")

    print("=" * 60)


if __name__ == "__main__":
    main()

"""
Usage examples
--------------
# Auto-generate 4 random block hashes, nonces 0-255, batches of 8
python collect_hidden_states.py \\
    --url http://localhost:8000 \\
    --num-hashes 4 \\
    --nonces 0:256 \\
    --output hidden_states.npz

# Explicit hashes, nonces 0-99
python collect_hidden_states.py \\
    --url http://localhost:8000 \\
    --block-hashes abc123 def456 \\
    --nonces 0:100 \\
    --output hidden_states.npz

# Explicit nonce list
python collect_hidden_states.py \\
    --url http://localhost:8000 \\
    --num-hashes 2 \\
    --nonces 0 5 10 42 99 \\
    --output hidden_states.npz
"""

import argparse
import base64
import hashlib
import os
import sys
from typing import List

import numpy as np
import requests


def generate_block_hashes(n: int) -> List[str]:
    """Generate n random 32-byte hex block hashes (SHA-256 of random bytes)."""
    return [hashlib.sha256(os.urandom(32)).hexdigest() for _ in range(n)]


def decode_vector(b64: str) -> np.ndarray:
    """Decode base64 FP16 little-endian → float32 numpy array."""
    return np.frombuffer(base64.b64decode(b64), dtype="<f2").astype(np.float32)


def parse_nonces(nonce_args: List[str]) -> List[int]:
    """Parse nonce specs: plain ints, 'start:end', or 'start:end:step'."""
    nonces = []
    for token in nonce_args:
        if ":" in token:
            parts = [int(p) for p in token.split(":")]
            if len(parts) == 2:
                nonces.extend(range(parts[0], parts[1]))
            elif len(parts) == 3:
                nonces.extend(range(parts[0], parts[1], parts[2]))
            else:
                raise ValueError(f"Invalid nonce range token: {token!r}")
        else:
            nonces.append(int(token))
    return nonces


def send_batch(
    url: str,
    block_hash: str,
    public_key: str,
    nonces: List[int],
    model: str,
    seq_len: int,
    k_dim: int,
    timeout: int,
) -> dict:
    endpoint = f"{url.rstrip('/')}/api/v1/pow/generate"
    payload = {
        "block_hash":   block_hash,
        "block_height": 0,
        "public_key":   public_key,
        "node_id":      0,
        "node_count":   1,
        "nonces":       nonces,
        "params":       {"model": model, "seq_len": seq_len, "k_dim": k_dim},
        "batch_size":   len(nonces),
        "wait":         True,
    }
    resp = requests.post(endpoint, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def save_npz(
    path: str,
    block_hashes: List[str],
    nonces: List[int],
    hidden_states: List[np.ndarray],
    reduced_hidden_states: List[np.ndarray],
) -> None:
    """Write accumulated data to a .npz file (overwrites if exists)."""
    np.savez(
        path,
        block_hashes=np.array(block_hashes, dtype=object),
        nonces=np.array(nonces, dtype=np.int64),
        hidden_states=np.stack(hidden_states).astype(np.float32),
        reduced_hidden_states=np.stack(reduced_hidden_states).astype(np.float32),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect hidden states from a vLLM PoC server into a .npz archive.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--url",        required=True,
                        help="Server base URL, e.g. http://localhost:8000")
    parser.add_argument("--public-key", default="default_public_key",
                        help="Public key string (default: 'default_public_key')")

    hash_group = parser.add_mutually_exclusive_group(required=True)
    hash_group.add_argument("--num-hashes", type=int, metavar="N",
                            help="Generate N random SHA-256 block hashes automatically")
    hash_group.add_argument("--block-hashes", nargs="+", metavar="HASH",
                            help="Explicit block hash(es) to iterate over")

    parser.add_argument("--nonces",     nargs="+", required=True, metavar="SPEC",
                        help='Nonces: plain ints ("0 1 2"), range ("0:100"), step ("0:100:5")')
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Nonces per HTTP request (default: 8)")
    parser.add_argument("--model",      default="RedHatAI/Qwen2.5-7B-Instruct-quantized.w8a16",
                        help="Model name sent to server")
    parser.add_argument("--seq-len",    type=int, default=64,
                        help="Sequence length (default: 64)")
    parser.add_argument("--k-dim",      type=int, default=12,
                        help="k_dim for PoC (default: 12)")
    parser.add_argument("--output",     default="hidden_states.npz",
                        help="Output .npz file (default: hidden_states.npz)")
    parser.add_argument("--checkpoint-every", type=int, default=256, metavar="N",
                        help="Save a checkpoint .npz every N records (default: 256)")
    args = parser.parse_args()

    # ── resolve block hashes ──────────────────────────────────────────────────
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

    batches = [nonces[i:i + args.batch_size] for i in range(0, len(nonces), args.batch_size)]
    total_requests = len(block_hashes) * len(batches)

    print(f"Block hashes      : {len(block_hashes)}")
    print(f"Nonces            : {len(nonces)}  ({len(batches)} batches × ≤{args.batch_size})")
    print(f"Total requests    : {total_requests}")
    print(f"Output            : {args.output}")
    print(f"Checkpoint every  : {args.checkpoint_every} records")

    # ── accumulate in memory ──────────────────────────────────────────────────
    acc_block_hashes:          List[str]        = []
    acc_nonces:                List[int]         = []
    acc_hidden_states:         List[np.ndarray]  = []
    acc_reduced_hidden_states: List[np.ndarray]  = []

    n_written = 0
    n_errors  = 0

    out_base = args.output[:-4] if args.output.endswith(".npz") else args.output

    for block_hash in block_hashes:
        print(f"\n── block_hash={block_hash} ──")
        for batch_idx, batch in enumerate(batches):
            tag = f"  batch {batch_idx+1}/{len(batches)} nonces={batch}"
            try:
                resp = send_batch(
                    url=args.url,
                    block_hash=block_hash,
                    public_key=args.public_key,
                    nonces=batch,
                    model=args.model,
                    seq_len=args.seq_len,
                    k_dim=args.k_dim,
                    timeout=300,
                )
            except Exception as e:
                n_errors += len(batch)
                print(f"{tag}  → FAILED: {e}", file=sys.stderr)
                continue

            artifacts = (
                resp if isinstance(resp, list)
                else resp.get("artifacts", resp.get("results", []))
            )

            batch_written = 0
            for artifact in artifacts:
                nonce   = artifact.get("nonce")
                hs_b64  = artifact.get("hidden_state_b64")
                rhs_b64 = artifact.get("reduced_hidden_state_b64")

                if hs_b64 is None or rhs_b64 is None:
                    print(f"  WARNING: nonce={nonce} missing hidden state fields"
                          " (server may need restart)", file=sys.stderr)
                    continue

                acc_block_hashes.append(block_hash)
                acc_nonces.append(nonce)
                acc_hidden_states.append(decode_vector(hs_b64))
                acc_reduced_hidden_states.append(decode_vector(rhs_b64))
                n_written += 1
                batch_written += 1

            print(f"{tag}  → collected {batch_written}/{len(artifacts)} records"
                  f"  (total: {n_written})")

            # ── periodic checkpoint ───────────────────────────────────────────
            if n_written > 0 and n_written % args.checkpoint_every == 0:
                ckpt = f"{out_base}_checkpoint_{n_written}.npz"
                save_npz(ckpt, acc_block_hashes, acc_nonces,
                         acc_hidden_states, acc_reduced_hidden_states)
                print(f"  ✓ checkpoint saved → {ckpt}")

    # ── final save ────────────────────────────────────────────────────────────
    if n_written == 0:
        print(f"\nNo records collected. Output file not written.", file=sys.stderr)
    else:
        save_npz(args.output, acc_block_hashes, acc_nonces,
                 acc_hidden_states, acc_reduced_hidden_states)
        hs_shape  = f"[{n_written} × {len(acc_hidden_states[0])}]"
        rhs_shape = f"[{n_written} × {len(acc_reduced_hidden_states[0])}]"
        print(f"\nDone.")
        print(f"  Records        : {n_written}  (errors: {n_errors})")
        print(f"  hidden_states  : {hs_shape}  float32")
        print(f"  reduced        : {rhs_shape}  float32")
        print(f"  Output         : {args.output}")
        print(f"\nLoad with:")
        print(f"  data = np.load({args.output!r})")
        print(f"  hs   = data['hidden_states']           # {hs_shape}")
        print(f"  rhs  = data['reduced_hidden_states']   # {rhs_shape}")


if __name__ == "__main__":
    main()

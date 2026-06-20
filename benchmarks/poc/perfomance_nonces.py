#!/usr/bin/env python3
"""Decode-PoC throughput — client/server, production 32-nonce batch flow.

Pure HTTP client: connect to a running server with --url (the rented box runs only
ML node + vLLM); loop /generate with 32-nonce batches (max_tokens decode steps) for
a duration and report nonces/s + steps/s. No prefill, no callback.

  # against a running server (vLLM engine or ML-node proxy):
  python perfomance_nonces.py --url http://HOST:PORT --target vllm  --max-tokens 256
  python perfomance_nonces.py --url http://HOST:PORT --target mlnode --max-tokens 256

  # local dev (auto-boot vLLM, then connect — same client path):
  python perfomance_nonces.py --max-tokens 256 [--eager]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from poc_validation import (  # noqa: E402
    request_generate, save_run, add_engine_args, deploy_from_args,
)

DEFAULT_MODEL = "RedHatAI/Qwen2.5-7B-Instruct-quantized.w8a16"
BATCH = 32  # production: ML-node default batch_size + --poc-max-batch-size cap


def run_decode(url, target, model, seq_len, max_tokens, duration, warmup):
    """Client-driven continuous 32-nonce batches of /generate (decode trajectory)."""
    nxt = 0

    def one():
        nonlocal nxt
        request_generate(url, target=target, model=model, nonces=list(range(nxt, nxt + BATCH)),
                         seq_len=seq_len, max_tokens=max_tokens)
        nxt += BATCH

    w_end = time.monotonic() + warmup
    while time.monotonic() < w_end:
        one()
    total, t0, deadline = 0, time.monotonic(), time.monotonic() + duration
    while time.monotonic() < deadline:
        one()
        total += BATCH
    return total, time.monotonic() - t0


def _report(max_tokens, total, elapsed, prov):
    nps = total / elapsed if elapsed else 0.0
    print(f"\n=== decode-PoC throughput ===")
    print(f"nonces/s = {nps:.2f}   nonces/min = {nps*60:.0f}   steps/s = {nps*(max_tokens+1):.0f}   "
          f"({total} nonces in {elapsed:.1f}s, batch={BATCH}, max_tokens={max_tokens})")
    keys = ("vllm_version", "vllm_commit", "gpu", "attention_backend",
            "cudagraph_mode", "dtype", "quantization")
    print("  provenance: " + "  ".join(f"{k}={prov[k]}" for k in keys if k in prov))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    add_engine_args(ap)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--warmup", type=float, default=15.0)
    ap.add_argument("--save")
    a = ap.parse_args()

    with deploy_from_args(a, a.model) as (url, srv):
        total, elapsed = run_decode(url, a.target, a.model, a.seq_len,
                                    a.max_tokens, a.duration, a.warmup)
    prov = a.prov
    _report(a.max_tokens, total, elapsed, prov)
    if a.save:
        nps = total / elapsed if elapsed else 0.0
        save_run(a.save,
                 {"model": a.model, "seq_len": a.seq_len, "max_tokens": a.max_tokens,
                  "batch_size": BATCH, **prov}, [],
                 results={"nonces_per_s": round(nps, 3),
                          "steps_per_s": round(nps * (a.max_tokens + 1), 1),
                          "total_nonces": total, "elapsed_s": round(elapsed, 1)})
        print(f"saved -> {a.save}")


if __name__ == "__main__":
    main()

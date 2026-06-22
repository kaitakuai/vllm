#!/usr/bin/env python3
"""Throughput — one tool for BOTH PoC nonces and real chat inference.

Pure HTTP client: connect to a running server with --url (the rented box runs only
ML node + vLLM); loop for a duration and report throughput in a common frame so PoC
cost reads directly against real-inference capacity:

  * --mode poc  : 32-nonce /generate batches (decode trajectory) -> nonces/min, steps/s
  * --mode chat : 32 concurrent /v1/chat/completions             -> req/min, tokens/s

req/min (chat) and nonces/min (PoC) are the SAME unit: one decode sequence. Both run
the same concurrency (BATCH) and max_tokens, so the two rows are directly comparable.

  # against a running server (vLLM engine or ML-node proxy):
  python perfomance_nonces.py --mode poc  --url http://HOST:PORT --target vllm  --max-tokens 256
  python perfomance_nonces.py --mode chat --url http://HOST:PORT               --max-tokens 256

  # local dev (auto-boot vLLM, then connect — same client path):
  python perfomance_nonces.py --mode poc --max-tokens 256 [--eager]
"""
import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from poc_validation import (  # noqa: E402
    request_generate, save_run, add_engine_args, deploy_from_args,
)

DEFAULT_MODEL = "RedHatAI/Qwen2.5-7B-Instruct-quantized.w8a16"
BATCH = 32  # production: ML-node default batch_size + --poc-max-batch-size cap;
            # also the chat concurrency, so req/min lines up with nonce/min.


def run_poc(url, target, model, seq_len, max_tokens, duration, warmup):
    """Client-driven continuous 32-nonce batches of /generate (decode trajectory).
    Returns (total_requests=nonces, total_steps, elapsed)."""
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
    elapsed = time.monotonic() - t0
    return total, total * (max_tokens + 1), elapsed


def run_poc_pipeline(url, target, model, seq_len, max_tokens, duration, warmup):
    """APPLES-TO-APPLES with run_chat, NO door gap: BATCH worker threads each fire a
    SINGLE-nonce /generate back-to-back, so ~BATCH nonces are always in flight and the
    vLLM scheduler keeps the GPU continuously saturated. The serial run_poc above sends
    one 32-nonce request then WAITS (a gap between batches, mirrors production load);
    this version removes that gap by overlapping requests exactly like run_chat's 32
    continuous single-sequence workers -> fair PoC-vs-inference comparison.
    Returns (total_nonces, total_steps, elapsed)."""
    import threading
    from concurrent.futures import ThreadPoolExecutor
    c = {"next": 0, "done": 0}
    lock = threading.Lock()
    state = {"deadline": 0.0}

    def worker():
        while time.monotonic() < state["deadline"]:
            with lock:
                n = c["next"]; c["next"] += 1
            try:
                request_generate(url, target=target, model=model, nonces=[n],
                                 seq_len=seq_len, max_tokens=max_tokens)
            except Exception:
                continue  # skip a transient error, keep the pipeline full
            with lock:
                c["done"] += 1

    def phase(seconds):
        state["deadline"] = time.monotonic() + seconds
        with ThreadPoolExecutor(max_workers=BATCH) as ex:
            for f in [ex.submit(worker) for _ in range(BATCH)]:
                f.result()

    if warmup:
        phase(warmup)
    with lock:
        c["done"] = 0
    t0 = time.monotonic()
    phase(duration)
    elapsed = time.monotonic() - t0
    return c["done"], c["done"] * (max_tokens + 1), elapsed


async def _chat_one(client, url, model, max_tokens, idx):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": f"Write a long detailed essay about topic number {idx}."}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "ignore_eos": True,  # force exactly max_tokens decode steps (fair tok/s)
    }
    r = await client.post(f"{url}/v1/chat/completions", json=body)
    r.raise_for_status()
    return r.json().get("usage", {}).get("completion_tokens", 0)


async def _run_chat(url, model, max_tokens, duration, warmup, concurrency):
    import httpx
    counters = {"req": 0, "tok": 0, "idx": 0}

    async def worker(client, deadline):
        while time.monotonic() < deadline:
            i = counters["idx"]; counters["idx"] += 1
            tok = await _chat_one(client, url, model, max_tokens, i)
            counters["req"] += 1; counters["tok"] += tok

    async with httpx.AsyncClient(timeout=600) as client:
        await asyncio.gather(*[_chat_one(client, url, model, 8, -1) for _ in range(concurrency)])  # warmup
        if warmup:
            wend = time.monotonic() + warmup
            await asyncio.gather(*[worker(client, wend) for _ in range(concurrency)])
        counters["req"] = counters["tok"] = 0
        t0 = time.monotonic(); deadline = t0 + duration
        await asyncio.gather(*[worker(client, deadline) for _ in range(concurrency)])
        elapsed = time.monotonic() - t0
    return counters["req"], counters["tok"], elapsed


def run_chat(url, model, max_tokens, duration, warmup):
    return asyncio.run(_run_chat(url, model, max_tokens, duration, warmup, BATCH))


def _results(mode, total_req, work, elapsed, max_tokens):
    rpm = total_req / elapsed * 60 if elapsed else 0.0
    wps = work / elapsed if elapsed else 0.0
    res = {"mode": mode, "req_per_min": round(rpm, 1),
           "total_req": total_req, "elapsed_s": round(elapsed, 1), "max_tokens": max_tokens}
    res["steps_per_s" if mode == "poc" else "tokens_per_s"] = round(wps, 1)
    if mode == "poc":  # back-compat keys
        res["nonces_per_s"] = round(total_req / elapsed if elapsed else 0.0, 3)
        res["total_nonces"] = total_req
    return res


def _print(res, prov):
    unit = "nonces" if res["mode"] == "poc" else "requests"
    work = f"steps/s={res['steps_per_s']}" if res["mode"] == "poc" else f"tokens/s={res['tokens_per_s']}"
    print(f"\n=== {res['mode']} throughput ===")
    print(f"req/min = {res['req_per_min']:.0f}   {work}   "
          f"({res['total_req']} {unit} in {res['elapsed_s']}s, "
          f"concurrency={BATCH}, max_tokens={res['max_tokens']})")
    keys = ("vllm_version", "vllm_commit", "gpu", "attention_backend",
            "cudagraph_mode", "dtype", "quantization")
    print("  provenance: " + "  ".join(f"{k}={prov[k]}" for k in keys if k in prov))


def _save_path(stem, mode, multi):
    if not multi:
        return stem
    return stem[:-5] + f".{mode}.json" if stem.endswith(".json") else f"{stem}.{mode}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["poc", "chat", "both"], default="poc")
    ap.add_argument("--model", required=True)
    add_engine_args(ap)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--warmup", type=float, default=15.0)
    ap.add_argument("--poc-load", choices=["serial", "pipeline"], default="pipeline",
                    help="pipeline=32 continuous single-nonce workers, NO gap, "
                         "apples-to-apples with chat (default). serial=one 32-nonce "
                         "request then wait (production load; has an inter-batch gap).")
    ap.add_argument("--save")
    a = ap.parse_args()

    modes = ["poc", "chat"] if a.mode == "both" else [a.mode]
    with deploy_from_args(a, a.model) as (url, srv):  # one server lifetime for all modes
        for mode in modes:
            if mode == "poc":
                poc_fn = run_poc_pipeline if a.poc_load == "pipeline" else run_poc
                total, work, elapsed = poc_fn(url, a.target, a.model, a.seq_len,
                                              a.max_tokens, a.duration, a.warmup)
            else:
                total, work, elapsed = run_chat(url, a.model, a.max_tokens, a.duration, a.warmup)
            res = _results(mode, total, work, elapsed, a.max_tokens)
            if mode == "poc":
                res["poc_load"] = a.poc_load   # serial (production) vs pipeline (fair)
            _print(res, a.prov)
            if a.save:
                meta = {"model": a.model, "mode": mode, "seq_len": a.seq_len,
                        "max_tokens": a.max_tokens, "batch_size": BATCH,
                        "poc_load": (a.poc_load if mode == "poc" else None), **a.prov}
                path = _save_path(a.save, mode, len(modes) > 1)
                save_run(path, meta, [], results=res)
                print(f"saved -> {path}")


if __name__ == "__main__":
    main()

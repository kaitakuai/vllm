#!/usr/bin/env python3
"""Chat-only decode throughput benchmark (no PoC). Fires N concurrent chat
completions with a fixed output length (ignore_eos) and reports tokens/sec +
latency. Used to compare cudagraph-ON vs --enforce-eager."""
import argparse, asyncio, time, httpx


async def one(client, url, model, max_tokens, idx):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": f"Write a long detailed essay about topic number {idx}."}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "ignore_eos": True,  # force exactly max_tokens decode steps (fair tok/s)
    }
    t0 = time.monotonic()
    r = await client.post(f"{url}/v1/chat/completions", json=body)
    r.raise_for_status()
    d = r.json()
    dt = time.monotonic() - t0
    out_tok = d.get("usage", {}).get("completion_tokens", 0)
    return out_tok, dt


async def run(url, model, n, max_tokens):
    async with httpx.AsyncClient(timeout=600) as client:
        # warmup
        await one(client, url, model, 8, 999)
        t0 = time.monotonic()
        res = await asyncio.gather(*[one(client, url, model, max_tokens, i) for i in range(n)])
        wall = time.monotonic() - t0
    toks = sum(t for t, _ in res)
    lats = sorted(dt for _, dt in res)
    print(f"n={n} max_tokens={max_tokens}")
    print(f"  total_output_tokens={toks}  wall={wall:.2f}s")
    print(f"  THROUGHPUT={toks/wall:.1f} tok/s")
    print(f"  latency p50={lats[len(lats)//2]:.2f}s  p95={lats[int(len(lats)*0.95)]:.2f}s  max={lats[-1]:.2f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=128)
    a = ap.parse_args()
    asyncio.run(run(a.url, a.model, a.n, a.max_tokens))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Localize the concurrent-PoC → chat corruption.

One server boot. Capture a fixed GREEDY chat baseline (no PoC), then for each PoC
scenario fire it concurrently and re-send the same prompts; report how many diverge.
Greedy (temp=0) → any divergence is engine-side. Scenarios isolate the trigger:
  - n1_decode    : 1 nonce, 256 decode steps   (single PoC decode req)
  - n32_prefill  : 32 nonces, max_tokens=0     (PoC prefill only, no decode)
  - n32_decode   : 32 nonces, 256 decode steps (full prod load)
"""
import sys, time, threading, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import requests
from poc_validation import deploy, request_generate, wait_ready

MODEL = "RedHatAI/Qwen2.5-7B-Instruct-quantized.w8a16"
PROMPTS = [
    "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and bakes muffins with 4. "
    "She sells the rest at $2 each. How much does she make per day? Show your reasoning step by step.",
    "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts total? "
    "Explain step by step.",
    "Weng earns $12 an hour for babysitting. Yesterday she babysat for 50 minutes. "
    "How much did she earn? Show your work.",
]
import os as _os
_ALL = [("n1_decode", 1, 256), ("n32_prefill", 32, 0), ("n32_decode", 32, 256)]
SCENARIOS = [s for s in _ALL if not _os.environ.get("REPRO_ONLY") or s[0] == _os.environ.get("REPRO_ONLY")]

def chat(url, prompt, max_tokens=320):
    r = requests.post(f"{url}/v1/chat/completions", json={
        "model": MODEL, "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0.0, "seed": 0,
        "frequency_penalty": 0.3}, timeout=600)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def run_scenario(url, name, nonces, mt, base):
    errs = []
    stop = threading.Event()
    def poc():
        try:
            while not stop.is_set():
                request_generate(url, model=MODEL, nonces=list(range(nonces)),
                                 seq_len=256, max_tokens=mt, k_dim=12, batch_size=32)
        except Exception as e:
            errs.append(repr(e))
    t = threading.Thread(target=poc); t.start()
    time.sleep(2.0)
    conc = [chat(url, p) for p in PROMPTS] if t.is_alive() else None
    # one more round to maximize overlap with decode
    conc = [chat(url, p) for p in PROMPTS]
    stop.set(); t.join()
    ndiff = sum(b != c for b, c in zip(base, conc))
    bad = [i for i, (b, c) in enumerate(zip(base, conc)) if b != c]
    print(f"[{name:12}] nonces={nonces} mt={mt}  diverged={ndiff}/{len(PROMPTS)}  "
          f"{'CORRUPT' if ndiff else 'clean'}  pocErr={len(errs)}")
    for i in bad[:1]:
        print(f"    e.g. prompt{i} CONC tail: {conc[i][-120:]!r}")
    return ndiff

def main():
    eager = os.environ.get("REPRO_EAGER", "0") == "1"
    extra = ["--max-model-len", "4096", "--no-enable-prefix-caching"]
    if os.environ.get("REPRO_NOASYNC") == "1":
        extra.append("--no-async-scheduling")
    print(f"=== eager={eager} noasync={os.environ.get('REPRO_NOASYNC')=='1'} ===", flush=True)
    with deploy(MODEL, eager=eager, extra_args=extra) as (url, srv):
        wait_ready(url, MODEL)
        base = [chat(url, p) for p in PROMPTS]
        print("baseline captured\n")
        for name, n, mt in SCENARIOS:
            run_scenario(url, name, n, mt, base)

if __name__ == "__main__":
    main()

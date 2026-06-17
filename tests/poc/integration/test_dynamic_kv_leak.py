"""Regression: PoC must RELEASE its KV blocks on completion — no leak — and must
never preempt/starve chat under sustained concurrent load.

PoC allocates real paged blocks on demand via the KVCacheManager (exactly like
chat) and frees them when the trajectory finishes. If freeing regresses, blocks
leak: every wave succeeds in a smoke test, but KV usage ratchets up until chat
starves in production — exactly the class the standing "always test KV +
multi-nonce" rule targets.

Contract (GPU, >=2 concurrent nonces):
1. NO LEAK — after several PoC waves complete, idle KV usage returns to its
   pre-PoC baseline (blocks were freed, not leaked).
2. NO STARVATION — every concurrent chat returns a non-empty response and every
   PoC wave returns a full batch of artifacts (PoC defers on contention, never
   preempts chat).
"""
import asyncio
import re

import httpx
import pytest

from tests.poc._server import (
    CANONICAL_MODEL as MODEL,
    DEFAULT_SERVER_ARGS as BASE_ARGS,
    PoCTestServer,
)
from tests.poc.utils import poc_request_body

POC_URL = "/api/v1/pow/generate"
TIMEOUT = 180
NONCES = [0, 1, 2, 3]      # >=2 concurrent nonces (the standing rule)
POC_MAX_TOKENS = 64
N_CHATS = 4
CHAT_MAX_TOKENS = 64
ROUNDS = 4
# Idle KV usage is ~0; allow a small margin for any settling. A leak ratchets up
# by whole PoC footprints (~33 blocks per nonce), far above this.
LEAK_TOL = 0.02
_USAGE_RE = re.compile(r"^vllm:kv_cache_usage_perc(?:\{[^}]*\})?\s+([0-9.eE+-]+)", re.M)


@pytest.fixture(scope="module")
def server():
    with PoCTestServer(MODEL, BASE_ARGS) as srv:   # default => dynamic KV
        yield srv


def _kv_usage(url: str) -> float:
    """Max kv_cache_usage_perc gauge from /metrics (0.0 when idle)."""
    r = httpx.get(f"{url}/metrics", timeout=30)
    r.raise_for_status()
    vals = [float(m) for m in _USAGE_RE.findall(r.text)]
    return max(vals) if vals else 0.0


def _settle_idle(url: str, baseline: float, tol: float, tries: int = 15) -> float:
    """Poll until KV usage drops back to ~baseline (blocks freed), else return last."""
    last = _kv_usage(url)
    for _ in range(tries):
        if last <= baseline + tol:
            return last
        # synchronous spin with a tiny server round-trip in between
        httpx.get(f"{url}/health", timeout=10)
        last = _kv_usage(url)
    return last


async def _poc(client, url, infk=None):
    body = poc_request_body("0xkvleak", NONCES, MODEL, wait=True, max_tokens=POC_MAX_TOKENS)
    if infk is not None:
        body["enforced_k_steps"] = infk
    r = await client.post(f"{url}{POC_URL}", json=body)
    r.raise_for_status()
    d = r.json()
    assert d.get("status") == "completed", d
    arts = d.get("artifacts", [])
    assert len(arts) == len(NONCES), f"expected {len(NONCES)} artifacts, got {len(arts)}"
    return arts


async def _chat(server, idx):
    client = server.get_async_client()
    r = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": f"Write a short paragraph about topic {idx}."}],
        max_tokens=CHAT_MAX_TOKENS, temperature=0.0,
    )
    return r.choices[0].message.content or ""


@pytest.mark.integration
def test_dynamic_kv_no_leak_no_starvation(server):
    url = server.url_root

    async def _round():
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            chats = [asyncio.create_task(_chat(server, i)) for i in range(N_CHATS)]
            arts = await _poc(client, url)
            texts = await asyncio.gather(*chats)
            return arts, texts

    # Warm up (absorb first-request init), then record the idle baseline.
    asyncio.run(_round())
    baseline = _settle_idle(url, 0.0, LEAK_TOL)

    for _ in range(ROUNDS):
        arts, texts = asyncio.run(_round())
        # No starvation: full PoC batch + every chat answered.
        assert len(arts) == len(NONCES)
        for i, t in enumerate(texts):
            assert t and t.strip(), f"chat {i} starved (empty) under PoC load"

    # No leak: idle KV usage returns to ~baseline once every wave has freed.
    settled = _settle_idle(url, baseline, LEAK_TOL)
    assert settled <= baseline + LEAK_TOL, (
        f"dynamic KV LEAK: idle usage {settled:.4f} stayed above baseline "
        f"{baseline:.4f}+{LEAK_TOL} after {ROUNDS} PoC waves — blocks not freed."
    )

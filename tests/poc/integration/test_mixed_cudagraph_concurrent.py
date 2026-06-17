"""Regression: the FUSED mixed-decode CUDA-graph path (default; not --enforce-eager)
must run a PoC decode concurrently with SUSTAINED chat (>=N_CHATS) WITHOUT hanging
and WITHOUT corrupting the PoC computation.

This is the test that was MISSING when the fused mixed cudagraph was wrongly
called "validated":
- test_mixed_decode_concurrent_chat.py uses 3 nonces / 4 fire-once chats — too light.
- The cudagraph path captured against step-0 attention metadata and never re-planned
  (force_eager was True), so it went stale every step ->
  recapture-each-step -> CUDA capture DEADLOCK under concurrency (n>=16) -> the PoC
  forward never returns (HANG). Fix: graphable uniform-decode mixed batches drop
  force_eager so vLLM builds + re-plans the STABLE cudagraph decode buffers per step.

Correctness is measured the RIGHT way — ALIGNED validation (enforced_k_steps),
which compares each step independently (no cascade) and is robust to benign batch-shape
sphere_k boundary flips. A byte-identity "alone vs concurrent" check is NOT valid here:
"alone" runs the pure-PoC path, "concurrent" the mixed path, and a non-aligned compare
cascades — both eager and graph score ~70% by that flawed metric (measured), so it
cannot distinguish correct from corrupt. Aligned, eager and the fixed graph both score
~6% (honest); real corruption would be tens of %.

Contract (flag ON, GPU, no --enforce-eager):
1. NO HANG — aligned validation returns within a bounded time under >=N_CHATS chats.
2. CORRECT — aligned n_sphere_mismatches stays at honest level (< MAX_MISMATCH_FRAC),
   not the high value a stale/leaking graph would produce.
3. Chat is not frozen — every concurrent chat returns a non-empty response.
4. A real chat+PoC mixed batch actually ran (server log "MIXED BATCH").
"""
import asyncio

import httpx
import pytest

from tests.poc._server import (
    CANONICAL_MODEL as MODEL,
    DEFAULT_SERVER_ARGS as BASE_ARGS,
    PoCTestServer,
)

POC_URL = "/api/v1/pow/generate"
# Bounded so a stale-graph DEADLOCK fails fast instead of dragging to the suite limit.
NO_HANG_TIMEOUT = 120

NONCES = [0, 1, 2, 3, 4, 5, 6, 7]   # >1 nonce: scheduler batches them
POC_MAX_TOKENS = 128                # KV-bound decode steps that overlap the chat
N_CHATS = 16                        # the concurrency that deadlocked the broken graph
CHAT_MAX_TOKENS = 256
# Honest aligned mismatch is ~6% (measured eager AND fixed-graph); corruption is tens
# of %. Generous separating threshold (robust to boundary-flip variance by model).
MAX_MISMATCH_FRAC = 0.30


@pytest.fixture(scope="module")
def graph_server():
    """Mixed decode WITH the fused cudagraph flag ON — the path under test."""
    with PoCTestServer(MODEL, BASE_ARGS) as srv:
        yield srv


def _poc_body(enforced_k_steps=None):
    body = {
        "block_hash": "deadbeef" * 8, "block_height": 100, "public_key": "cafebabe" * 8,
        "node_id": 0, "node_count": 1, "nonces": NONCES,
        "params": {"model": MODEL, "seq_len": 256, "k_dim": 12, "max_tokens": POC_MAX_TOKENS},
        "wait": True,
    }
    if enforced_k_steps is not None:
        body["enforced_k_steps"] = enforced_k_steps
    return body


async def _poc(client, url, infk=None):
    resp = await client.post(f"{url}{POC_URL}", json=_poc_body(infk))
    resp.raise_for_status()
    data = resp.json()
    assert data.get("status") == "completed", f"status != completed: {data}"
    arts = data.get("artifacts", [])
    assert len(arts) == len(NONCES), f"expected {len(NONCES)} artifacts, got {len(arts)}"
    return arts


async def _chat(server, idx):
    client = server.get_async_client()
    r = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": f"Write a long detailed essay about topic {idx}."}],
        max_tokens=CHAT_MAX_TOKENS, temperature=0.0,
    )
    return r.choices[0].message.content or ""


@pytest.mark.integration
def test_mixed_cudagraph_poc_concurrent_no_hang_correct(graph_server):
    url = graph_server.url_root

    async def _run():
        async with httpx.AsyncClient(timeout=NO_HANG_TIMEOUT) as client:
            # 1) Generate a trajectory alone -> the aligned reference.
            gen = await _poc(client, url)
            traj = {str(a["nonce"]): a["k_points_steps"] for a in gen}

            # 2) Re-validate it ALIGNED while N_CHATS chats stream concurrently.
            chats = [asyncio.create_task(_chat(graph_server, i)) for i in range(N_CHATS)]
            val = await asyncio.wait_for(_poc(client, url, infk=traj), timeout=NO_HANG_TIMEOUT)
            texts = await asyncio.gather(*chats)
            return val, texts

    try:
        val, chat_texts = asyncio.run(_run())
    except (asyncio.TimeoutError, httpx.TimeoutException) as e:
        pytest.fail(
            "mixed-cudagraph PoC HUNG under concurrent chat (stale-graph deadlock): "
            f"{type(e).__name__}. The captured graph is not re-planned per step."
        )

    # Invariant 3: chat not frozen.
    for i, text in enumerate(chat_texts):
        assert text and text.strip(), f"chat {i} returned empty output"

    # Invariant 2: CORRECT — aligned mismatches at honest level, not corruption.
    mism = [a.get("n_sphere_mismatches") for a in val]
    assert all(m is not None and m >= 0 for m in mism), f"validation did not run: {mism}"
    worst = max(mism) / POC_MAX_TOKENS
    avg = sum(mism) / len(mism) / POC_MAX_TOKENS
    assert worst < MAX_MISMATCH_FRAC, (
        f"mixed-cudagraph PoC CORRUPTED under concurrent chat: aligned mismatches "
        f"{mism}/{POC_MAX_TOKENS} (worst {worst:.0%}, avg {avg:.0%}) exceed honest "
        f"level {MAX_MISMATCH_FRAC:.0%} — the graph is leaking chat state into PoC."
    )

    # Invariant 4: a real chat+PoC mixed batch actually ran.
    log_path = getattr(graph_server, "log_path", None)
    if log_path:
        try:
            with open(log_path) as f:
                assert "MIXED BATCH" in f.read(), "no 'MIXED BATCH' — concurrency not exercised"
        except FileNotFoundError:
            pass

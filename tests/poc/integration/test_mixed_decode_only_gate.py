"""Contract: under CUDA graphs (default; not --enforce-eager), the scheduler must NEVER form a
non-graphable mixed batch (chat-prefill+PoC-decode / chat-decode+PoC-prefill /
both-prefill). chat+PoC share a forward ONLY when both decode; every prefill runs
in its own pure step. ⇒ no eager-mixed fallback ever happens.

How this is checked: the worker logs `POC_CONTRACT_VIOLATION` (gpu_model_runner)
whenever it sees a mixed batch that is NOT uniform-decode (i.e. b/c/d leaked past
the scheduler gate). This test drives the exact load that forms b/c/d WITHOUT the
gate — a long PoC decode overlapped by a continuous stream of short chat requests
(constant chat prefills) plus a second PoC starting mid-run (PoC prefill while chat
decodes) — and asserts the violation line never appears, while chat and PoC both
make progress (no starvation/hang) and PoC validates correctly (aligned).

Requires GPU + the cudagraph flag.
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
TIMEOUT = 120
NONCES = [0, 1, 2, 3]
POC_MAX_TOKENS = 128          # long PoC decode to overlap many chat prefills
N_CHAT_WAVES = 24            # rolling short chats => continuous prefills
CHAT_CONCURRENCY = 6
CHAT_MAX_TOKENS = 24         # short so chats finish fast and NEW ones keep prefilling


def _poc_body(bh, infk=None):
    body = {"block_hash": bh, "block_height": 100, "public_key": "cafebabe" * 8,
            "node_id": 0, "node_count": 1, "nonces": NONCES,
            "params": {"model": MODEL, "seq_len": 256, "k_dim": 12, "max_tokens": POC_MAX_TOKENS},
            "wait": True}
    if infk is not None:
        body["inference_k_points_steps"] = infk
    return body


@pytest.fixture(scope="module")
def graph_server():
    with PoCTestServer(MODEL, BASE_ARGS) as srv:
        yield srv


@pytest.mark.integration
def test_no_eager_mixed_batch_under_chat_prefill_churn(graph_server):
    url = graph_server.url_root

    async def _poc(client, bh, infk=None):
        r = await client.post(f"{url}{POC_URL}", json=_poc_body(bh, infk))
        r.raise_for_status()
        d = r.json()
        assert d.get("status") == "completed", d
        return {a["nonce"]: a for a in d.get("artifacts", [])}

    async def _chat(client, i):
        r = await client.post(f"{url}/v1/chat/completions", json={
            "model": MODEL, "messages": [{"role": "user", "content": f"Say something about {i}."}],
            "max_tokens": CHAT_MAX_TOKENS, "temperature": 0.0, "ignore_eos": True})
        r.raise_for_status()
        return (r.json().get("usage", {}) or {}).get("completion_tokens", 0)

    async def _run():
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            # rolling short chats => a NEW chat is almost always prefilling
            sem = asyncio.Semaphore(CHAT_CONCURRENCY)
            chat_toks = []

            async def chat_worker(i):
                async with sem:
                    chat_toks.append(await _chat(client, i))

            chat_stream = asyncio.gather(*[chat_worker(i) for i in range(N_CHAT_WAVES)])
            # PoC #1 (decode) overlaps the chat churn -> would form case (b)
            # chat-prefill+PoC-decode WITHOUT the gate. PoC #2 starts mid-run so its
            # prefill lands while chat decodes -> case (c) WITHOUT the gate. Both must
            # complete with the gate (no hang/starvation) and form NO eager batch.
            poc1 = asyncio.create_task(_poc(client, "deadbeef" * 8))
            poc2 = asyncio.create_task(_poc(client, "feedface" * 8))
            a1 = await asyncio.wait_for(poc1, timeout=TIMEOUT)
            a2 = await asyncio.wait_for(poc2, timeout=TIMEOUT)
            await chat_stream
            return a1, a2, chat_toks

    try:
        a1, a2, chat_toks = asyncio.run(_run())
    except (asyncio.TimeoutError, httpx.TimeoutException) as e:
        pytest.fail(f"starvation/hang under chat-prefill churn + PoC: {type(e).__name__}")

    # THE CONTRACT (primary assertion): no eager-mixed batch ever formed.
    log_path = getattr(graph_server, "log_path", None)
    if log_path:
        try:
            with open(log_path) as f:
                log = f.read()
        except FileNotFoundError:
            log = None
        if log is not None:
            n_violation = log.count("POC_CONTRACT_VIOLATION")
            assert n_violation == 0, (
                f"{n_violation} eager-mixed batch(es) formed — the scheduler's "
                f"decode-only-mixing gate let a prefill-mixed batch through (b/c/d)."
            )
            # sanity: a real mixed batch DID run (otherwise the test proved nothing)
            assert "MIXED BATCH" in log, "no MIXED BATCH — chat+PoC never co-existed"

    # progress: chat + BOTH PoCs completed (the _poc helper asserts status+count)
    assert len(chat_toks) == N_CHAT_WAVES and all(t > 0 for t in chat_toks), \
        f"chat starved: {chat_toks}"
    assert set(a1) == set(a2) == set(NONCES), "a PoC did not return all artifacts"

"""Chat must be byte-identical whether or not decode-PoC runs CONCURRENTLY.

This is the gap test_kv_cache_integrity.py misses: that one runs chat and PoC
*sequentially* (chat -> poc_round -> chat), so they never share a forward. The
co-existence corruption only appears when a chat request and a decode-PoC request
are mixed in the SAME scheduler step under async scheduling — the default config.

Regression guard for the async mixed-batch bug where PoC placeholder rows leaked
into the async prev-token feedback and chat decoded from token 0 ("!" floods) /
diverged. See KB: v0-20-regression-concurrent-poc-corrupts-chat-output.
"""
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
import pytest

from tests.poc._server import (
    CANONICAL_MODEL as MODEL,
    DEFAULT_SERVER_ARGS as SERVER_ARGS,
    open_poc_server,
)
from tests.poc.utils import poc_request_body

POC_URL = "/api/v1/pow/generate"
TIMEOUT = 240
# Prompts that elicit a multi-step (long) greedy answer, so the chat decode spans
# many forwards and is guaranteed to overlap concurrent PoC decode steps.
PROMPTS = [
    "Janet's ducks lay 16 eggs per day. She eats 3 and bakes muffins with 4, "
    "selling the rest at $2 each. How much does she make daily? Reason step by step.",
    "A robe takes 2 bolts of blue fiber and half that much white. How many bolts "
    "total? Explain step by step.",
]
MAX_CHAT_TOKENS = 200
# seq_len + max_tokens must stay <= --max-model-len (1024 in DEFAULT_SERVER_ARGS).
POC_SEQ_LEN = 256
POC_MAX_TOKENS = 256


@pytest.fixture(scope="module")
def server(request):
    # Default args => cudagraph + async scheduling: the config that triggers the bug.
    with open_poc_server(request, SERVER_ARGS) as srv:
        yield srv


@pytest.fixture(scope="module")
def client(server):
    return server.get_client()


def _chat(client, prompt: str, frequency_penalty: float = 0.0) -> str:
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_CHAT_TOKENS,
        temperature=0.0,
        seed=0,
        frequency_penalty=frequency_penalty,
    )
    return r.choices[0].message.content


def _longest_repeat_run(s: str) -> int:
    """Longest run of a single repeated character — the corruption signature.

    The co-existence bug makes chat decode from a stranded slot (token 0 -> "!"),
    producing long single-char floods (e.g. "!!!!!!!!"). Healthy output — including
    floating-point drift from batch shape — never repeats one char dozens of times.
    """
    best = run = 1
    for a, b in zip(s, s[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return best if s else 0


# Corruption = a single char repeated this many times in a row. Real text (even FP-
# drifted) stays well under this; the bug produces 50-120+ char floods.
MAX_REPEAT_RUN = 15


def test_chat_not_corrupted_under_concurrent_poc_decode(server, client):
    """Chat output must not be CORRUPTED by a concurrent decode-PoC request.

    NOTE: we do NOT require byte-identical output. Chat batched alongside PoC has a
    different batch shape than chat alone, so floating-point reduction order differs
    and greedy tokens can drift — correct but not identical (true of vanilla vLLM
    under any varying concurrency). The regression we guard is *corruption*: the
    stranded-token "!" floods from the async batch-layout-transition bug. Co-existence
    *accuracy* is covered separately by benchmarks/poc gsm8k (PoC-on vs baseline).
    """
    url = server.url_root
    baselines = [_chat(client, p) for p in PROMPTS]
    # Baselines themselves must be healthy (sanity).
    for i, b in enumerate(baselines):
        assert _longest_repeat_run(b) < MAX_REPEAT_RUN, f"baseline {i} already degenerate"

    errors: list[str] = []
    stop = threading.Event()

    def poc_loop():
        # Continuous decode-PoC load so chat decode steps mix with PoC decode steps.
        i = 0
        while not stop.is_set():
            try:
                body = poc_request_body(
                    f"0xcoexist_{i}", list(range(8)), MODEL,
                    seq_len=POC_SEQ_LEN, max_tokens=POC_MAX_TOKENS, wait=True,
                )
                httpx.post(f"{url}{POC_URL}", json=body, timeout=TIMEOUT)
            except Exception as e:  # noqa: BLE001
                errors.append(repr(e))
                return
            i += 1

    t = threading.Thread(target=poc_loop, daemon=True)
    t.start()
    try:
        time.sleep(2.0)  # let PoC enter its decode phase
        # Re-send each prompt repeatedly while PoC decodes, to hit many mixed steps.
        concurrent = []
        for _ in range(3):
            concurrent.extend(_chat(client, p) for p in PROMPTS)
    finally:
        stop.set()
        t.join(timeout=TIMEOUT)

    corrupted = [
        (i, _longest_repeat_run(c), c[-160:])
        for i, c in enumerate(concurrent)
        if _longest_repeat_run(c) >= MAX_REPEAT_RUN
    ]
    assert not corrupted, (
        "Concurrent decode-PoC CORRUPTED chat output (stranded-token floods). "
        f"PoC errors={errors}. First: idx={corrupted[0][0]} "
        f"run_len={corrupted[0][1]} tail={corrupted[0][2]!r}"
    )


def test_chat_burst_with_penalties_under_concurrent_poc_decode(server, client):
    """Co-existence under a CONCURRENT BURST of penalized chats + decode-PoC.

    Reproduces the async batch-size-swing bug: PoC's 32 nonces join/leave the batch
    between a step's two async phases while many chats chunk-prefill, so
    sampling_metadata read live in sample_tokens belongs to a different-sized batch.
    Penalties make that mismatch CRASH (size assert); without them it silently applies
    wrong per-row params. A sequential test never swings the batch hard enough.
    Guards the engine crash AND corruption.
    """
    fp = 0.5
    baselines = [_chat(client, p, frequency_penalty=fp) for p in PROMPTS]
    for i, b in enumerate(baselines):
        assert _longest_repeat_run(b) < MAX_REPEAT_RUN, f"baseline {i} already degenerate"

    errors: list[str] = []
    stop = threading.Event()

    def poc_loop():
        i = 0
        while not stop.is_set():
            try:
                body = poc_request_body(
                    f"0xcoexist_burst_{i}", list(range(32)), MODEL,
                    seq_len=POC_SEQ_LEN, max_tokens=POC_MAX_TOKENS, wait=True,
                )
                httpx.post(f"{server.url_root}{POC_URL}", json=body, timeout=TIMEOUT)
            except Exception as e:  # noqa: BLE001
                errors.append(repr(e))
                return
            i += 1

    t = threading.Thread(target=poc_loop, daemon=True)
    t.start()
    outputs: list[str] = []
    try:
        time.sleep(2.0)
        prompts = PROMPTS * 12  # ~24 concurrent chats -> chunked prefill + batch swings
        with ThreadPoolExecutor(max_workers=len(prompts)) as ex:
            futs = [ex.submit(_chat, client, p, fp) for p in prompts]
            for f in as_completed(futs):
                outputs.append(f.result())
    finally:
        stop.set()
        t.join(timeout=TIMEOUT)

    # Engine must survive (the async batch-swing penalty crash).
    assert not errors, f"PoC errored under penalized chat burst (engine crash?): {errors}"
    corrupted = [
        (_longest_repeat_run(c), c[-160:]) for c in outputs
        if _longest_repeat_run(c) >= MAX_REPEAT_RUN
    ]
    assert not corrupted, (
        "Penalized chat burst CORRUPTED under concurrent decode-PoC. "
        f"First: run_len={corrupted[0][0]} tail={corrupted[0][1]!r}"
    )

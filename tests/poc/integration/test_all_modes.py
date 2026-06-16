"""Integration tests covering all PoC request modes.

Covers pure chat, pure PoC, mixed concurrent batches, hook caching,
different block hashes, and high-concurrency stress.
"""

import asyncio
import concurrent.futures
import math
import random
import time

import httpx
import pytest

from tests.poc._server import CANONICAL_MODEL as MODEL, DEFAULT_SERVER_ARGS as SERVER_ARGS, open_poc_server
from tests.poc.utils import check_artifact, decode_artifact_vector, poc_request_body



@pytest.fixture(scope="module")
def server(request):
    with open_poc_server(request, SERVER_ARGS) as srv:
        yield srv


@pytest.fixture(scope="module")
def server_url(server):
    return server.url_root


@pytest.fixture(scope="module")
def model_name():
    return MODEL


@pytest.fixture(scope="module")
def client(server):
    return server.get_client()


@pytest.fixture(scope="module")
def async_client(server):
    return server.get_async_client()

_CHAT_TEMPLATES = [
    "Explain how a {thing} works in 2-3 sentences.",
    "Write a short poem about {thing}.",
    "What are 3 interesting facts about {thing}?",
    "Compare and contrast {thing} with {thing2}.",
    "Describe {thing} to a 5-year-old.",
    "What would happen if {thing} didn't exist?",
    "Write a haiku about {thing}.",
    "Tell me a joke involving {thing}.",
]

_THINGS = [
    "quantum computing", "black holes", "sourdough bread", "the Roman Empire",
    "electric cars", "photosynthesis", "jazz music", "neural networks",
    "volcanoes", "origami", "the stock market", "espresso",
    "coral reefs", "satellites", "penguins", "compilers",
]


def _random_chat_prompt() -> str:
    template = random.choice(_CHAT_TEMPLATES)
    return template.format(thing=random.choice(_THINGS), thing2=random.choice(_THINGS))


@pytest.mark.integration
class TestPureChat:
    """Basic chat endpoint smoke test."""

    def test_simple_completion(self, client, model_name):
        """Chat endpoint returns a non-empty response."""
        completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "What is 2+2? Answer briefly."}],
            max_tokens=20,
        )
        assert len(completion.choices[0].message.content) > 0


@pytest.mark.integration
class TestPurePoC:
    def test_generate_returns_valid_artifacts(self, server_url, model_name):
        """generate endpoint returns completed status with valid artifacts."""
        k_dim = 12
        body = poc_request_body("0xtest_pure_poc", [1, 2, 3, 4, 5], model_name, k_dim=k_dim)
        resp = httpx.post(f"{server_url}/api/v1/pow/generate", json=body, timeout=60)
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("status") == "completed"

        artifacts = data.get("artifacts", [])
        valid = [a for a in artifacts if check_artifact(a, k_dim)]
        assert len(valid) == 5

        for artifact in artifacts:
            vec = decode_artifact_vector(artifact["vector_b64"])
            assert all(math.isfinite(v) for v in vec), \
                f"Artifact nonce={artifact['nonce']} contains NaN or inf"
            norm = math.sqrt(sum(v * v for v in vec))
            assert abs(norm - 1.0) < 0.02, \
                f"Artifact nonce={artifact['nonce']} norm={norm:.4f} is not ≈1.0"

        nonces_returned = [a["nonce"] for a in artifacts]
        assert len(set(nonces_returned)) == len(artifacts), \
            "All returned artifact nonces must be distinct"

    def test_artifact_vector_shape(self, server_url, model_name):
        """Each artifact vector has the expected k_dim length."""
        k_dim = 12
        body = poc_request_body("0xtest_vector_shape", [0], model_name, k_dim=k_dim)
        resp = httpx.post(f"{server_url}/api/v1/pow/generate", json=body, timeout=60)
        assert resp.status_code == 200
        data = resp.json()
        artifact = data["artifacts"][0]
        vec = decode_artifact_vector(artifact["vector_b64"])
        assert len(vec) == k_dim


@pytest.mark.integration
class TestMixedBatch:
    def test_concurrent_chat_and_poc(self, client, server_url, model_name):
        """Simultaneous chat and PoC requests complete successfully."""
        def _chat():
            return client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "Count from 1 to 10."}],
                max_tokens=50,
            )

        def _poc():
            return httpx.post(
                f"{server_url}/api/v1/pow/generate",
                json=poc_request_body("0xtest_mixed", list(range(10, 20)), model_name),
                timeout=60,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            chat_future = pool.submit(_chat)
            poc_future = pool.submit(_poc)
            chat_result = chat_future.result()
            poc_resp = poc_future.result()

        assert len(chat_result.choices[0].message.content) > 0

        assert poc_resp.status_code == 200
        poc_data = poc_resp.json()
        assert poc_data.get("status") == "completed"
        assert len(poc_data.get("artifacts", [])) > 0


@pytest.mark.integration
class TestDifferentBlockHash:
    def test_distinct_hashes_produce_distinct_vectors(self, server_url, model_name):
        """All artifacts from different block hashes are pairwise distinct."""
        hashes = ["0xhash_a", "0xhash_b", "0xhash_c"]
        nonces = [1, 2, 3]
        all_vectors: dict[str, list[str]] = {}

        for i, bh in enumerate(hashes):
            body = poc_request_body(bh, nonces, model_name, block_height=4000 + i)
            resp = httpx.post(f"{server_url}/api/v1/pow/generate", json=body, timeout=60)
            assert resp.status_code == 200
            artifacts = resp.json().get("artifacts", [])
            assert artifacts, f"No artifacts for {bh}"
            all_vectors[bh] = [a["vector_b64"] for a in artifacts]

        hash_list = list(hashes)
        for j in range(len(hash_list)):
            for k in range(j + 1, len(hash_list)):
                bh_a, bh_b = hash_list[j], hash_list[k]
                for n_idx in range(len(nonces)):
                    assert all_vectors[bh_a][n_idx] != all_vectors[bh_b][n_idx], (
                        f"Nonce {nonces[n_idx]}: {bh_a} and {bh_b} produced identical vector"
                    )


@pytest.mark.integration
class TestHighConcurrency:
    def test_concurrent_requests_all_succeed(self, client, server_url, model_name):
        """10 chat + 10 PoC concurrent requests all complete successfully."""
        num_chat = 10
        num_poc = 10

        def _chat(idx):
            try:
                result = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": _random_chat_prompt()}],
                    max_tokens=150,
                )
                return bool(result.choices)
            except Exception:
                return False

        def _poc(idx):
            resp = httpx.post(
                f"{server_url}/api/v1/pow/generate",
                json=poc_request_body(
                    f"0xconcurrency_{idx}",
                    [idx * 100 + j for j in range(5)],
                    model_name,
                    block_height=5000 + idx,
                ),
                timeout=60,
            )
            return resp.status_code == 200

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_chat + num_poc) as pool:
            chat_futures = [pool.submit(_chat, i) for i in range(num_chat)]
            poc_futures = [pool.submit(_poc, i) for i in range(num_poc)]
            chat_results = [f.result() for f in concurrent.futures.as_completed(chat_futures)]
            poc_results = [f.result() for f in concurrent.futures.as_completed(poc_futures)]

        chat_ok = sum(chat_results)
        poc_ok = sum(poc_results)

        assert chat_ok == num_chat, f"Chat: {chat_ok}/{num_chat} succeeded"
        assert poc_ok == num_poc, f"PoC: {poc_ok}/{num_poc} succeeded"


_LONG_PROMPT = (
    "Repeat the word 'token' exactly 150 times, separated by spaces. "
    "Do not add anything else."
)
_MAX_TOKENS = 400
_MIN_TOKEN_FRACTION = 0.80


@pytest.mark.integration
@pytest.mark.asyncio
class TestChatNotTruncatedByPoC:
    async def test_single_chat_not_truncated(self, server_url, model_name, async_client):
        """A long chat generation is not truncated by a concurrent PoC request.

        Starts a long chat, injects a PoC request mid-generation, and asserts
        the chat response still reaches at least 80 % of max_tokens.
        """
        async with httpx.AsyncClient(timeout=120) as http_client:
            chat_task = asyncio.create_task(
                async_client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": _LONG_PROMPT}],
                    max_tokens=_MAX_TOKENS,
                )
            )
            await asyncio.sleep(0.3)
            poc_task = asyncio.create_task(
                http_client.post(
                    f"{server_url}/api/v1/pow/generate",
                    json=poc_request_body(
                        "0xtruncation_single", list(range(8)), model_name
                    ),
                )
            )
            chat_result, poc_resp = await asyncio.gather(
                chat_task, poc_task, return_exceptions=True
            )

        assert not isinstance(chat_result, Exception), f"Chat raised: {chat_result}"
        assert not isinstance(poc_resp, Exception), f"PoC raised: {poc_resp}"
        assert poc_resp.status_code == 200, f"PoC HTTP {poc_resp.status_code}: {poc_resp.text}"
        assert poc_resp.json().get("status") == "completed"

        tokens = chat_result.usage.completion_tokens
        min_expected = int(_MAX_TOKENS * _MIN_TOKEN_FRACTION)
        assert tokens >= min_expected, (
            f"Chat was truncated by concurrent PoC: "
            f"{tokens} tokens received, expected >= {min_expected}"
        )

    async def test_multiple_chats_not_truncated(self, server_url, model_name, async_client):
        """Three concurrent chat requests all survive a mid-stream PoC call.

        All three must reach at least 80 % of max_tokens.
        """
        n_chat = 3
        async with httpx.AsyncClient(timeout=120) as http_client:
            chat_tasks = [
                asyncio.create_task(
                    async_client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": _LONG_PROMPT}],
                        max_tokens=_MAX_TOKENS,
                    )
                )
                for _ in range(n_chat)
            ]
            await asyncio.sleep(0.3)
            poc_task = asyncio.create_task(
                http_client.post(
                    f"{server_url}/api/v1/pow/generate",
                    json=poc_request_body(
                        "0xtruncation_multi", list(range(10, 18)), model_name
                    ),
                )
            )
            results = await asyncio.gather(*chat_tasks, poc_task, return_exceptions=True)

        poc_resp = results[-1]
        chat_results = results[:-1]

        assert not isinstance(poc_resp, Exception), f"PoC raised: {poc_resp}"
        assert poc_resp.status_code == 200
        assert poc_resp.json().get("status") == "completed"

        min_expected = int(_MAX_TOKENS * _MIN_TOKEN_FRACTION)
        for i, result in enumerate(chat_results):
            assert not isinstance(result, Exception), f"Chat {i} raised: {result}"
            tokens = result.usage.completion_tokens
            assert tokens >= min_expected, (
                f"Chat {i} truncated by concurrent PoC: "
                f"{tokens} tokens received, expected >= {min_expected}"
            )

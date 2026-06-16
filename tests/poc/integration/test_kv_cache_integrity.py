"""Integration tests verifying chat KV is not corrupted by PoC.

PoC writes KV into its own paged manager blocks (allocated on demand like chat),
so chat's KV must be untouched. Checked from the outside:
1. Reproducibility — temperature=0 chat gives identical output before/after PoC.
2. Multi-round — chat remains consistent across several PoC rounds.
"""

import pytest
import httpx

from tests.poc._server import CANONICAL_MODEL as MODEL, DEFAULT_SERVER_ARGS as SERVER_ARGS, open_poc_server
from tests.poc.utils import poc_request_body

POC_URL = "/api/v1/pow/generate"
CHAT_URL = "/v1/chat/completions"
TIMEOUT = 120

DETERMINISTIC_PROMPT = "What is the capital of France? Answer with just the city name."
MAX_CHAT_TOKENS = 20


@pytest.fixture(scope="module")
def server(request):
    with open_poc_server(request, SERVER_ARGS) as srv:
        yield srv


@pytest.fixture(scope="module")
def url(server):
    return server.url_root


@pytest.fixture(scope="module")
def client(server):
    return server.get_client()


def _chat(client, prompt: str, max_tokens: int = MAX_CHAT_TOKENS) -> str:
    """Send a deterministic (temperature=0) chat request and return the response text."""
    result = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return result.choices[0].message.content


def _poc_round(url: str, block_hash: str, nonces: list[int]) -> dict:
    """Run a PoC round synchronously in exclusive blocking mode."""
    body = poc_request_body(
        block_hash, nonces, MODEL,
        wait=True,
        blocking=True,
    )
    resp = httpx.post(f"{url}{POC_URL}", json=body, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


@pytest.mark.integration
class TestChatReproducibility:
    def test_chat_unchanged_after_single_poc_round(self, client, url):
        """Response to a deterministic prompt is identical before and after one PoC round."""
        before = _chat(client, DETERMINISTIC_PROMPT)
        poc_data = _poc_round(url, "0xkv_integrity_1", list(range(8)))
        assert poc_data["status"] == "completed"
        after = _chat(client, DETERMINISTIC_PROMPT)
        assert before == after, (
            f"Chat response changed after PoC.\n"
            f"Before: {before!r}\n"
            f"After:  {after!r}"
        )

    def test_chat_unchanged_after_multiple_poc_rounds(self, client, url):
        """Response remains identical across 5 PoC rounds."""
        baseline = _chat(client, DETERMINISTIC_PROMPT)
        for i in range(5):
            poc_data = _poc_round(url, f"0xkv_integrity_round{i}", list(range(4)))
            assert poc_data["status"] == "completed"
            current = _chat(client, DETERMINISTIC_PROMPT)
            assert baseline == current, (
                f"Chat response changed after round {i}.\n"
                f"Baseline: {baseline!r}\n"
                f"Round {i}: {current!r}"
            )

    def test_chat_works_immediately_after_poc(self, client, url):
        """Chat succeeds on the very first request after PoC completes (no lock left held)."""
        poc_data = _poc_round(url, "0xkv_immediate_after", list(range(4)))
        assert poc_data["status"] == "completed"
        response = _chat(client, "Say 'hello'.")
        assert len(response) > 0, "Chat must work immediately after PoC without delay"


@pytest.mark.integration
class TestChatQualityAfterPoC:
    def test_factual_answer_correct_after_poc(self, client, url):
        """Model gives a correct factual answer after a PoC round."""
        _poc_round(url, "0xkv_factual", list(range(6)))
        response = _chat(client, DETERMINISTIC_PROMPT)
        assert "paris" in response.lower(), (
            f"Expected 'Paris' in response after PoC, got: {response!r}"
        )

    def test_arithmetic_correct_after_poc(self, client, url):
        """Simple arithmetic is correct after a PoC round."""
        _poc_round(url, "0xkv_arithmetic", list(range(4)))
        n = 13
        response = _chat(client, f"What is {n} + {n}? Reply with just the number.")
        assert str(n + n) in response, (
            f"Expected '{n + n}' in arithmetic response after PoC, got: {response!r}"
        )

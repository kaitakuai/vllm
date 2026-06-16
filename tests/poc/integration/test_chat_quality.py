"""Tests that chat outputs are not corrupted when mixed with concurrent PoC requests."""

import concurrent.futures

import httpx
import pytest

from tests.poc._server import CANONICAL_MODEL as MODEL, DEFAULT_SERVER_ARGS as SERVER_ARGS, open_poc_server
from tests.poc.utils import poc_request_body



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


def _is_corrupted(content: str) -> tuple[bool, list[str]]:
    """Return (is_corrupted, list_of_detected_signs) for a chat response."""
    signs: list[str] = []

    if "OkayOkay" in content or content.count("Okay") > 5:
        signs.append("repeated 'Okay'")
    if content.count("[]") > 2 or content.count("()") > 3:
        signs.append("excess brackets")
    if "https" in content and "://" in content:
        signs.append("random URLs")
    if len(content.strip()) < 5:
        signs.append("too short")
    if content.count("\n") > 10:
        signs.append("excess newlines")

    alphanum_ratio = sum(c.isalnum() or c.isspace() for c in content) / max(len(content), 1)
    if alphanum_ratio < 0.7:
        signs.append(f"garbled ({alphanum_ratio:.1%} readable)")

    return bool(signs), signs


@pytest.mark.integration
class TestChatQualityUnderLoad:
    @pytest.mark.parametrize("num_tests", [20])
    def test_no_corrupted_outputs(self, client, server_url, model_name, num_tests):
        """All chat responses must be valid when PoC requests run concurrently."""

        def _chat(idx: int):
            try:
                result = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": f"What is {idx} plus {idx}?"}],
                    max_tokens=150,
                    temperature=0.5,
                )
                return idx, result.choices[0].message.content, True
            except Exception as exc:
                return idx, str(exc), False

        def _poc(idx: int):
            resp = httpx.post(
                f"{server_url}/api/v1/pow/generate",
                json=poc_request_body(
                    f"0xtest_quality_{idx}",
                    [idx * 10 + j for j in range(5)],
                    model_name,
                    block_height=1000 + idx,
                ),
                timeout=60,
            )
            return resp.status_code == 200

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_tests * 2) as pool:
            chat_futures = [pool.submit(_chat, i) for i in range(num_tests)]
            poc_futures = [pool.submit(_poc, i) for i in range(num_tests)]
            chat_results = [f.result() for f in chat_futures]
            poc_results = [f.result() for f in poc_futures]

        corrupted_indices: list[int] = []
        for idx, content, success in sorted(chat_results, key=lambda x: x[0]):
            if not success:
                corrupted_indices.append(idx)
                continue
            is_bad, _ = _is_corrupted(content)
            if is_bad:
                corrupted_indices.append(idx)

        poc_success_rate = sum(poc_results) / num_tests

        assert poc_success_rate == 1.0, (
            f"Only {sum(poc_results)}/{num_tests} PoC requests succeeded"
        )
        assert not corrupted_indices, (
            f"{len(corrupted_indices)} chat outputs were corrupted: {corrupted_indices}"
        )

    def test_math_answers_are_correct(self, client, model_name):
        """Chat responses to simple arithmetic must contain the correct answer."""
        n = 7
        result = client.chat.completions.create(
            model=model_name,
            messages=[{
                "role": "user",
                "content": f"What is {n} plus {n}? Reply with just the number.",
            }],
            max_tokens=20,
        )
        content = result.choices[0].message.content
        assert str(n + n) in content, f"Expected '{n + n}' in response: {content!r}"

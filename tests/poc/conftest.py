"""Shared pytest fixtures and configuration for PoC tests.

Server fixture strategy
-----------------------
Pass ``--poc-port PORT`` to connect to an already-running vLLM server.
Omit it to have pytest auto-launch a server via ``PoCTestServer``
(a self-contained helper defined in ``tests/poc/_server.py``).

``--poc-model`` selects the HuggingFace model (defaults to the canonical
quantized Qwen2.5-7B used across the PoC tests and benchmarks).
"""

from typing import Generator

import pytest

from tests.poc._server import (
    CANONICAL_MODEL,
    DEFAULT_SERVER_ARGS,
    open_poc_server,
)

DEFAULT_MODEL = CANONICAL_MODEL
DEFAULT_MAX_WAIT = 300


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--poc-model",
        default=DEFAULT_MODEL,
        help=f"HuggingFace model to serve (default: {DEFAULT_MODEL})",
    )
    parser.addoption(
        "--poc-port",
        type=int,
        default=None,
        help="Port of an already-running vLLM server; skips auto-launch.",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: marks tests that require a running vLLM server",
    )
    config.addinivalue_line(
        "markers",
        "gpu: marks tests that require a GPU (e.g. MLA-capable for Kimi/DeepSeek wrap check)",
    )


@pytest.fixture(scope="session")
def model_name(request: pytest.FixtureRequest) -> str:
    """HuggingFace model identifier used for the test session."""
    return request.config.getoption("--poc-model")


@pytest.fixture(scope="session")
def poc_server(request: pytest.FixtureRequest, model_name: str) -> Generator:
    """Start or connect to a vLLM server for the entire test session.

    Yields a ``PoCTestServer`` (or a compatible shim for existing servers)
    with ``url_root``, ``url_for()``, ``get_client()``, and
    ``get_async_client()`` attributes. Honors ``--poc-port`` via the shared
    ``open_poc_server`` helper.
    """
    with open_poc_server(request, DEFAULT_SERVER_ARGS, model=model_name) as srv:
        yield srv


@pytest.fixture(scope="session")
def server_url(poc_server) -> str:
    """Base URL of the running vLLM server (e.g. ``http://127.0.0.1:8100``)."""
    return poc_server.url_root


@pytest.fixture(scope="session")
def client(poc_server):
    """Synchronous OpenAI client pointed at the PoC server."""
    return poc_server.get_client()


@pytest.fixture(scope="session")
def async_client(poc_server):
    """Asynchronous OpenAI client pointed at the PoC server."""
    return poc_server.get_async_client()

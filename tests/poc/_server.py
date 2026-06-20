"""Standalone vLLM server context manager for PoC tests.

No dependencies on the parent vLLM project — only the standard library,
``httpx``, and ``openai`` (optional, for typed clients).

Usage
-----
    with PoCTestServer("RedHatAI/Qwen2.5-7B-Instruct-quantized.w8a16", ["--gpu-memory-utilization", "0.5"]) as srv:
        resp = requests.get(f"{srv.url_root}/health")
        client = srv.get_async_client()
"""

import os
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager

import httpx


def _find_free_port() -> int:
    """Return an available TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


class PoCTestServer:
    """Launch ``vllm serve`` in a subprocess and wait for it to be healthy.

    Supports the context manager protocol — the server is terminated on exit.

    vLLM stdout and stderr are written to a temporary log file. If the process
    exits unexpectedly the last 60 lines of that file are included in the
    ``RuntimeError`` so CI logs contain the actual failure reason.

    Parameters
    ----------
    model:
        HuggingFace model id or local path passed directly to ``vllm serve``.
    vllm_serve_args:
        Extra CLI arguments forwarded to ``vllm serve`` (do not include
        ``--port``; it is assigned automatically unless *port* is given).
    host:
        Interface to bind (default ``127.0.0.1``).
    port:
        Fixed port to use. If *None* a free port is chosen automatically.
    env_dict:
        Additional environment variables merged on top of the current
        environment before launching the subprocess.
    max_wait_seconds:
        How long to poll ``/health`` before raising ``RuntimeError``.
    """

    DUMMY_API_KEY = "token-abc123"

    def __init__(
        self,
        model: str,
        vllm_serve_args: list[str],
        *,
        host: str = "127.0.0.1",
        port: int | None = None,
        env_dict: dict[str, str] | None = None,
        max_wait_seconds: float = 300,
    ) -> None:
        self.model = model
        self.host = host
        self.port = port if port is not None else _find_free_port()

        env = os.environ.copy()
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        env["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
        if env_dict:
            env.update(env_dict)

        cmd = [
            "vllm", "serve", model,
            "--host", host,
            "--port", str(self.port),
            "--poc-decode",
            *vllm_serve_args,
        ]

        self._log_file = tempfile.NamedTemporaryFile(
            mode="w",
            prefix=f"vllm_server_{self.port}_",
            suffix=".log",
            delete=False,
        )
        self.log_path: str = self._log_file.name

        print(f"Launching PoCTestServer: {' '.join(cmd)}", flush=True)
        print(f"  server log: {self.log_path}", flush=True)

        self.proc: subprocess.Popen = subprocess.Popen(
            cmd,
            env=env,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
        )
        self._wait_for_server(timeout=max_wait_seconds)

    def _read_log_tail(self, n_lines: int = 60) -> str:
        """Return the last *n_lines* lines of the vLLM server log."""
        try:
            self._log_file.flush()
            with open(self.log_path) as f:
                lines = f.readlines()
            tail = lines[-n_lines:] if len(lines) > n_lines else lines
            return "".join(tail)
        except Exception:
            return "(could not read server log)"

    def _wait_for_server(self, timeout: float) -> None:
        """Poll ``/health`` until the server responds 200 or *timeout* elapses."""
        url = self.url_for("health")
        start = time.monotonic()
        while True:
            try:
                if httpx.get(url, timeout=5).status_code == 200:
                    return
            except Exception:
                pass

            if self.proc.poll() is not None:
                tail = self._read_log_tail()
                raise RuntimeError(
                    f"vLLM server process exited unexpectedly "
                    f"(code {self.proc.returncode}).\n"
                    f"Server log ({self.log_path}) — last lines:\n"
                    f"{tail}"
                )
            if time.monotonic() - start > timeout:
                tail = self._read_log_tail()
                raise RuntimeError(
                    f"Server did not become healthy within {timeout:.0f}s.\n"
                    f"Server log ({self.log_path}) — last lines:\n"
                    f"{tail}"
                )
            time.sleep(0.5)

    @property
    def url_root(self) -> str:
        """Base URL, e.g. ``http://127.0.0.1:8100``."""
        return f"http://{self.host}:{self.port}"

    def url_for(self, *parts: str) -> str:
        """Join *parts* onto the base URL with forward slashes."""
        return self.url_root + "/" + "/".join(parts)

    def get_client(self, **kwargs):
        """Return a synchronous ``openai.OpenAI`` client pointed at this server."""
        import openai

        kwargs.setdefault("timeout", 600)
        return openai.OpenAI(
            base_url=self.url_for("v1"),
            api_key=self.DUMMY_API_KEY,
            max_retries=0,
            **kwargs,
        )

    def get_async_client(self, **kwargs):
        """Return an asynchronous ``openai.AsyncOpenAI`` client pointed at this server."""
        import openai

        kwargs.setdefault("timeout", 600)
        return openai.AsyncOpenAI(
            base_url=self.url_for("v1"),
            api_key=self.DUMMY_API_KEY,
            max_retries=0,
            **kwargs,
        )

    def __enter__(self) -> "PoCTestServer":
        return self

    def __exit__(self, *args) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(8)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        try:
            self._log_file.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Shared test config + a single connect-or-launch helper
# ---------------------------------------------------------------------------
# Canonical model + server args used across ALL PoC tests and benchmarks.
CANONICAL_MODEL = "RedHatAI/Qwen2.5-7B-Instruct-quantized.w8a16"
# Memory tuning for the 20 GB test vehicle (7B-w8a16):
#  - PoC cudagraph capture allocates per-(batch_size, step) pools at RUNTIME,
#    outside vLLM's init budget, so 0.9 util OOMs during capture.
#  - PoC needs only seq_len=256 + max_tokens=256 = 512 tokens total, and chat in
#    tests is short, so --max-model-len 1024 shrinks the engine's own cudagraph/
#    activation footprint and frees GPU for cudagraph capture.
#  - util 0.8 leaves headroom for cudagraph capture alongside dynamic KV.
DEFAULT_SERVER_ARGS = ["--gpu-memory-utilization", "0.8", "--max-model-len", "1024"]


class _ExistingServer:
    """Adapter exposing the PoCTestServer interface for an already-running server."""

    DUMMY_API_KEY = PoCTestServer.DUMMY_API_KEY

    def __init__(self, url_root: str):
        self.url_root = url_root

    def url_for(self, *parts: str) -> str:
        return self.url_root + "/" + "/".join(parts)

    def get_client(self, **kwargs):
        import openai
        kwargs.setdefault("timeout", 600)
        return openai.OpenAI(base_url=self.url_for("v1"),
                             api_key=self.DUMMY_API_KEY, max_retries=0, **kwargs)

    def get_async_client(self, **kwargs):
        import openai
        kwargs.setdefault("timeout", 600)
        return openai.AsyncOpenAI(base_url=self.url_for("v1"),
                                  api_key=self.DUMMY_API_KEY, max_retries=0, **kwargs)


@contextmanager
def open_poc_server(request, server_args=None, model=None):
    """Connect to a ``--poc-port`` server if given, else launch a PoCTestServer.

    Single connect-or-launch path so every test honors ``--poc-port`` /
    ``--poc-model`` and shares one canonical config. Used by the per-module
    ``server`` fixtures across the integration suite.
    """
    import pytest

    port = request.config.getoption("--poc-port")
    model = model or request.config.getoption("--poc-model")
    if port is not None:
        base = f"http://localhost:{port}"
        try:
            resp = httpx.get(f"{base}/health", timeout=5)
            if resp.status_code != 200:
                pytest.skip(f"Server on port {port} unhealthy (HTTP {resp.status_code})")
        except Exception as exc:
            pytest.skip(f"Cannot connect to server on port {port}: {exc}")
        yield _ExistingServer(base)
    else:
        with PoCTestServer(model, server_args or DEFAULT_SERVER_ARGS) as srv:
            yield srv

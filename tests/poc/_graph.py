"""Shared CUDA-graph profiling helpers for PoC.

Boots a PoC server with the torch profiler, runs ONE PoC request under profiling,
and exposes the resulting Chrome/Perfetto-format trace. Used by:
  - tests/poc/integration/test_cudagraph_engaged.py (assert decode is graphed)
  - benchmarks/poc/graph_coverage.py (CLI count)
  - benchmarks/poc/graph_timeline.py (render a CUDA-graph timeline PNG)
"""
import glob
import gzip
import json
import os
import time

import httpx

from tests.poc._server import (
    CANONICAL_MODEL as MODEL,
    DEFAULT_SERVER_ARGS as BASE_ARGS,
    PoCTestServer,
)

POC_URL = "/api/v1/pow/generate"


def _poc_body(max_tokens: int = 8) -> dict:
    return {
        "block_hash": "deadbeef" * 8, "block_height": 100,
        "public_key": "cafebabe" * 8, "node_id": 0, "node_count": 1,
        "nonces": [1, 2],
        "params": {"model": MODEL, "seq_len": 256, "k_dim": 12,
                   "max_tokens": max_tokens},
        "wait": True,
    }


def load_trace_events(trace_path: str) -> list:
    """Return the traceEvents list from a (optionally gzipped) trace JSON."""
    op = gzip.open if trace_path.endswith(".gz") else open
    with op(trace_path, "rt") as f:
        data = json.load(f)
    return data.get("traceEvents", data) if isinstance(data, dict) else data


def count_graph_launches(trace_path: str) -> int:
    """Number of cudaGraphLaunch events in a trace."""
    return sum(1 for e in load_trace_events(trace_path)
               if isinstance(e, dict) and "cudaGraphLaunch" in str(e.get("name", "")))


def profile_poc_request(server_extra_args=None, max_tokens: int = 8,
                        prof_dir: str = "/tmp/poc_prof", cgmode: str | None = None):
    """Boot a PoC server with the torch profiler, run ONE PoC request under
    profiling, and return ``(total_cudaGraphLaunch, trace_paths)``.

    server_extra_args: extra ``vllm serve`` args (e.g. ["--enforce-eager"]).
    max_tokens: 0 => prefill-only; >0 => prefill + that many decode steps.
    cgmode: optional CUDAGraphMode override (e.g. "FULL").
    """
    extra = list(server_extra_args or [])
    if cgmode:
        extra += ["--compilation-config", json.dumps({"cudagraph_mode": cgmode})]
    os.makedirs(prof_dir, exist_ok=True)
    for f in glob.glob(f"{prof_dir}/*"):
        os.remove(f)
    # v0.20 enables the /start_profile|/stop_profile endpoints via ProfilerConfig
    # (the old VLLM_TORCH_PROFILER_DIR env path is gone); pass it explicitly.
    extra += ["--profiler-config",
              json.dumps({"profiler": "torch", "torch_profiler_dir": prof_dir})]
    with PoCTestServer(MODEL, BASE_ARGS + extra) as srv:
        url = srv.url_root
        # warmup: lazy init + first-request graph capture
        httpx.post(f"{url}{POC_URL}", json=_poc_body(max_tokens), timeout=180).raise_for_status()
        httpx.post(f"{url}/start_profile", timeout=60).raise_for_status()
        httpx.post(f"{url}{POC_URL}", json=_poc_body(max_tokens), timeout=180).raise_for_status()
        httpx.post(f"{url}/stop_profile", timeout=120).raise_for_status()
        time.sleep(8)  # let the trace flush
    traces = sorted(set(glob.glob(f"{prof_dir}/*.pt.trace.json*")
                        + glob.glob(f"{prof_dir}/*.json*")))
    total = sum(count_graph_launches(t) for t in traces)
    return total, traces

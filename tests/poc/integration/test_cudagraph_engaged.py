"""Regression: PoC decode must actually run through CUDA graphs (default mode),
and eager mode must run with none — proven from the profiler trace, not log lines.

This guards criterion 3 (cudagraph engaged + speedup): if a refactor silently
dropped PoC off the captured-graph path, decode would fall back to eager and these
counts would collapse. The trace `cudaGraphLaunch` count is the objective signal.
"""
import pytest

from tests.poc._graph import profile_poc_request


@pytest.mark.integration
def test_decode_runs_through_cudagraph():
    """Default (cudagraph) mode: a decode PoC request replays captured graphs."""
    total, traces = profile_poc_request(max_tokens=8, prof_dir="/tmp/poc_prof_cg_on")
    assert traces, "no profiler trace produced (VLLM_TORCH_PROFILER_DIR unsupported?)"
    assert total > 0, (
        f"decode PoC ran with ZERO cudaGraphLaunch — not graphed (got {total}). "
        f"PoC fell off vLLM's captured-graph path.")


@pytest.mark.integration
def test_eager_runs_without_cudagraph():
    """--enforce-eager: the same request replays NO graphs (clean baseline). Proves
    the graph in the default run is real, and that eager still works."""
    total, traces = profile_poc_request(server_extra_args=["--enforce-eager"],
                                        max_tokens=8, prof_dir="/tmp/poc_prof_cg_off")
    assert traces, "no profiler trace produced"
    assert total == 0, (
        f"eager run unexpectedly replayed {total} cudaGraphLaunch — "
        f"--enforce-eager should disable cudagraph entirely.")

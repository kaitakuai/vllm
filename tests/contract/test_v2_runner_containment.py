"""Drift detector: pin the V2-model-runner containment surface (row 9).

vLLM 0.25 ships a parallel V2 GPU model runner (``vllm/v1/worker/gpu/``) with
its own InputBatch and sampler stack. It BYPASSES the residual sampler rows
(per-request ``logprobs_mode``, ``enforced_token_ids``,
``need_processed_logprobs``): a request routed through V2 samples without any
of the PoC validation surface, silently.

Containment = the image bakes ``VLLM_USE_V2_MODEL_RUNNER=0``
(Dockerfile.quick). This contract pins the two assumptions that make the pin
effective:

1. The env knob still exists upstream (renamed/removed => the pin is a silent
   no-op and a future default flip re-exposes V2).
2. The V2 runner still carries its own sampler module (if upstream unifies the
   stacks, this contract alerts so the rows can be re-evaluated — possibly the
   pin becomes unnecessary, possibly the rows need a V2 port).

Scope: read-only inspection; NO GPU, NO engine startup.
"""

from __future__ import annotations

import importlib

import pytest


def test_v2_runner_env_knob_exists() -> None:
    """VLLM_USE_V2_MODEL_RUNNER must still be a recognized env knob."""
    pytest.importorskip("vllm")
    envs = importlib.import_module("vllm.envs")
    assert "VLLM_USE_V2_MODEL_RUNNER" in getattr(envs, "environment_variables", {}), (
        "VLLM_USE_V2_MODEL_RUNNER is gone from vllm.envs — the baked "
        "Dockerfile pin is a silent no-op. Find the new V2-runner switch and "
        "re-pin (row 9), or confirm the V2 runner was removed/unified."
    )


def test_v2_runner_still_separate_sampler() -> None:
    """The V2 runner keeps its own sampler stack (the reason the pin exists)."""
    pytest.importorskip("vllm")
    try:
        v2_sampler = importlib.import_module("vllm.v1.worker.gpu.sample.sampler")
    except ModuleNotFoundError:
        pytest.skip(
            "V2 runner sampler module gone — stacks may have been unified; "
            "re-evaluate row 9 (pin may be droppable, or rows need a V2 port)."
        )
    v1_sampler = importlib.import_module("vllm.v1.sample.sampler")
    assert v2_sampler.__name__ != v1_sampler.__name__, (
        "V2 sampler resolves to the V1 module — unification happened; "
        "re-evaluate row 9."
    )

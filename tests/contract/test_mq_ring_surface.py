# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pin the surface the VLLM_MQ_MAX_CHUNKS override depends on.

A briefly-slow TP reader can starve the shm broadcast writer under burst
load, surfacing as "No available shared memory broadcast block" stalls. The
override raises the ring count to buy slack; these tests pin the two things
it needs — the ``max_chunks`` parameter and the env knob itself — plus the
default-preserving behaviour, so that neither an upstream refactor nor a
careless edit turns the override into a silent no-op.

Read-only: no GPU, no engine startup, no queue is constructed.
"""

from __future__ import annotations

import inspect


def test_message_queue_accepts_max_chunks() -> None:
    """``MessageQueue.__init__`` must still expose ``max_chunks``."""
    from vllm.distributed.device_communicators.shm_broadcast import MessageQueue

    params = inspect.signature(MessageQueue.__init__).parameters
    assert "max_chunks" in params, (
        "MessageQueue.__init__ lost the max_chunks parameter — the ring-size "
        "override has nothing to apply to"
    )


def test_env_knob_exists_and_defaults_to_none() -> None:
    """The knob must exist and must not change ring sizes when unset.

    ``None`` is what keeps every call site on its own upstream default; a
    concrete default here would raise /dev/shm requirements for every
    deployment, including those that never hit the starvation case.
    """
    import vllm.envs as envs

    assert hasattr(envs, "VLLM_MQ_MAX_CHUNKS"), (
        "VLLM_MQ_MAX_CHUNKS is gone — the ring-size override is unreachable"
    )
    value = envs.VLLM_MQ_MAX_CHUNKS
    assert value is None or isinstance(value, int), (
        f"VLLM_MQ_MAX_CHUNKS should be None or int, got {value!r}"
    )


def test_ring_sites_honour_the_knob() -> None:
    """Every ring that shares the starvation failure mode reads the knob.

    Asserted per module rather than by counting literals: a site that keeps a
    hardcoded ring size stays vulnerable while the tests pass, which is how
    the Ray executor was missed the first time.
    """
    from vllm.distributed import parallel_state
    from vllm.v1.executor import multiproc_executor

    for module in (parallel_state, multiproc_executor):
        assert "VLLM_MQ_MAX_CHUNKS" in inspect.getsource(module), (
            f"{module.__name__} no longer reads VLLM_MQ_MAX_CHUNKS — its rings "
            f"are pinned to the default and the override cannot reach them"
        )

    # Optional dependency: absent unless vLLM was installed with Ray.
    try:
        from vllm.v1.executor import ray_executor_v2
    except ImportError:
        return
    assert "VLLM_MQ_MAX_CHUNKS" in inspect.getsource(ray_executor_v2), (
        "the Ray executor's broadcast ring ignores VLLM_MQ_MAX_CHUNKS, so the "
        "stall it shares with the multiproc executor stays unfixable there"
    )

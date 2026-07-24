"""Drift detector: pin the shm MessageQueue ring-size surface (REBASE.md row 8).

Row 8 enlarges the shm broadcast rings from their defaults (6 GroupCoordinator
chunks / 10 executor chunks) to 64: a briefly-slow TP reader starves the writer
under burst load -> "No available shared memory broadcast block" engine stalls.
Ported from the 0.20 fat-fork (kaitakuai/vllm#15).

Two modes (see ``test_sampler_surface.py`` header + REBASE.md):
    * in-fork job — runs against the residual wheel; ALL assertions MUST pass.
    * upstream-drift job — ring values fail on upstream (expected rebase ALERT);
      the signature test failing means upstream removed the knob entirely.

Scope: read-only source inspection; NO GPU, NO engine startup.
"""
from __future__ import annotations

import inspect
import re


def test_message_queue_accepts_max_chunks() -> None:
    """``MessageQueue.__init__`` must still expose ``max_chunks``."""
    from vllm.distributed.device_communicators.shm_broadcast import MessageQueue

    params = inspect.signature(MessageQueue.__init__).parameters
    assert "max_chunks" in params, (
        "MessageQueue.__init__ lost the max_chunks parameter — the row 8 "
        "ring enlargement has no surface to apply to"
    )


def test_group_coordinator_rings_enlarged() -> None:
    """All three GroupCoordinator ring sites carry max_chunks=64."""
    from vllm.distributed import parallel_state

    sites = re.findall(r"1 << 22,\s*(\d+)", inspect.getsource(parallel_state))
    assert sites == ["64", "64", "64"], (
        f"expected three '1 << 22, 64' ring sites in parallel_state, got {sites} "
        "— row 8 patch missing or upstream moved the call sites"
    )


def test_rpc_broadcast_mq_ring_enlarged() -> None:
    """The per-step executor ring (the primary stall site) carries max_chunks=64."""
    from vllm.v1.executor import multiproc_executor

    src = inspect.getsource(multiproc_executor)
    assert "rpc_broadcast_mq = MessageQueue(" in src, (
        "rpc_broadcast_mq construction moved — re-anchor the row 8 executor hunk"
    )
    assert "max_chunks=64" in src, (
        "rpc_broadcast_mq no longer passes max_chunks=64 (default 10) — row 8 "
        "executor hunk missing"
    )

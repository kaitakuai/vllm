"""Drift detector: pin the enforced_tokens / inference-validation REQUEST surface.

The SAMPLER half of enforced-token replay is pinned in ``test_sampler_surface.py``
(commits 1c5368212 / c2db96992 / f2bbeaac8). THIS file pins the REQUEST-INGESTION
half that bridges HTTP -> SamplingParams: the ``vllm.validation.EnforcedTokens``
helper and the ``ChatCompletionRequest.{enforced_tokens, enforced_str,
logprobs_mode}`` fields, plus the ``OpenAIServingChat`` glue that writes
``sampling_params.enforced_token_ids``.

Originating commit: the enforced_tokens request-ingestion patch (REBASE.md row 7) —
``feat(validation): add enforced_tokens request ingestion`` (validation.py + chat
protocol.py + chat serving.py + engine/serving.py).

Without this layer the sampler enforcement IS present but NEVER fires: a validator's
``enforced_tokens`` payload is silently dropped by Pydantic, so inference validation
does not work. A future upstream rebase that drops these hunks would reintroduce that
bug with GREEN CI unless this contract fails loudly.

Two modes (see ``test_sampler_surface.py`` header + REBASE.md):
    * in-fork job — runs against the residual wheel; ALL assertions MUST pass.
    * upstream-drift job — runs against unmodified upstream; these fail (expected),
      which is the rebase ALERT signal.

Scope: read-only module/class inspection; NO GPU, NO engine startup, NO forward pass.
"""
from __future__ import annotations

import importlib
import inspect

import pytest


# ---------------------------------------------------------------------------- #
# vllm.validation.EnforcedTokens helper (new module)
# ---------------------------------------------------------------------------- #

def test_validation_module_has_enforced_tokens() -> None:
    """Pin the ``vllm.validation`` helper classes added by the ingestion commit.

    What breaks if this assertion fails:
        - In-fork: the enforced_tokens ingestion commit (REBASE.md row 7) did not
          apply — re-add ``vllm/validation.py``.
        - Upstream-drift: upstream still has no ``vllm.validation`` module
          (expected on the drift job).
    """
    pytest.importorskip("vllm")
    mod = importlib.import_module("vllm.validation")
    for name in ("EnforcedToken", "EnforcedTokens"):
        assert getattr(mod, name, None) is not None, (
            f"vllm.validation.{name} missing — the enforced_tokens ingestion "
            f"commit (REBASE.md row 7) did not apply. Inference validation replay "
            f"is dropped at the HTTP layer (Pasha's bug)."
        )
    enforced_tokens = mod.EnforcedTokens
    for meth in (
        "encode",
        "from_content",
        "get_enforced_token_ids",
        "detect_logprobs_mode",
    ):
        assert hasattr(enforced_tokens, meth), (
            f"EnforcedTokens.{meth} missing — re-port the ingestion commit "
            f"(REBASE.md row 7)."
        )


# ---------------------------------------------------------------------------- #
# ChatCompletionRequest enforced_tokens / enforced_str / logprobs_mode fields
# ---------------------------------------------------------------------------- #

def test_chat_request_has_enforced_token_fields() -> None:
    """Pin the request fields that carry a validator's enforced-token payload.

    What breaks if this assertion fails:
        - In-fork: the request-protocol hunk did not apply; Pydantic SILENTLY
          DROPS the validator's ``enforced_tokens`` payload (the exact bug the
          ingestion commit fixes).
        - Upstream-drift: upstream ChatCompletionRequest lacks these (expected).
    """
    pytest.importorskip("vllm")
    mod = importlib.import_module(
        "vllm.entrypoints.openai.chat_completion.protocol"
    )
    cls = getattr(mod, "ChatCompletionRequest", None)
    assert cls is not None, (
        "vllm.entrypoints.openai.chat_completion.protocol.ChatCompletionRequest "
        "missing — module restructured; re-port the ingestion commit."
    )
    # Pydantic v2 model field set.
    fields = set(getattr(cls, "model_fields", {}) or {})
    if not fields:  # extreme-fallback for non-pydantic shape
        fields = set(getattr(cls, "__annotations__", {}) or {})
    for name in ("enforced_tokens", "enforced_str", "logprobs_mode"):
        assert name in fields, (
            f"ChatCompletionRequest.{name} missing from model_fields — the "
            f"enforced_tokens ingestion commit (REBASE.md row 7) did not apply. "
            f"A validator's payload is silently dropped -> inference validation "
            f"broken."
        )


# ---------------------------------------------------------------------------- #
# OpenAIServingChat bridges request fields -> sampling_params.enforced_token_ids
# ---------------------------------------------------------------------------- #

def test_chat_serving_writes_enforced_token_ids() -> None:
    """Pin that the chat serving path converts the request fields into
    ``sampling_params.enforced_token_ids``.

    What breaks if this assertion fails:
        - The serving glue (serving.py enforced block) did not apply; the request
          fields exist but never reach the sampler — enforced_tokens parsed yet
          never enforced. Re-port the ingestion commit (REBASE.md row 7).
    """
    pytest.importorskip("vllm")
    mod = importlib.import_module(
        "vllm.entrypoints.openai.chat_completion.serving"
    )
    cls = getattr(mod, "OpenAIServingChat", None)
    assert cls is not None, (
        "OpenAIServingChat missing — module restructured; re-port the ingestion "
        "commit."
    )
    src = inspect.getsource(cls)
    assert "enforced_token_ids" in src, (
        "OpenAIServingChat no longer writes sampling_params.enforced_token_ids — "
        "the serving bridge from the ingestion commit was lost. enforced_tokens "
        "would be parsed but never enforced. Re-port the ingestion commit "
        "(REBASE.md row 7)."
    )

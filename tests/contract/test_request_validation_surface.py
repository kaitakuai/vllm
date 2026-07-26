# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pin the HTTP -> SamplingParams path for enforced-token replay.

The sampler half of replay is pinned in ``test_sampler_surface.py``. This
file pins the ingestion half: the request fields, the helper that decodes
them, and the validation that keeps a replay id out of the embedding lookup
unless the model can embed it.

Both halves have to be present for replay to work, and the failure mode when
the ingestion half is missing is silent — the payload is dropped by the
request model and validation degrades into an ordinary generate, with no
error anywhere. That is what these tests exist to make loud.

Read-only: no GPU, no engine startup, no forward pass.
"""

from __future__ import annotations

import importlib

import pytest


def test_validation_module_has_enforced_tokens() -> None:
    """The helper that turns a replay payload into token ids must exist."""
    pytest.importorskip("vllm")
    mod = importlib.import_module("vllm.validation")
    for name in ("EnforcedToken", "EnforcedTokens"):
        assert getattr(mod, name, None) is not None, (
            f"vllm.validation.{name} is missing, so a replay payload cannot be "
            f"decoded and inference validation silently degrades to a plain "
            f"generate"
        )
    for meth in ("encode", "get_enforced_token_ids"):
        assert hasattr(mod.EnforcedTokens, meth), (
            f"EnforcedTokens.{meth} is missing — the ingestion path cannot "
            f"produce token ids"
        )


def test_replay_ids_are_range_checked() -> None:
    """Out-of-range replay ids must be rejected before they reach the engine.

    This is the difference between a 400 and a dead worker: an id outside the
    vocabulary reaches an embedding lookup, where it is a device-side assert
    that takes down the process and every other request on it.
    """
    pytest.importorskip("vllm")
    mod = importlib.import_module("vllm.validation")
    validate = getattr(mod, "validate_enforced_token_ids", None)
    assert validate is not None, (
        "vllm.validation.validate_enforced_token_ids is missing — replay ids "
        "would reach the embedding lookup unchecked"
    )

    vocab_size = 32000
    validate([0, 1, vocab_size - 1], vocab_size)  # in range: must not raise

    for bad in ([vocab_size], [-1], [-5]):
        with pytest.raises(ValueError):
            validate(bad, vocab_size)


def test_replay_payload_is_size_bounded() -> None:
    """The payload must have ceilings, so a large body cannot exhaust memory."""
    pytest.importorskip("vllm")
    mod = importlib.import_module("vllm.validation")
    for name in ("MAX_ENFORCED_TOKENS", "MAX_TOP_TOKENS_PER_POSITION"):
        limit = getattr(mod, name, None)
        assert isinstance(limit, int) and limit > 0, (
            f"vllm.validation.{name} is missing or not a positive int — the "
            f"replay payload is unbounded"
        )

    with pytest.raises(ValueError):
        mod.EnforcedTokens(
            tokens=[{"token": "1"}] * (mod.MAX_ENFORCED_TOKENS + 1),
        )


def test_chat_request_carries_replay_fields() -> None:
    """The request model must accept the replay fields.

    If it does not, the payload is dropped during request parsing and the
    validator gets an ordinary completion back while believing it verified
    something.
    """
    pytest.importorskip("vllm")
    mod = importlib.import_module("vllm.entrypoints.openai.chat_completion.protocol")
    cls = getattr(mod, "ChatCompletionRequest", None)
    assert cls is not None, (
        "ChatCompletionRequest is missing — the request protocol module was "
        "restructured"
    )
    fields = set(getattr(cls, "model_fields", {}) or {})
    if not fields:
        fields = set(getattr(cls, "__annotations__", {}) or {})
    for name in ("enforced_tokens", "enforced_str", "logprobs_mode"):
        assert name in fields, (
            f"ChatCompletionRequest.{name} is missing, so that part of a "
            f"replay request is silently discarded"
        )


def test_sampling_params_carries_replay_ids() -> None:
    """The field the ingestion path writes into must exist on SamplingParams.

    Checked on the destination type rather than by reading the serving source,
    so the test tracks the contract instead of a particular implementation.
    """
    pytest.importorskip("vllm")
    mod = importlib.import_module("vllm.sampling_params")
    cls = mod.SamplingParams
    fields = set(getattr(cls, "__struct_fields__", ()) or ()) or set(
        getattr(cls, "__annotations__", {}) or {}
    )
    assert "enforced_token_ids" in fields, (
        "SamplingParams.enforced_token_ids is missing — the ingestion path has "
        "nowhere to write decoded replay ids, so enforcement never fires"
    )

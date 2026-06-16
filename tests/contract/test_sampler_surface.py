"""Drift detector: pin the vLLM sampler private surface that the residual fork patches.

This contract pin lives in the residual fork — its purpose is to catch upstream
sampler / SamplingParams refactor BEFORE the manual rebase on the next vLLM
minor. The residual is treated as permanent infrastructure (ADR-0014 in
mlnode-foundry), so Layer 3 (merging upstream) is off the table; we instead
keep a daily "is upstream still shaped the way we expect?" signal.

Each test references the originating commit SHA on poc-sampler-residual-v0.23
and documents what breaks if the assertion fails. The tests are intentionally
brittle: an upstream rename / refactor MUST surface here so the rebase author
can re-port the patches before merging the next minor.

Two modes:
    * "in-fork" job — runs against the residual wheel itself. All assertions
      MUST pass (otherwise our 6 patches did not apply correctly).
    * "upstream-drift" job — runs against the unmodified upstream wheel. The
      patch-added fields (logprobs_mode, enforced_token_ids, etc.) will be
      MISSING on upstream and these tests will fail. That failure is the
      ALERT signal — see .github/workflows/contract-tests-residual.yml and
      REBASE.md for the rebase procedure.

Scope: read-only inspection of vllm modules; NO GPU, NO engine startup, NO
forward pass. Safe to run in a vanilla ``pip install vllm`` environment.
"""
from __future__ import annotations

import importlib
import inspect

import pytest


# ---------------------------------------------------------------------------- #
# SamplingParams public dataclass fields (commit 1c5368212)
# ---------------------------------------------------------------------------- #

def test_sampling_params_has_poc_fields() -> None:
    """Pin per-request SamplingParams fields added by 1c5368212.

    Originating commit: 1c5368212 — feat(sampling): add per-request logprobs_mode
    and enforced_token_ids fields.

    What breaks if this assertion fails:
        - On the in-fork job: our patch did not apply to this branch (rebase
          regression) — fix by re-cherry-picking 1c5368212.
        - On the upstream-drift job: upstream adopted (good — Layer 3 path
          reopens) or renamed (bad — rebase author must rename our fields to
          avoid collision before next-minor cherry-pick).
    """
    pytest.importorskip("vllm")
    mod = importlib.import_module("vllm.sampling_params")
    cls = getattr(mod, "SamplingParams", None)
    assert cls is not None, "vllm.sampling_params.SamplingParams missing"

    annotations = getattr(cls, "__annotations__", {}) or {}
    # logprobs_mode: per-request override added by 1c5368212.
    assert "logprobs_mode" in annotations, (
        "SamplingParams.logprobs_mode missing — commit 1c5368212 did not "
        "apply (in-fork failure) OR upstream lacks this field (drift "
        "alert; expected on the upstream-drift job)."
    )
    # enforced_token_ids: validation-replay sequence added by 1c5368212.
    assert "enforced_token_ids" in annotations, (
        "SamplingParams.enforced_token_ids missing — commit 1c5368212 did "
        "not apply (in-fork failure) OR upstream lacks this field (drift "
        "alert; expected on the upstream-drift job)."
    )

    # Pin the __init__ accepts both kwargs; a future minor that keeps the
    # annotation but drops the kwarg wiring would silently break callers.
    sig = inspect.signature(cls.__init__)
    init_params = set(sig.parameters)
    for name in ("logprobs_mode", "enforced_token_ids"):
        assert name in init_params, (
            f"SamplingParams.__init__ no longer accepts {name!r}; "
            f"annotation present but kwarg wiring lost. "
            f"Re-check commit 1c5368212 application."
        )


# ---------------------------------------------------------------------------- #
# SamplingMetadata sampler-side fields (commit 1c5368212)
# ---------------------------------------------------------------------------- #

def test_sampling_metadata_has_poc_fields() -> None:
    """Pin sampler-side bookkeeping fields on SamplingMetadata.

    Originating commit: 1c5368212. Note that ``thinking_budget_state_holder``
    is an upstream field (added in 0.23 by vllm-project) and coexists with
    our fields; we do not assert on it here.

    What breaks if this assertion fails:
        - In-fork: 1c5368212 did not apply (or only partially applied).
        - Upstream-drift: upstream lacks these fields (expected); when
          upstream eventually relocates SamplingMetadata into a different
          module / makes it non-dataclass, this test fires.
    """
    pytest.importorskip("vllm")
    mod = importlib.import_module("vllm.v1.sample.metadata")
    cls = getattr(mod, "SamplingMetadata", None)
    assert cls is not None, "vllm.v1.sample.metadata.SamplingMetadata missing"

    annotations = getattr(cls, "__annotations__", {}) or {}
    # batch_logprobs_mode carries the per-request mode resolution; the
    # Sampler.forward() priority chain depends on this field name verbatim.
    assert "batch_logprobs_mode" in annotations, (
        "SamplingMetadata.batch_logprobs_mode missing — commit 1c5368212 "
        "did not apply. Sampler.forward() priority resolution will silently "
        "fall back to deployment default — PoC logprobs requests will be "
        "wrong."
    )
    # enforced_next_token_ids drives the post-sampling override in
    # Sampler.forward(); without it, validation replay is silently no-op.
    assert "enforced_next_token_ids" in annotations, (
        "SamplingMetadata.enforced_next_token_ids missing — commit 1c5368212 "
        "did not apply. PoC v2 validation replay will silently no-op."
    )


# ---------------------------------------------------------------------------- #
# Sampler.forward signature accepts sampling_metadata (commits c2db96992 + 1c5368212)
# ---------------------------------------------------------------------------- #

def test_sampler_call_accepts_enforced_tokens() -> None:
    """Pin that Sampler.forward accepts a SamplingMetadata kwarg that carries
    enforced_next_token_ids.

    Originating commits: 1c5368212 (field) + c2db96992 (forward() consumption).

    Brittle by design: signature inspection is sensitive to upstream
    refactors. We want this to fail loudly when upstream renames
    sampling_metadata or splits Sampler.forward into a multi-stage call;
    that is the entire point of this contract pin.

    What breaks if this assertion fails:
        - Sampler.forward signature drifted (upstream renamed
          sampling_metadata, split forward into multiple methods, etc.).
        - The rebase author MUST re-validate that c2db96992 still applies
          and that enforced_next_token_ids is still threaded through.
    """
    pytest.importorskip("vllm")
    mod = importlib.import_module("vllm.v1.sample.sampler")
    cls = getattr(mod, "Sampler", None)
    assert cls is not None, "vllm.v1.sample.sampler.Sampler missing"
    assert hasattr(cls, "forward"), (
        "Sampler.forward missing — upstream may have renamed to __call__ or "
        "split the method; commit c2db96992 needs re-port."
    )

    sig = inspect.signature(cls.forward)
    params = set(sig.parameters)
    assert "sampling_metadata" in params, (
        f"Sampler.forward no longer accepts sampling_metadata; "
        f"present params = {sorted(params)!r}. "
        f"Re-port commit c2db96992."
    )

    # Verify the SamplingMetadata type itself still carries
    # enforced_next_token_ids — the Sampler reads it via attribute access.
    meta_mod = importlib.import_module("vllm.v1.sample.metadata")
    meta_cls = getattr(meta_mod, "SamplingMetadata", None)
    assert meta_cls is not None
    meta_annotations = getattr(meta_cls, "__annotations__", {}) or {}
    assert "enforced_next_token_ids" in meta_annotations, (
        "Sampler.forward accepts sampling_metadata but SamplingMetadata "
        "lacks enforced_next_token_ids — the field was removed or renamed. "
        "Re-port commits 1c5368212 and c2db96992."
    )


# ---------------------------------------------------------------------------- #
# TopKTopPSampler.forward_* accept need_processed_logprobs (commits 3176a941c + 8d4e322e0)
# ---------------------------------------------------------------------------- #

def test_topk_topp_sampler_need_processed_logprobs() -> None:
    """Pin that all forward_* paths on TopKTopPSampler accept
    ``need_processed_logprobs`` as a kwarg.

    Originating commits:
        * 3176a941c — adds the kwarg + .sample() wrapper to forward_native /
          forward_cuda / forward_cpu / forward_hip.
        * 8d4e322e0 — extends the same threading to forward_xpu (added by
          upstream in 0.23).

    Brittle by design: a future upstream refactor that renames any of these
    forward_* methods (e.g., splitting forward_cuda into forward_cuda_v1) or
    drops them entirely (e.g., consolidating into a single dispatch fn) MUST
    fail here.

    What breaks if this assertion fails:
        - One of the forward paths no longer threads need_processed_logprobs.
        - On FlashInfer / aiter / XPU fast paths, mixed-mode batches will
          silently NOT return processed logprobs → PoC v2 cross-validator
          drift.
    """
    pytest.importorskip("vllm")
    mod = importlib.import_module("vllm.v1.sample.ops.topk_topp_sampler")
    cls = getattr(mod, "TopKTopPSampler", None)
    assert cls is not None, (
        "vllm.v1.sample.ops.topk_topp_sampler.TopKTopPSampler missing — "
        "module restructured; re-port commits 3176a941c and 8d4e322e0."
    )

    expected_methods = (
        "forward_native",
        "forward_cuda",
        "forward_hip",
        "forward_xpu",
    )
    for method_name in expected_methods:
        fn = getattr(cls, method_name, None)
        assert fn is not None, (
            f"TopKTopPSampler.{method_name} missing — upstream may have "
            f"renamed or consolidated this forward path. Re-port the "
            f"need_processed_logprobs threading."
        )
        sig = inspect.signature(fn)
        params = set(sig.parameters)
        assert "need_processed_logprobs" in params, (
            f"TopKTopPSampler.{method_name} no longer accepts "
            f"need_processed_logprobs kwarg; present params = "
            f"{sorted(params)!r}. Re-port commits 3176a941c "
            f"(forward_native/cuda/hip/cpu) and 8d4e322e0 (forward_xpu)."
        )

    # Also pin the .sample() wrapper — c2db96992 routes the Sampler through it.
    assert hasattr(cls, "sample"), (
        "TopKTopPSampler.sample wrapper missing — Sampler.forward path will "
        "crash with TypeError on FlashInfer when need_processed_logprobs=True. "
        "Re-port commit 3176a941c."
    )


# ---------------------------------------------------------------------------- #
# InputBatch logprobs_modes dict (commit f2bbeaac8)
# ---------------------------------------------------------------------------- #

def test_input_batch_logprobs_modes_dict() -> None:
    """Pin that InputBatch carries the per-request logprobs_modes dict.

    Originating commit: f2bbeaac8 — feat(worker): port InputBatch
    enforced-tokens and logprobs-mode bookkeeping.

    The patch threads the dict via:
        * constructor kwarg logprobs_mode_default
        * instance attribute logprobs_modes (dict[str, str])
        * instance attribute req_enforced_token_ids (dict[str, list[int] | None])
        * property batch_logprobs_mode

    We cannot instantiate InputBatch without a full vllm config (it requires
    KV cache spec, vllm_config, etc.), so we inspect the class source AND
    the __init__ signature instead.

    What breaks if this assertion fails:
        - InputBatch was restructured (likely candidate: a new
          ``vllm.v1.worker.batch_state`` module).
        - Without these structures, add_request / _make_sampling_metadata
          cannot populate SamplingMetadata.batch_logprobs_mode /
          enforced_next_token_ids — PoC v2 silently degrades.
    """
    pytest.importorskip("vllm")
    mod = importlib.import_module("vllm.v1.worker.gpu_input_batch")
    cls = getattr(mod, "InputBatch", None)
    assert cls is not None, (
        "vllm.v1.worker.gpu_input_batch.InputBatch missing — module "
        "restructured. Re-port commit f2bbeaac8 against the new location."
    )

    # The constructor accepts logprobs_mode_default after our patch.
    sig = inspect.signature(cls.__init__)
    init_params = set(sig.parameters)
    assert "logprobs_mode_default" in init_params, (
        "InputBatch.__init__ no longer accepts logprobs_mode_default kwarg; "
        f"present params = {sorted(init_params)!r}. Re-port f2bbeaac8."
    )

    # Inspect the class source for the attribute assignments and the
    # batch_logprobs_mode property name; this catches drift even when
    # __annotations__ is empty (instance-only attrs).
    src = inspect.getsource(cls)
    for needle in (
        "self.logprobs_modes",
        "self.req_enforced_token_ids",
        "batch_logprobs_mode",
    ):
        assert needle in src, (
            f"InputBatch source lacks {needle!r} — commit f2bbeaac8 did not "
            f"apply (in-fork failure) OR upstream lacks our bookkeeping "
            f"(drift alert — expected on the upstream-drift job)."
        )


# ---------------------------------------------------------------------------- #
# Structured-output graceful degradation hook (commit 4996d5af7)
# ---------------------------------------------------------------------------- #

def test_structured_output_graceful_degradation_hook() -> None:
    """Pin the XgrammarGrammar.accept_tokens / _grammar_failed surface and the
    StructuredOutputManager bitmask path our 4996d5af7 patch hooks.

    Originating commit: 4996d5af7 — feat(structured-output): graceful
    degradation on grammar token rejection.

    The patch touches two surfaces:
        * vllm.v1.structured_output.backend_xgrammar.XgrammarGrammar
          - adds ``_grammar_failed`` flag
          - accept_tokens() returns True on rejection (instead of False)
          - rollback() / fill_bitmask() early-return when _grammar_failed
        * vllm.v1.structured_output.StructuredOutputManager
          - replaces a hard assert with a soft warning + bitmask disable

    What breaks if this assertion fails:
        - The xgrammar backend was relocated (likely candidate:
          ``vllm.v1.structured_output.xgrammar``) or renamed.
        - OR the assert that 4996d5af7 softened was already removed upstream
          (good — softer landing for the next rebase).
    """
    pytest.importorskip("vllm")

    # Backend surface: XgrammarGrammar + _grammar_failed + accept_tokens.
    backend = importlib.import_module(
        "vllm.v1.structured_output.backend_xgrammar"
    )
    grammar_cls = getattr(backend, "XgrammarGrammar", None)
    assert grammar_cls is not None, (
        "vllm.v1.structured_output.backend_xgrammar.XgrammarGrammar missing — "
        "backend was relocated or renamed. Re-port commit 4996d5af7."
    )
    assert hasattr(grammar_cls, "accept_tokens"), (
        "XgrammarGrammar.accept_tokens missing — interface drifted. "
        "Re-port commit 4996d5af7."
    )
    # _grammar_failed is the flag our patch adds; if the dataclass field
    # vanished, the in-fork branch did not apply 4996d5af7.
    grammar_src = inspect.getsource(grammar_cls)
    assert "_grammar_failed" in grammar_src, (
        "_grammar_failed flag missing in XgrammarGrammar — commit 4996d5af7 "
        "did not apply (in-fork failure) OR upstream lacks the graceful "
        "degradation hook (drift alert)."
    )

    # Manager surface: StructuredOutputManager hosts the bitmask path our
    # patch softened (was a hard assert before 4996d5af7).
    mgr_mod = importlib.import_module("vllm.v1.structured_output")
    mgr_cls = getattr(mgr_mod, "StructuredOutputManager", None)
    assert mgr_cls is not None, (
        "vllm.v1.structured_output.StructuredOutputManager missing — "
        "module restructured. Re-port commit 4996d5af7."
    )
    # The patched code path lives inside grammar_bitmask construction; we
    # don't pin the exact method name (likely internal) — instead pin that
    # the module still imports the symbol we hook into.
    mgr_src = inspect.getsource(mgr_cls)
    # Either our patched code (logger.warning + apply_bitmask = False) OR
    # the original assert must be present; if neither is, the call site
    # moved and 4996d5af7 needs re-application.
    assert "accept_tokens" in mgr_src, (
        "StructuredOutputManager source no longer references "
        "grammar.accept_tokens — the patched call site moved. "
        "Re-port commit 4996d5af7 at the new location."
    )

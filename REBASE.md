# Rebase procedure — kaitakuai sampler-stack residual fork

This is a permanent thin fork of `vllm-project/vllm`, owned by Kaitaku.
ADR-0014 in `mlnode-foundry/docs/adr/0014-residual-fork-permanent-infra.md`
treats this branch as permanent infrastructure (the Layer 3 upstream pathway
is deferred without timeline). Each new vLLM minor is rebased mechanically
by cherry-picking the same commit stack (6 sampler + 1 request-ingestion) onto the upstream tag.

## When upstream cuts `vM.N`

1. Fetch upstream and the tag.

   ```bash
   git fetch upstream
   git fetch upstream refs/tags/vM.N:refs/tags/vM.N
   ```

2. Cut the new residual branch off the upstream tag.

   ```bash
   git checkout -b poc-sampler-residual-vM.N vM.N
   ```

3. Cherry-pick the 7 SHAs (6 sampler + row 7 request-ingestion) in chronological
   order from the most-recent residual branch (`poc-sampler-residual-v<prev>`):

   | # | SHA on v0.23 branch | Subject |
   |---|---------------------|---------|
   | 1 | `1c5368212` | feat(sampling): add per-request `logprobs_mode` and `enforced_token_ids` fields |
   | 2 | `3176a941c` | feat(sampler): add `need_processed_logprobs` and `.sample()` wrapper |
   | 3 | `c2db96992` | feat(sampler): port PoC v2 mixed-mode sampling and enforced tokens |
   | 4 | `f2bbeaac8` | feat(worker): port `InputBatch` enforced-tokens and logprobs-mode bookkeeping |
   | 5 | `4996d5af7` | feat(structured-output): graceful degradation on grammar token rejection |
   | 6 | `8d4e322e0` | fix(sampler): thread `need_processed_logprobs` through `forward_xpu` |
   | 7 | `e41e9e606` | feat(validation): add enforced_tokens request ingestion (HTTP -> SamplingParams) |

   Rows 1-6 are the **sampler stack** (private `vllm.v1.*` surfaces). Row 7 is the
   **request-ingestion layer** added 2026-06-25: `vllm/validation.py`
   (`EnforcedTokens` helper) + the `ChatCompletionRequest.{enforced_tokens,
   enforced_str, logprobs_mode}` fields and the `OpenAIServingChat` glue that writes
   `sampling_params.enforced_token_ids`. Without row 7 the sampler enforcement
   (rows 1-6) is present but never fires — a validator's payload is silently dropped
   by Pydantic and inference validation does not work. Its surfaces are pinned by
   `tests/contract/test_request_validation_surface.py` (rows 1-6 by
   `test_sampler_surface.py`). NOTE: row 7 touches the OpenAI entrypoint layer
   (`vllm/entrypoints/openai/**`), which churns faster upstream than the sampler
   internals — expect its hunks to be the most rebase-conflict-prone.

   ```bash
   git cherry-pick 1c5368212 3176a941c c2db96992 f2bbeaac8 4996d5af7 8d4e322e0 e41e9e606
   ```

   > **TODO (future rebase):** these SHAs are the commit IDs on
   > `poc-sampler-residual-v0.23`. After the first cherry-pick onto
   > `poc-sampler-residual-v0.24`, the SHAs will be new — record them
   > here for the v0.25 rebase. Use `git log --grep` to find the
   > corresponding commits on the prior residual branch if SHAs are
   > forgotten:
   >
   > ```bash
   > git log --oneline --grep='feat(sampling): add per-request logprobs_mode' poc-sampler-residual-v<prev>
   > git log --oneline --grep='feat(sampler): add need_processed_logprobs' poc-sampler-residual-v<prev>
   > git log --oneline --grep='feat(sampler): port PoC v2 mixed-mode' poc-sampler-residual-v<prev>
   > git log --oneline --grep='feat(worker): port InputBatch' poc-sampler-residual-v<prev>
   > git log --oneline --grep='feat(structured-output): graceful degradation' poc-sampler-residual-v<prev>
   > git log --oneline --grep='fix(sampler): thread need_processed_logprobs through forward_xpu' poc-sampler-residual-v<prev>
   > git log --oneline --grep='feat(validation): add enforced_tokens request ingestion' poc-sampler-residual-v<prev>
   > ```

4. Update `setup.py` `get_vllm_version()` to bump the local-version
   identifier:

   ```python
   KAITAKUAI_DEFAULT_VERSION = "M.N.0+gonka.sampler1"
   ```

5. Commit the bump.

   ```bash
   git commit -am "chore: tag as M.N.0+gonka.sampler1"
   ```

6. Inspect `Dockerfile.quick` and bump the base image:

   ```dockerfile
   FROM vllm/vllm-openai:vM.N.0-cu129
   ```

   Commit that bump together with any version-string mirrors in
   `.github/workflows/build-sampler-residual.yml`.

7. Push the new residual branch.

   ```bash
   git push -u origin poc-sampler-residual-vM.N
   ```

8. CI workflow `contract-tests-residual.yml` runs on the push.
   The **in-fork** job MUST be green — that confirms all 6 patches still
   apply cleanly and the patched surfaces are still in place.

9. The **upstream-drift** job (`continue-on-error: true`) MAY fail.
   If it does, document the NEW upstream signatures in
   `tests/contract/test_sampler_surface.py` BEFORE merging:
   * If upstream renamed a field, update the contract assertion so it
     accepts either name (or pin only the new name if upstream is now
     the source of truth for that field).
   * If upstream removed a hook our patch depended on, the patch itself
     needs revision — see "What if a cherry-pick conflicts" below.

10. Update foundry to consume the new image. The residual image digest
    is pinned in `mlnode-foundry/stage2.lock.cue` (or its successor —
    wiring lands in the next step of this rollout). Bump:

    ```cue
    sampler_residual_image: "ghcr.io/kaitakuai/vllm-sampler-residual@sha256:<new-digest>"
    ```

## What if a cherry-pick conflicts

The 6 commits touch only narrow surfaces:

* `vllm/sampling_params.py` — pure additions to a dataclass
* `vllm/v1/sample/metadata.py` — pure additions to a dataclass
* `vllm/v1/sample/sampler.py` — `forward()` body changes
* `vllm/v1/sample/ops/topk_topp_sampler.py` — `forward_*` kwarg threading
* `vllm/v1/worker/gpu_input_batch.py` — new dict attrs + property
* `vllm/v1/structured_output/__init__.py` + `backend_xgrammar.py` — assert → warn

### Rule 1: keep both adjacent additions

When upstream adds a field next to one of ours (e.g.,
`thinking_budget_state_holder` next to `enforced_next_token_ids`), the
default cherry-pick will conflict because both additions touch the same
hunk anchor. Resolve by KEEPING BOTH additions — they are independent.

### Escalation when the conflict is non-trivial

If upstream restructured the touched function (e.g., split
`Sampler.forward` into multiple methods, moved `InputBatch.add_request`
into a new module), do NOT fight the conflict line-by-line:

1. Abort the cherry-pick: `git cherry-pick --abort`.
2. Open a draft PR on `gonka-poc` (or the foundry consumer repo) that
   describes the upstream restructure and propose the re-port.
3. Use the gonka-poc PR review process to land the re-port — pair-review
   with `@baychak`, run the full PoC v2 cross-validator harness, and
   only then resume the cherry-pick of the remaining commits.
4. Update this REBASE.md with a note about the restructure so the next
   minor's rebase author is forewarned.

### When the contract tests fail in-fork

If `tests/contract/test_sampler_surface.py` fails on the in-fork job after
a cherry-pick:

* Check `git status` — did the cherry-pick produce conflict markers that
  were committed by mistake?
* Re-run the failed assertion locally with `pytest tests/contract -v -k
  <name>` and inspect the actual annotation set / signature.
* If a patch silently skipped a hunk (e.g., `git apply --reject` was
  used during conflict resolution), re-apply by hand.

## Owner

Kaitaku ML Node team — `@baychak`.

## Renewal cadence

Per vLLM minor — historically ~3 weeks between minors (v0.20 → v0.21 →
v0.22 → v0.23 over 9 weeks). Schedule the rebase within 1 week of the
upstream tag landing so the foundry consumer never lags by more than two
minors.

# vllm-poc (Stage 2 / S2 image)

`ghcr.io/kaitakuai/vllm-poc` — the runnable Gonka Proof-of-Compute v2 image.

## What it is

S2 is a thin overlay on top of the **S1 sampler residual** base:

```
ghcr.io/kaitakuai/vllm-sampler-residual:0.23.0-gonka.sampler1   (S1: patched vLLM 0.23.0)
  + pip install git+https://github.com/kaitakuai/gonka-poc@<ref>  (out-of-tree PoC plugin)
  = ghcr.io/kaitakuai/vllm-poc                                    (S2: runnable PoC image)
```

The S1 residual carries the 6-commit sampler stack (enforced-token sampling,
`logprobs_mode`, etc.) as a patched `vllm==0.23.0+gonka.sampler1` wheel. S2
adds the `gonka-poc` plugin, which registers via vLLM's `vllm.general_plugins`
entry point and installs the `gonka-vllm-serve` console script.

The plugin is installed with `pip install --no-deps`: the residual base
already provides the patched vLLM wheel and every gonka-poc runtime
dependency (see "Dependency audit" below). `--no-deps` prevents pip from
pulling a vanilla vLLM wheel over the residual patches.

## Build-args

| Arg | Required | Default | Notes |
|-----|----------|---------|-------|
| `RESIDUAL_IMAGE` | yes (effectively) | `ghcr.io/kaitakuai/vllm-sampler-residual:0.23.0-gonka.sampler1` | The S1 base. **Pin to a digest** (`...@sha256:...`) in the publishing workflow for reproducibility; the mutable tag is only a convenience default. |
| `GONKA_POC_REF` | yes | _none_ | gonka-poc git ref. **Full SHA preferred**; a tag/branch also works. The workflow resolves it to a SHA and uses the first 9 chars in the immutable image tag. |

## Tags

| Tag | Mutability | Meaning |
|-----|------------|---------|
| `0.23.0` | mutable | branch HEAD; moves on every build |
| `0.23.0-<gonka_poc_sha9>` | immutable | pins the exact gonka-poc plugin commit |

Both tags resolve to the same digest per build. cosign signs the **digest**
(keyless, recorded in Rekor); SLSA provenance + an SPDX SBOM are attached as
attestations.

## Dependency audit (gonka-poc on top of the residual base)

gonka-poc's declared runtime deps (`pyproject.toml`):
`vllm>=0.23.0,<0.24`, `torch`, `numpy`, `scipy>=1.10`, `aiohttp>=3.9`,
`fastapi<0.137`, `pydantic>=2`, `starlette`.

On the current S1 residual base **every one is already satisfied**:

| Dep | Provided by |
|-----|-------------|
| `vllm` (patched), `torch`, `numpy` | residual base wheel |
| `scipy>=1.10` | residual `Dockerfile.quick` (`pip install scipy`) |
| `fastapi<0.137` | base vLLM `requirements/common.txt`: `fastapi[standard] >=0.115.0,<0.137` |
| `aiohttp>=3.9` | base vLLM common: `aiohttp >= 3.13.3` |
| `pydantic>=2` | base vLLM common: `pydantic >= 2.12.0` |
| `starlette` | transitively via `fastapi[standard]` in base |

So the second `pip install` layer installs **nothing new today**. It is kept
as a fail-loud floor-version pin in case a future residual base drops one of
these. No additional, base-absent gonka-poc dependency was found.

## How to run

`gonka-vllm-serve` accepts every flag stock `vllm serve` accepts, plus it
wires in the PoC router + gating. Per the gonka-poc README, these MUST be set:

- env `VLLM_ALLOW_INSECURE_SERIALIZATION=1` (baked into this image)
- `--worker-extension-cls gonka_poc.worker.PoCWorkerExtension` (operator MUST
  pass explicitly — the wrapper does not auto-inject it)
- `--attention-backend FLASHINFER` (or `TRITON_ATTN`)
- `--logprobs-mode processed_logprobs`
- `--enforce-eager` (compiled drift breaks cross-validator bit-compat)

## Build + push (USER-REQUIRED, outward)

**This image is not built by anyone else.** The publishing workflow
(`build-vllm-poc.yml`) pushes to a public GHCR repo (`ghcr.io/kaitakuai/...`)
and signs with cosign. Per org policy, pushing content outward to public
registries requires explicit confirmation. **The user must run the build and
push it** — this draft does not do so. Trigger via `workflow_dispatch`
(pinning `residual_image` to a digest and `gonka_poc_ref` to a SHA) or by
pushing `Dockerfile.vllm-poc` to the `residual/vllm-poc` branch.

## Downstream: foundry stage3.lock.cue

The resulting **digest** (`ghcr.io/kaitakuai/vllm-poc@sha256:...`) feeds
foundry `stage3.lock.cue` field `stage2.image`.

> **Blocker / flag:** the foundry `stage2.image` schema regex currently does
> NOT allow a `vllm-poc` image reference (it was written for the prior image
> naming). The regex must be **relaxed to allow `vllm-poc`** before this
> digest can be locked in. Coordinate the foundry schema change with whoever
> owns `stage3.lock.cue` / the foundry CUE schema.

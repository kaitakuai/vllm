# What's in a decode-PoC session folder

A session folder is `benchmarks/poc/scope/reports/<session>/`, `<session>` =
`<honest-model-slug>__<gpu-name>__<timestamp>`. It's the working directory `run_scope.sh`
writes into, and — unmodified — exactly what gets mirrored to S3 with `--push`. Exact contents
depend on the flags (`--mla`, `--no-fi`, `--no-perf`, `--no-gsm`, `--xhw`).

## Always present
| File | What it is |
|---|---|
| `report.html` | the rendered report ([scope/simplify_report.py](scope/simplify_report.py)) |
| `REPRODUCE.md` | exact repro command + params / commit / GPU |
| `serve_<model-slug>_<profile>.log` | one per server boot (honest × profiles, fraud × its ref profiles) |

## Performance (skipped by `--no-perf`)
- `perf_<profile>.poc.json` + `perf_<profile>.chat.json` per perf profile (up to `cg-flashattn`,
  `cg-flashinfer`, `eager-flashattn`, `eager-flashinfer`; fewer under `--mla`/`--no-fi`) —
  [perfomance_nonces.py](perfomance_nonces.py) `--mode both`.

## Trajectories (generate — [collect.py](collect.py) `--mode generate`)
- `gen_honest_<profile>.json` — one per honest-reference profile (incl. the validator baseline).
- `gen_fraud_<profile>.json` — one per fraud-reference profile.

## Validations ([collect.py](collect.py) `--mode validate`)
- `val_<validator-profile>__gen_<honest|fraud>_<profile>.json` — one per same-HW check (typically
  5: floor, cross-backend honest, cross-engine honest, fraud, fraud cross-backend).
- with `--xhw`: `val_<VAL>__xhw_<peerGPUslug>_<honest|fraud>_<profile>.json` per peer trajectory.

## Co-existence / GSM8K (skipped by `--no-gsm` — [quality_gsm8k.py](quality_gsm8k.py))
- `gsm_<VAL>_on.json` / `gsm_<VAL>_off.json` — top-level summaries (accuracy on vs off).
- `gsm_<VAL>_{on,off}/` subdirs: raw lm-eval `results_*.json` + `samples_gsm8k_*.jsonl`,
  `chat_<b>_poc_<0|1>_mt_<M>/{lm_eval.log,run_stats.json}`, `poc_artifacts.json` (on-run only),
  `results_table.md`.

## Only with `--push`
- `.manifest` — sorted list of every relative path ([scope/s3.sh](scope/s3.sh) writes it pre-upload).
- `report.html` gets an "Artifacts & Reproduce (S3)" section injected in place
  ([scope/inject_s3.py](scope/inject_s3.py)).

## Sanitization
Before rendering, `run_scope.sh` sed-scrubs the home dir/username out of every
`.log/.md/.html/.json/.jsonl` — local absolute paths never leak into the archive.

See also: [HOWTO.md](HOWTO.md) (how to run) · [README.md](README.md) (per-tool reference).

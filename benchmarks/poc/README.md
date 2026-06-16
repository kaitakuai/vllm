# PoC benchmark instruments — what each measures

All share one core metric where relevant: **k-id (sphere_k) mismatch** — the
validator re-derives a trajectory teacher-forced against the prover's reference
(seed each step from the reference k, no cascade) and counts steps where its k
differs. honest ≈ 0, fraud high, `gap = fraud − honest`.

| File | Measures | Question it answers |
|------|----------|---------------------|
| `poc_validation.py` | k-id mismatch core (inference → validation) | the shared library both drivers build on |
| `honest_fraud_sequential.py` | honest vs fraud separation, single-GPU (one model at a time) | "does the seal separate honest from a cheaper (AWQ) model?" |
| `perfomance_nonces.py` | nonces/sec across (n_nonces, max_tokens), CG vs eager | "what's the decode/prefill throughput; how much does cudagraph help?" |
| `chat_throughput.py` | chat-only decode throughput (no PoC) | baseline chat speed |
| `quality_gsm8k.py` | gsm8k accuracy + throughput under concurrent PoC load | "does chat quality hold while PoC runs?" (criterion 2) |
| `graph_inspect.py` | `cudaGraphLaunch` count and/or timeline PNG (one profiling run) | "is the PoC forward graphed?" (decode>0, eager=0) + a picture of where graphs replay. `--render PATH` for the PNG; accepts an existing trace to skip the boot |
| `tests/poc/_graph.py` | shared boot+profile+count helper | used by graph_inspect + the cudagraph test |

## Lineage
- Core metric originates in Barbara's `bs/poc-context-fix` (two-server inference +
  validation, `collect_validation.sh`). We kept the metric, switched orchestration
  to **single-GPU sequential** (`honest_fraud_sequential.py`).
- Mykola's `kaitakuai/experiments` use the same k-id metric, adding cross-HW/TP and
  compiled-vs-eager axes (`matrix-gen.sh`/`matrix-prof.sh`/`analyze_matrix.py`).

## Removed (dead)
- `graph_report.py` — read a graph-capture kernel dump (`VLLM_POC_GRAPH_DUMP`) that
  the native path no longer produces; superseded by `graph_inspect.py`.
- `honest_validation.py` — old L2-distance two-server variant; superseded by the
  k-id metric in `poc_validation.py`.
- `graph_coverage.py` + `graph_timeline.py` — merged into `graph_inspect.py` (count
  and/or render from one profiling run; matplotlib imported lazily only for render).

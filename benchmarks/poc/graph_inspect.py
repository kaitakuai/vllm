#!/usr/bin/env python3
"""Inspect CUDA-graph usage of a PoC request: count cudaGraphLaunch and/or render
a timeline PNG. One profiling run feeds both outputs.

Usage:
  # boot a server, profile one PoC request, print the launch count:
  python benchmarks/poc/graph_inspect.py
  # ...and also render a timeline picture:
  python benchmarks/poc/graph_inspect.py --render graph.png
  # render/count an EXISTING trace (no server boot):
  python benchmarks/poc/graph_inspect.py <trace.json[.gz]> --render graph.png

Env (auto-boot mode): POC_GI_MAXTOK (default 8; 0 = prefill-only),
                      POC_GI_EAGER=1 (--enforce-eager baseline; expect 0 launches),
                      POC_GI_CGMODE (CUDAGraphMode override, e.g. FULL).
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tests.poc._graph import (  # noqa: E402
    count_graph_launches,
    load_trace_events,
    profile_poc_request,
)

# kernel-name substring -> (category, colour); first match wins
_CATS = [
    ("attention", ("flashinfer", "paged", "attention", "attn", "fmha", "bmm"), "#d62728"),
    ("gemm", ("marlin", "awq", "gptq", "cutlass", "gemm", "matmul", "addmm",
              "_mm_", "wgmma", "ampere"), "#1f77b4"),
    ("norm", ("rmsnorm", "rms_norm", "layernorm", "layer_norm", "norm"), "#2ca02c"),
    ("rope", ("rope", "rotary"), "#9467bd"),
    ("elementwise", ("elementwise", "where", "vectorized", "add", "mul", "copy",
                     "fill", "silu", "gelu", "activation"), "#ff7f0e"),
]
_OTHER = ("other", "#7f7f7f")
_LAUNCH_COLOR = "#111111"


def _classify(name: str):
    n = name.lower()
    for cat, subs, color in _CATS:
        if any(s in n for s in subs):
            return cat, color
    return _OTHER


def render(trace_path: str, out_path: str) -> int:
    """Render a cudaGraphLaunch + GPU-kernel timeline PNG. matplotlib imported
    lazily so the count path / CI never depends on it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    events = [e for e in load_trace_events(trace_path)
              if isinstance(e, dict) and e.get("ph") == "X"
              and e.get("ts") is not None and e.get("dur")]
    if not events:
        print("no timed events in trace")
        return 1
    t0 = min(e["ts"] for e in events)

    def ms(ts):
        return (ts - t0) / 1000.0

    launches = [e for e in events if e.get("cat") == "cuda_runtime"
                and "cudaGraphLaunch" in str(e.get("name", ""))]
    kernels = [e for e in events if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]

    Y_GPU, Y_LAUNCH, H = 0.0, 1.0, 0.7
    fig, ax = plt.subplots(figsize=(14, 3.4))
    used = {}
    for e in kernels:
        cat, color = _classify(str(e.get("name", "")))
        ax.broken_barh([(ms(e["ts"]), max(e["dur"] / 1000.0, 0.002))],
                       (Y_GPU, H), facecolors=color, edgecolors="none")
        used[cat] = color
    span = (max(ms(e["ts"]) for e in events) - min(ms(e["ts"]) for e in events)) or 1.0
    min_w = span * 0.002
    for e in launches:
        ax.broken_barh([(ms(e["ts"]), max(e["dur"] / 1000.0, min_w))],
                       (Y_LAUNCH, H), facecolors=_LAUNCH_COLOR, edgecolors="none")
    ax.set_yticks([Y_GPU + H / 2, Y_LAUNCH + H / 2])
    ax.set_yticklabels(["GPU kernels", "cudaGraphLaunch"])
    ax.set_xlabel("time (ms)")
    ax.set_title(f"PoC CUDA-graph timeline — {len(launches)} graph launches, "
                 f"{len(kernels)} GPU kernels")
    legend = [Patch(facecolor=_LAUNCH_COLOR, label="cudaGraphLaunch")]
    legend += [Patch(facecolor=c, label=cat) for cat, c in used.items()]
    ax.legend(handles=legend, ncol=len(legend), fontsize=8,
              loc="upper center", bbox_to_anchor=(0.5, -0.28))
    ax.set_ylim(-0.2, Y_LAUNCH + H + 0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"wrote {out_path}  ({len(launches)} launches, {len(kernels)} kernels)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", nargs="?", help="trace .json[.gz]; omit to boot+profile")
    ap.add_argument("--render", metavar="PATH", help="also write a timeline PNG")
    ap.add_argument("--no-count", action="store_true", help="skip the launch count")
    args = ap.parse_args()

    traces = [args.trace] if args.trace else None
    if traces is None:
        extra = ["--enforce-eager"] if os.environ.get("POC_GI_EAGER") == "1" else None
        mt = int(os.environ.get("POC_GI_MAXTOK", "8"))
        cgmode = os.environ.get("POC_GI_CGMODE")
        total, traces = profile_poc_request(server_extra_args=extra, max_tokens=mt,
                                            prof_dir="/tmp/poc_prof_gi", cgmode=cgmode)
        if not traces:
            print("NO TRACE PRODUCED — check profiler support")
            return 1
        if not args.no_count:
            for t in traces:
                print(f"{os.path.basename(t)}: cudaGraphLaunch = {count_graph_launches(t)}")
            print(f"\nTOTAL cudaGraphLaunch during PoC request: {total}")
            print("GRAPHED" if total > 0 else "NOT GRAPHED (eager/compiled-only)")
    elif not args.no_count:
        total = sum(count_graph_launches(t) for t in traces)
        print(f"TOTAL cudaGraphLaunch: {total}")
        print("GRAPHED" if total > 0 else "NOT GRAPHED (eager/compiled-only)")

    if args.render:
        # render the GPU-side (rank-0) trace — the largest one
        gpu_traces = [t for t in traces if "async_llm" not in t] or traces
        return render(max(gpu_traces, key=os.path.getsize), args.render)
    return 0


if __name__ == "__main__":
    sys.exit(main())

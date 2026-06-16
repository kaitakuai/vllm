"""
Benchmark nonces_per_second across different (n_nonces, max_tokens) configurations.

Usage
-----
Connect to an existing server::

    python perfomance_nonces.py \\
        --url http://localhost:8000 \\
        --model my-model \\
        --seq-len 256

Auto-launch a server (omit ``--url``)::

    python perfomance_nonces.py \\
        --model RedHatAI/Qwen2.5-7B-Instruct-quantized.w8a16 \\
        --seq-len 256 \\
        --server-args "--gpu-memory-utilization 0.5 --max-model-len 4096"
"""

import argparse
import hashlib
import json
import os
import shlex
import statistics
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import requests

BENCHMARK_DURATION_SEC: float = 60.0
WARMUP_REQUESTS: int = 3

DEFAULT_BLOCK_HEIGHT = 1
DEFAULT_NODE_ID = 0
DEFAULT_NODE_COUNT = 1
DEFAULT_K_DIM = 12
DEFAULT_PUBLIC_KEY = "0" * 64


def _random_block_hash() -> str:
    """Generate a random 32-byte hex block hash."""
    return hashlib.sha256(os.urandom(32)).hexdigest()


def _warmup(
    base_url: str,
    model: str,
    seq_len: int,
    batch_size: int,
    block_hash: str,
    n_requests: int = WARMUP_REQUESTS,
) -> None:
    """Send a few dummy requests to warm up the GPU before benchmarking."""
    url = f"{base_url}/api/v1/pow/generate"
    body = {
        "block_hash": block_hash,
        "block_height": DEFAULT_BLOCK_HEIGHT,
        "public_key": DEFAULT_PUBLIC_KEY,
        "node_id": DEFAULT_NODE_ID,
        "node_count": DEFAULT_NODE_COUNT,
        "nonces": list(range(batch_size)),
        "params": {"model": model, "seq_len": seq_len, "k_dim": DEFAULT_K_DIM, "max_tokens": 0},
        "batch_size": batch_size,
        "wait": True,
    }
    print(f"Warming up ({n_requests} requests) ...")
    for i in range(n_requests):
        try:
            resp = requests.post(url, json=body, timeout=120)
            resp.raise_for_status()
            print(f"  warmup {i + 1}/{n_requests} OK")
        except Exception as exc:
            print(f"  warmup {i + 1}/{n_requests} failed: {exc}")
    print()


def _run_combo(
    base_url: str,
    model: str,
    seq_len: int,
    n_nonces: int,
    max_tokens: int,
    duration_sec: float,
    block_hash: str,
    batch_size: int,
) -> dict[str, Any]:
    """Send repeated /generate requests for *duration_sec* and measure nonces/s."""
    url = f"{base_url}/api/v1/pow/generate"
    nonces = list(range(n_nonces))

    body = {
        "block_hash": block_hash,
        "block_height": DEFAULT_BLOCK_HEIGHT,
        "public_key": DEFAULT_PUBLIC_KEY,
        "node_id": DEFAULT_NODE_ID,
        "node_count": DEFAULT_NODE_COUNT,
        "nonces": nonces,
        "params": {"model": model, "seq_len": seq_len, "k_dim": DEFAULT_K_DIM, "max_tokens": max_tokens},
        "batch_size": n_nonces,
        "wait": True,
    }

    total_nonces = 0
    total_requests = 0
    errors = 0
    nonces_per_second_samples: list[float] = []

    deadline = time.monotonic() + duration_sec
    while time.monotonic() < deadline:
        t0 = time.monotonic()
        try:
            resp = requests.post(url, json=body, timeout=duration_sec + 10)
            resp.raise_for_status()
            elapsed = time.monotonic() - t0
            total_nonces += n_nonces
            total_requests += 1
            nonces_per_second_samples.append(n_nonces / elapsed if elapsed > 0 else 0.0)
        except Exception as exc:
            errors += 1
            print(f"  [error] n_nonces={n_nonces} max_tokens={max_tokens}: {exc}")

    wall_time = duration_sec - max(0.0, deadline - time.monotonic())
    nps_median = statistics.median(nonces_per_second_samples) if nonces_per_second_samples else 0.0

    return {
        "n_nonces": n_nonces,
        "max_tokens": max_tokens,
        "total_nonces": total_nonces,
        "total_requests": total_requests,
        "errors": errors,
        "wall_time_sec": round(wall_time, 3),
        "nonces_per_second_samples": [round(v, 3) for v in nonces_per_second_samples],
        "nonces_per_second_median": round(nps_median, 3),
    }


def _write_markdown_table(
    results: list[dict[str, Any]],
    config: dict[str, Any],
    md_path: str,
) -> None:
    """Write a flat results table with one row per (n_nonces, max_tokens) combo."""
    columns = ["model_name", "n_nonces", "max_tokens", "nonces_per_second_median", "total_nonces", "duration"]

    rows: list[list[str]] = []
    for r in results:
        rows.append([
            config["model"],
            str(r["n_nonces"]),
            str(r["max_tokens"]),
            f"{r['nonces_per_second_median']:.3f}",
            str(r["total_nonces"]),
            f"{r['wall_time_sec']:.1f}s",
        ])

    col_widths = [
        max(len(columns[i]), max((len(row[i]) for row in rows), default=0))
        for i in range(len(columns))
    ]

    def fmt_row(cells: list[str]) -> str:
        return "| " + " | ".join(c.ljust(col_widths[i]) for i, c in enumerate(cells)) + " |"

    sep = "| " + " | ".join("-" * w for w in col_widths) + " |"

    lines = [
        "# PoC Benchmark — nonces/s (median)",
        "",
        fmt_row(columns),
        sep,
        *[fmt_row(r) for r in rows],
    ]

    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _run_benchmark(url: str, args: argparse.Namespace) -> None:
    """Execute the full nonces/s sweep against *url* and write output files."""
    block_hash = _random_block_hash()
    _warmup(url, args.model, args.seq_len, args.batch_size, block_hash, args.warmup)

    combos = [(n, t) for n in args.nonces for t in args.max_tokens]
    total = len(combos)

    print(f"Block hash: {block_hash}")
    print(f"Running {total} configs × {args.duration:.0f}s each "
          f"(~{total * args.duration / 60:.1f} min total)\n")

    results = []
    for i, (n_nonces, max_tokens) in enumerate(combos, 1):
        print(f"[{i}/{total}] n_nonces={n_nonces}, max_tokens={max_tokens} ...")
        r = _run_combo(
            base_url=url,
            model=args.model,
            seq_len=args.seq_len,
            n_nonces=n_nonces,
            max_tokens=max_tokens,
            duration_sec=args.duration,
            block_hash=block_hash,
            batch_size=args.batch_size,
        )
        results.append(r)
        print(f"         nonces/s median={r['nonces_per_second_median']:.2f}, "
              f"requests={r['total_requests']}, errors={r['errors']}, "
              f"samples={len(r['nonces_per_second_samples'])}")

    config = {
        "url": url,
        "model": args.model,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "duration_sec": args.duration,
        "block_hash": block_hash,
        "number_of_nonces_sweep": args.nonces,
        "max_tokens_sweep": args.max_tokens,
    }

    with open(args.output, "w") as f:
        json.dump({"config": config, "results": results}, f, indent=2)
    print(f"\nResults saved to {args.output}")

    _write_markdown_table(results, config, args.md_output)
    print(f"Table saved to {args.md_output}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark vLLM PoC nonces/s",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--url", help="Base server URL; omit to auto-launch a server")
    parser.add_argument("--model", help="Model name / HuggingFace id")
    parser.add_argument("--seq-len", type=int, help="seq_len param")
    parser.add_argument("--batch-size", type=int, default=16, help="batch_size per request")
    parser.add_argument(
        "--nonces",
        type=int,
        nargs="+",
        default=[32, 64],
        metavar="N",
        help="Space-separated list of n_nonces values to sweep",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        nargs="+",
        default=[0, 32, 64, 128, 256],
        metavar="T",
        help="Space-separated list of max_tokens values to sweep",
    )
    parser.add_argument("--duration", type=float, default=BENCHMARK_DURATION_SEC, help="Seconds per config")
    parser.add_argument("--warmup", type=int, default=WARMUP_REQUESTS, help="Warmup requests before benchmarking")
    parser.add_argument("--output", default="benchmark_results.json", help="Output JSON file path")
    parser.add_argument("--md-output", default="benchmark_results.md", help="Output Markdown table file path")
    parser.add_argument("--from-json", metavar="FILE", help="Skip benchmark; regenerate table from existing JSON")
    parser.add_argument(
        "--server-args",
        default="",
        metavar="ARGS",
        help=(
            "Extra args forwarded to 'vllm serve' when auto-launching "
            "(quoted string, e.g. \"--gpu-memory-utilization 0.5 --max-model-len 4096\")"
        ),
    )
    args = parser.parse_args()

    if args.from_json:
        with open(args.from_json) as f:
            data = json.load(f)
        _write_markdown_table(data["results"], data["config"], args.md_output)
        print(f"Table saved to {args.md_output}")
        return

    if not args.model or not args.seq_len:
        parser.error("--model and --seq-len are required when not using --from-json")

    if args.url:
        _run_benchmark(args.url, args)
    else:
        from tests.poc._server import PoCTestServer

        extra = shlex.split(args.server_args) if args.server_args else []
        with PoCTestServer(args.model, extra) as srv:
            _run_benchmark(srv.url_root, args)


if __name__ == "__main__":
    main()

import argparse
import asyncio
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import aiohttp
import numpy as np
from poc_validation import (  # noqa: E402
    save_run, env_info, add_engine_args, deploy_from_args,
)


async def _send_poc_request(
    base_url: str,
    model: str,
    block_hash: str,
    nonces: list[int],
    public_key: str = "test_node",
    max_tokens: int = 0,
    timeout: int = 600,
) -> tuple[dict[str, Any] | None, float]:
    """Send one PoC generate request and return (result, elapsed_seconds).

    max_tokens > 0 runs decode PoC (the proposal's purpose); 0 = prefill-only.
    Proposal API: max_tokens lives under params.
    """
    url = f"{base_url}/api/v1/pow/generate"
    payload = {
        "block_hash": block_hash,
        "block_height": 100,
        "public_key": public_key,
        "node_id": 0,
        "node_count": 1,
        "nonces": nonces,
        "params": {"model": model, "seq_len": 256, "k_dim": 12,
                   "max_tokens": max_tokens},
        "wait": True,
    }
    t0 = time.time()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout)
            ) as resp:
                result = await resp.json() if resp.status == 200 else {"error": f"status {resp.status}"}
    except Exception as exc:
        result = {"error": str(exc)}
    return result, time.time() - t0


async def _poc_sender_loop(
    base_url: str,
    model: str,
    interval_seconds: float,
    requests_per_interval: int,
    stop_event: asyncio.Event,
    poc_artifacts: list[dict[str, Any]],
    poc_times: list[float],
    max_tokens: int = 0,
    nonces_per_request: int = 32,
    poc_timeout: int = 600,
) -> None:
    """Maintain *requests_per_interval* PoC requests IN FLIGHT continuously until
    *stop_event*.

    Each of N workers fires a request, records it, and immediately fires the next —
    so exactly N PoC requests co-exist with gsm8k at all times (no interval gaps, no
    pile-up). ``interval_seconds`` is kept for signature compatibility but unused.
    """
    n_inflight = requests_per_interval
    counters = {"nonce": 0, "req": 0}
    print(
        f"[PoC] Continuous load: {n_inflight} requests in flight "
        f"({nonces_per_request} nonce(s) x {max_tokens} decode steps each)\n"
    )

    async def worker():
        while not stop_event.is_set():
            nonces = list(range(counters["nonce"],
                                counters["nonce"] + nonces_per_request))
            counters["nonce"] += nonces_per_request
            counters["req"] += 1
            rid = counters["req"]
            result, elapsed = await _send_poc_request(
                base_url, model, f"block_{rid}", nonces,
                max_tokens=max_tokens, timeout=poc_timeout)
            ok = bool(result) and "error" not in result
            poc_artifacts.append({
                "request_id": rid,
                "timestamp": time.time(),
                "block_hash": f"block_{rid}",
                "nonces": nonces,
                "result": result,
                "elapsed_time": elapsed,
                "ok": ok,
            })
            poc_times.append(elapsed)
            if len(poc_artifacts) % 16 == 0:
                done = sum(1 for a in poc_artifacts if a.get("ok"))
                print(f"[PoC] {len(poc_artifacts)} requests done, {done} OK")

    workers = [asyncio.create_task(worker()) for _ in range(n_inflight)]
    await stop_event.wait()
    for w in workers:
        w.cancel()
    await asyncio.gather(*workers, return_exceptions=True)

    print(f"[PoC] Stopped: {len(poc_artifacts)} requests sent")


async def _stream_process_output(process: subprocess.Popen, log_file: Path) -> int:
    """Stream stdout of *process* to console and *log_file*; return exit code."""
    with open(log_file, "w", buffering=1) as log_f:
        while True:
            return_code = process.poll()
            line = process.stdout.readline()
            if line:
                print(line, end="", flush=True)
                log_f.write(line)
            if return_code is not None:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log_f.write(line)
                break
            await asyncio.sleep(0.01)
    return return_code


def _parse_gsm8k_results(output_path: Path, since: float) -> dict[str, Any] | None:
    """Find the most recent gsm8k results JSON written after *since*."""
    candidates = [
        f
        for d in output_path.glob("*/")
        for f in d.glob("results_*.json")
        if f.stat().st_mtime >= since
    ]
    if not candidates:
        return None

    results_file = max(candidates, key=lambda p: p.stat().st_mtime)
    try:
        with open(results_file) as f:
            data = json.load(f)
        gsm8k = data.get("results", {}).get("gsm8k", {})
        if gsm8k:
            return {
                "strict_match": gsm8k.get("exact_match,strict-match"),
                "flexible_extract": gsm8k.get("exact_match,flexible-extract"),
                "results_file": str(results_file),
            }
    except Exception as exc:
        print(f"Error parsing gsm8k results: {exc}")
    return None


def _parse_run_stats(stats_file: Path) -> dict[str, Any]:
    """Extract display-ready fields from a ``run_stats.json`` file."""
    with open(stats_file) as f:
        data = json.load(f)

    model = data.get("model_name", "Unknown")
    if "/" in model:
        model = model.split("/")[-1]

    folder = stats_file.parent.name
    # poc_requests/poc_max_tokens come from run_stats.json (robust); fall back to
    # parsing the folder name (chat_{bs}_poc_{n}_mt_{m}) for older runs.
    poc_requests = data.get("poc_requests")
    if poc_requests is None:
        poc_requests = 0
        if folder.startswith("chat_") and "_poc_" in folder:
            parts = folder.split("_")
            if len(parts) >= 4:
                poc_requests = int(parts[3])

    gsm8k = data.get("gsm8k") or {}
    return {
        "model": model,
        "batch_size": data.get("batch_size", 0),
        "poc_requests": poc_requests,
        "poc_max_tokens": data.get("poc_max_tokens", 0),
        "strict_match": gsm8k.get("strict_match", 0.0),
        "flexible_extract": gsm8k.get("flexible_extract", 0.0),
        "elapsed": data.get("elapsed_seconds", 0),
        "median_time": data.get("poc_median_time"),
        "folder": folder,
    }


def generate_table(eval_results_dir: Path, output_file: Path | None = None) -> None:
    """Collect all ``run_stats.json`` files under *eval_results_dir* and write a Markdown table.

    Rows are sorted by (model, batch_size, poc_requests). When *output_file* is
    given the table is written there; otherwise it is printed to stdout.
    """
    rows: list[dict[str, Any]] = []
    for folder in eval_results_dir.iterdir():
        if not folder.is_dir() or not folder.name.startswith("chat_"):
            continue
        stats_file = folder / "run_stats.json"
        if not stats_file.exists():
            continue
        try:
            rows.append(_parse_run_stats(stats_file))
        except Exception as exc:
            print(f"Warning: could not parse {stats_file}: {exc}", file=sys.stderr)

    if not rows:
        print("No results found.", file=sys.stderr)
        return

    rows.sort(key=lambda r: (r["model"], r["batch_size"], r["poc_requests"]))

    header = "| Configuration | Batch Size | PoC Batch | Decode (max_tokens) | Accuracy (Strict/Flexible) | Time gsm8k (s) | Median Time PoC (s) |"
    sep    = "|---------------|------------|-----------|---------------------|----------------------------|----------------|---------------------|"
    lines  = [header, sep]
    for r in rows:
        median_str = f"{r['median_time']:.3f}" if r["median_time"] is not None else ""
        lines.append(
            f"| {r['model']:<13} | {r['batch_size']:<10} | {r['poc_requests']:<9} "
            f"| {r['poc_max_tokens']:<19} "
            f"| {r['strict_match']:.4f} / {r['flexible_extract']:.4f}          "
            f"| {int(r['elapsed']):<14} | {median_str:<19} |"
        )

    table = "\n".join(lines) + "\n"
    if output_file:
        output_file.write_text(table)
        print(f"Table saved to: {output_file}")
    else:
        print(table)


async def _run_eval(args: argparse.Namespace) -> int:
    """Execute one lm-eval run and collect PoC timing data."""
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    poc_requests = 0 if args.disable_poc else args.poc_requests
    poc_max_tokens = 0 if args.disable_poc else args.max_tokens
    # include max_tokens so decode vs prefill runs (same bs/poc) don't collide
    run_name = f"chat_{args.batch_size}_poc_{poc_requests}_mt_{poc_max_tokens}"
    run_output_path = output_path / run_name
    run_output_path.mkdir(parents=True, exist_ok=True)

    server_url = getattr(args, "server_url", None) or f"http://{args.host}:{args.port}"

    print("=" * 80)
    print(f"Model     : {args.model_name}")
    print(f"Batch size: {args.batch_size}")
    print(f"Server    : {server_url}")
    if args.disable_poc:
        print("PoC       : Disabled")
    else:
        print(f"PoC       : {args.poc_requests} requests in flight (continuous), "
              f"{args.poc_nonces} nonce(s) x {args.max_tokens} steps each")
    print(f"Output    : {run_output_path}")
    print("=" * 80)
    print()

    cmd = [
        "lm-eval",
        "--model", "local-chat-completions",
        "--model_args", (
            f"model={args.model_name},base_url={server_url}/v1/chat/completions,"
            f"num_concurrent={args.batch_size},"
            f"timeout={args.client_timeout},max_retries={args.max_retries}"
        ),
        "--tasks", args.tasks,
        "--output_path", str(output_path),
        "--log_samples",
        "--apply_chat_template",
    ]
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]

    print("Starting lm_eval...")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    start_time = time.time()

    poc_artifacts: list[dict[str, Any]] = []
    poc_times: list[float] = []
    stop_event = asyncio.Event()

    poc_task = None
    if not args.disable_poc:
        poc_task = asyncio.create_task(
            _poc_sender_loop(
                server_url,
                args.model_name,
                args.poc_interval,
                args.poc_requests,
                stop_event,
                poc_artifacts,
                poc_times,
                args.max_tokens,
                args.poc_nonces,
                args.poc_timeout,
            )
        )

    try:
        return_code = await _stream_process_output(process, run_output_path / "lm_eval.log")
        elapsed = time.time() - start_time

        if poc_task:
            stop_event.set()
            await poc_task

        print()
        print("=" * 80)
        print(f"Completed in {elapsed:.1f}s ({elapsed / 60:.1f}min)")
        print("=" * 80)
        print()

        run_stats: dict[str, Any] = {
            "model_name": args.model_name,
            "batch_size": args.batch_size,
            "server_host": args.host,
            "server_port": args.port,
            "tasks": args.tasks,
            "elapsed_seconds": elapsed,
            "return_code": return_code,
            "poc_enabled": not args.disable_poc,
            "poc_requests": 0 if args.disable_poc else args.poc_requests,
            "poc_nonces": 0 if args.disable_poc else args.poc_nonces,
            "poc_max_tokens": 0 if args.disable_poc else args.max_tokens,
        }

        if return_code == 0:
            gsm8k = _parse_gsm8k_results(output_path, start_time)
            if gsm8k:
                run_stats["gsm8k"] = gsm8k
                print(f"GSM8K  strict:   {gsm8k['strict_match']:.4f}")
                print(f"GSM8K  flexible: {gsm8k['flexible_extract']:.4f}")
                print(f"       file:     {gsm8k['results_file']}")
                print()

        if poc_artifacts:
            n_ok = sum(1 for a in poc_artifacts if a.get("ok"))
            nonces_done = sum(
                len(a["nonces"]) for a in poc_artifacts if a.get("ok"))
            nonces_per_sec = nonces_done / elapsed if elapsed > 0 else 0.0
            run_stats["poc_requests_sent"] = len(poc_artifacts)
            run_stats["poc_requests_successful"] = n_ok
            run_stats["poc_nonces_processed"] = nonces_done
            run_stats["poc_nonces_per_sec"] = nonces_per_sec
            if poc_times:
                run_stats["poc_median_time"] = float(np.median(poc_times))
            print(f"PoC absolute time:   {elapsed:.1f}s")
            print(f"PoC requests OK:     {n_ok}/{len(poc_artifacts)}")
            print(f"PoC nonces processed:{nonces_done}")
            print(f"PoC NONCES/SEC:      {nonces_per_sec:.3f}")
            print()

            artifacts_file = run_output_path / "poc_artifacts.json"
            artifacts_file.write_text(json.dumps(poc_artifacts, indent=2))
            print(f"PoC artifacts saved to: {artifacts_file}")

        stats_file = run_output_path / "run_stats.json"
        stats_file.write_text(json.dumps(run_stats, indent=2))

        # collect-format result (accuracy + provenance) for offline analyze.py
        if getattr(args, "save", None):
            gsm = run_stats.get("gsm8k") or {}
            save_run(args.save,
                     {"role": "gsm8k", "model": args.model_name, "tasks": args.tasks,
                      "limit": args.limit, "batch_size": args.batch_size,
                      "poc_enabled": not args.disable_poc, "poc_max_tokens": poc_max_tokens,
                      **getattr(args, "prov", {})}, [],
                     results={"strict_match": gsm.get("strict_match"),
                              "flexible_extract": gsm.get("flexible_extract"),
                              "n_samples": args.limit, "elapsed_s": round(elapsed, 1),
                              "poc_nonces_per_s": run_stats.get("poc_nonces_per_sec")})
            print(f"saved -> {args.save}")
        print(f"Run stats saved to: {stats_file}")

        table_path = Path(args.table_output) if args.table_output else output_path / "results_table.md"
        generate_table(output_path, table_path)

        return 0 if return_code == 0 else 1

    except KeyboardInterrupt:
        print("\nInterrupted!")
        process.terminate()
        if poc_task:
            stop_event.set()
            await poc_task
        return 130


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run lm-eval with concurrent PoC requests and generate a comparison table."
    )
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--batch_size", type=int)
    add_engine_args(parser)  # --url/--target/--profile/--configs/--eager/--dtype (shared)
    parser.add_argument(
        "--host",
        default=None,
        metavar="IP",
        help="Connect-only host/IP of an ALREADY-running server (with --port; no deploy).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Connect-only port of an already-running server (with --host). "
             "Omit and use --url (remote deploy) or neither (local boot) to deploy.",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Write a collect-format result JSON (accuracy + provenance) for analyze.py.",
    )
    parser.add_argument("--output_path", type=str, default="./eval_results")
    parser.add_argument("--tasks", type=str, default="gsm8k")
    parser.add_argument("--poc_interval", type=float, default=1.0)
    parser.add_argument("--poc_requests", type=int, default=1,
                        help="Concurrent PoC requests fired per interval.")
    parser.add_argument("--poc_nonces", type=int, default=32,
                        help="Nonces per PoC request (prod PoC batch = 32).")
    parser.add_argument(
        "--max_tokens", type=int, default=256,
        help="PoC decode steps during the run (256 = decode PoC, the proposal's "
             "purpose; 0 = prefill-only). Requires the server started with --poc-decode.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit gsm8k to first N questions (fast experiments). Omit = full set.",
    )
    parser.add_argument(
        "--client_timeout", type=int, default=1200,
        help="lm-eval HTTP client timeout (s) per chat request. Bump high so a "
             "slow/contended server does not close the session (the NA root cause "
             "in defer mode). Default 1200 (lm-eval's own default is 300).",
    )
    parser.add_argument(
        "--max_retries", type=int, default=5,
        help="lm-eval HTTP client max retries per chat request (default 5; "
             "lm-eval's own default is 3).",
    )
    parser.add_argument(
        "--poc_timeout", type=int, default=600,
        help="Per-PoC-request HTTP timeout (s) in the background load loop. "
             "Default 600 (was hardcoded 60).",
    )
    parser.add_argument("--disable_poc", action="store_true")
    parser.add_argument(
        "--server-args",
        default="",
        metavar="ARGS",
        help=(
            "Extra args forwarded to 'vllm serve' when auto-launching "
            "(quoted string, e.g. \"--gpu-memory-utilization 0.5 --max-model-len 4096\")"
        ),
    )
    parser.add_argument(
        "--table-output",
        metavar="FILE",
        default=None,
        help="Path for the Markdown comparison table (default: {output_path}/results_table.md)",
    )
    parser.add_argument(
        "--table-only",
        action="store_true",
        help="Skip the eval run; only regenerate the comparison table from existing results.",
    )
    args = parser.parse_args()

    if args.table_only:
        output_path = Path(args.output_path)
        if not output_path.exists():
            print(f"Error: {output_path} does not exist.", file=sys.stderr)
            return 1
        table_path = Path(args.table_output) if args.table_output else output_path / "results_table.md"
        generate_table(output_path, table_path)
        return 0

    if not args.model_name or args.batch_size is None:
        parser.error("--model_name and --batch_size are required unless --table-only is set.")

    if args.host is not None and args.port is None:
        parser.error("--host requires --port.")

    # connect-only to an ALREADY-running server (no deploy): explicit --host/--port
    if args.port is not None:
        args.host = args.host or "127.0.0.1"
        args.server_url = f"http://{args.host}:{args.port}"
        args.prov = dict(env_info())
        return asyncio.run(_run_eval(args))

    # deploy via the shared path (remote --url, or local boot) — same as collect/perf.
    # gsm8k chat needs more context than PoC's 1024 default, so request a larger
    # --max-model-len unless the caller already set one via --server-args.
    extra = shlex.split(args.server_args) if args.server_args else []
    if not any(s.startswith("--max-model-len") for s in extra):
        extra += ["--max-model-len", "4096"]
    extra += ["--no-enable-prefix-caching"]
    with deploy_from_args(args, args.model_name, extra_args=extra) as (url, srv):
        args.server_url = url
        return asyncio.run(_run_eval(args))


if __name__ == "__main__":
    sys.exit(main())

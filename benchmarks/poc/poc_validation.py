"""Send PoC inference requests to the inference server, then re-run each as a
validation request on the validation server.

Validation mode: the validation server receives the inference sphere_k_steps and
uses them to seed every decode step so both servers run identical forward passes.
Any k-id difference at a step counts as one hardware mismatch (no cascading).

Usage:
    python poc_compare.py [--inference URL] [--validation URL] [--n N]
                          [--block-hash HASH] [--public-key KEY]
                          [--nonce NONCE] [--seq-len SEQ_LEN]
                          [--max-tokens MAX_TOKENS] [--output FILE]
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional
import hashlib
import os

import requests

# --- shared core: boot a server, one request builder, and save/load for
# precomputed artifacts. Used by every PoC benchmark so the flow + on-disk format
# are identical and reproducible. ---
import contextlib


@contextlib.contextmanager
def serve(model: str, eager: bool = False, extra_args: Optional[List[str]] = None):
    """Boot ONE PoC server (sequential; two 7B don't co-fit on 20GB) and yield the
    server handle (srv.url_root, srv.log_path, srv.serve_args). cudagraph by
    default, --enforce-eager when eager. Wraps tests/poc/_server.PoCTestServer."""
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tests.poc._server import DEFAULT_SERVER_ARGS, PoCTestServer
    args = list(DEFAULT_SERVER_ARGS) + (["--enforce-eager"] if eager else []) + list(extra_args or [])
    with PoCTestServer(model, args) as srv:
        srv.serve_args = args
        yield srv


def vllm_config(srv) -> Dict[str, Any]:
    """Parse the booted server's ACTUAL engine config from its startup log + args:
    attention backend, cudagraph mode, dtype, quantization, max_model_len,
    enforce_eager, KV cache size. High-class provenance for each run."""
    import re
    cfg: Dict[str, Any] = {"model": getattr(srv, "model", None),
                           "server_args": getattr(srv, "serve_args", None)}
    try:
        with open(srv.log_path) as f:
            log = f.read()
    except Exception:
        return cfg
    pats = {
        # matches both the auto-select log ("Using FLASH_ATTN attention backend out
        # of ...") and the explicit-flag log ("Using AttentionBackendEnum.FLASHINFER
        # backend."); [A-Z_]+ excludes lowercase like "Using inductor backend".
        "attention_backend": r"Using (?:AttentionBackendEnum\.)?([A-Z_]+)(?: attention)? backend",
        "cudagraph_mode": r"'cudagraph_mode':\s*<CUDAGraphMode\.(\w+)",
        "dtype": r"dtype=torch\.(\w+)",
        "quantization": r"quantization=([\w./-]+)",
        "max_model_len": r"max_seq_len=(\d+)",
        "enforce_eager": r"enforce_eager=(True|False)",
        "kv_cache_tokens": r"GPU KV cache size:\s*([\d,]+) tokens",
    }
    for key, pat in pats.items():
        m = re.search(pat, log)
        if m:
            cfg[key] = m.group(1)
    return cfg


def remote_config(url: str, want_backend: Optional[str], eager: bool) -> Dict[str, Any]:
    """Provenance for a server we connect to but don't boot (connect/remote mode): no
    log to parse, so take the vLLM version + served model over HTTP and label the
    engine/attention from what was REQUESTED (a remote can silently fall back, so
    these are 'requested', verified only on local boots)."""
    cfg: Dict[str, Any] = {"server_url": url}
    try:
        cfg["vllm_version"] = requests.get(f"{url}/version", timeout=5).json().get("version")
    except Exception:
        pass
    try:
        data = requests.get(f"{url}/v1/models", timeout=5).json().get("data") or []
        if data:
            cfg["model"] = data[0]["id"]
    except Exception:
        pass
    if want_backend:
        cfg["attention_backend"] = want_backend
    cfg["cudagraph_mode"] = "NONE" if eager else "FULL_AND_PIECEWISE"
    return cfg


# PoC endpoint prefix per target: vLLM engine directly vs the ML-node proxy.
POC_PREFIX = {"vllm": "/api/v1/pow", "mlnode": "/api/v1/inference/pow"}

# vLLM serve flags PoC needs (used for both local boot and remote ML-node deploy).
DEFAULT_POC_SERVE_ARGS = ["--poc-decode",
                          "--gpu-memory-utilization", "0.8", "--max-model-len", "1024"]

# Named engine profiles (graph/eager, attention backend, env) live in poc_configs.json
# beside this file. A profile = {eager: bool, env: {VLLM_*: ...}, args: [extra flags]}.
DEFAULT_CONFIGS = str(Path(__file__).resolve().parent / "poc_configs.json")


def load_profile(name: Optional[str], path: Optional[str] = None):
    """Resolve a named engine profile -> (eager, serve_args, env). name None ->
    (False, [], {}). `env` is vLLM env vars (e.g. VLLM_ATTENTION_BACKEND) the caller
    must export before booting (works for local serve; a remote --url ML node can't
    receive env over inference/up). Unknown name is a hard error listing the choices."""
    if not name:
        return False, [], {}
    with open(path or DEFAULT_CONFIGS) as f:
        profs = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
    if name not in profs:
        raise SystemExit(f"unknown profile '{name}'; available: {', '.join(sorted(profs))}")
    p = profs[name]
    return bool(p.get("eager", False)), list(p.get("args", [])), dict(p.get("env", {}))


def wait_ready(base_url: str, model: Optional[str] = None, timeout: int = 900,
               poll: float = 5.0) -> str:
    """Block until the server is READY FOR INFERENCE — /v1/models lists a model
    (the engine finished loading), not merely /health up. Returns the served model
    id. If `model` is given, also require it to be the one served."""
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            r = requests.get(f"{base_url}/v1/models", timeout=10)
            if r.status_code == 200:
                data = r.json().get("data") or []
                if data:
                    served = data[0]["id"]
                    if model is None or served == model:
                        return served
                    last = f"serving {served}, want {model}"
        except requests.RequestException as e:
            last = str(e)
        time.sleep(poll)
    raise TimeoutError(f"{base_url} not ready for inference within {timeout}s ({last})")


def inference_up(base_url: str, model: str, dtype: str = "bfloat16",
                 additional_args: Optional[List[str]] = None, timeout: int = 900) -> None:
    """Deploy a model on a remote ML node (POST /api/v1/inference/up) and wait until
    READY FOR INFERENCE (the model is loaded + served). additional_args are vLLM
    serve flags (e.g. --poc-decode ...). No IP hardcoded — base_url is --url."""
    requests.post(f"{base_url}/api/v1/inference/up",
                  json={"model": model, "dtype": dtype,
                        "additional_args": additional_args or DEFAULT_POC_SERVE_ARGS},
                  timeout=timeout).raise_for_status()
    wait_ready(base_url, model=model, timeout=timeout)


def inference_down(base_url: str, timeout: int = 300) -> None:
    try:
        requests.post(f"{base_url}/api/v1/inference/down", json={}, timeout=timeout)
    except Exception:
        pass


@contextlib.contextmanager
def deploy(model: str, *, url: Optional[str] = None, eager: bool = False,
           dtype: str = "bfloat16", extra_args: Optional[List[str]] = None):
    """Make `model` available and yield its base url — client/server friendly:
      - url given  : deploy on the REMOTE ML node (inference_up/down). The rented
                     box runs the node; we only drive it over HTTP.
      - url None   : boot vLLM LOCALLY (serve()) for dev.
    Works for ANY model; nothing IP-hardcoded."""
    if url:
        # Connect mode: if the server at `url` ALREADY serves `model`, use it as-is
        # (a raw vLLM already running — e.g. a docker image you booted yourself). No
        # control API needed. Otherwise treat `url` as a remote ML node and boot via
        # /api/v1/inference/up (and tear down after).
        already = False
        try:
            wait_ready(url, model=model, timeout=8)
            already = True
        except Exception:
            already = False
        if already:
            yield url, None          # connect: external server, leave it running
        else:
            args = list(DEFAULT_POC_SERVE_ARGS) + list(extra_args or []) + (["--enforce-eager"] if eager else [])
            inference_up(url, model, dtype=dtype, additional_args=args)
            try:
                yield url, None      # remote ML node: no local server handle
            finally:
                inference_down(url)
    else:
        with serve(model, eager, extra_args) as srv:
            yield srv.url_root, srv  # local: handle for log-based provenance


# --- shared CLI scaffolding so all PoC tools (collect / perf / gsm8k) boot the
# same way: same flags, same profile resolution, same provenance. ---

def add_engine_args(ap: argparse.ArgumentParser) -> None:
    """Register the engine/deploy flags every PoC tool shares."""
    ap.add_argument("--url", help="ML-node base url (client/server); else auto-boot vLLM locally")
    ap.add_argument("--target", choices=["vllm", "mlnode"], default="vllm")
    ap.add_argument("--profile", help="named engine profile from poc_configs.json (e.g. eager-flashinfer)")
    ap.add_argument("--configs", help="profiles file (default: poc_configs.json beside this script)")
    ap.add_argument("--eager", action="store_true", help="force eager (also via a profile)")
    ap.add_argument("--dtype", default="bfloat16")


def _requested_backend(serve_args: List[str]) -> Optional[str]:
    """The --attention-backend value in a serve-arg list, if any."""
    for i, a in enumerate(serve_args):
        if a == "--attention-backend" and i + 1 < len(serve_args):
            return serve_args[i + 1]
        if a.startswith("--attention-backend="):
            return a.split("=", 1)[1]
    return None


@contextlib.contextmanager
def deploy_from_args(args, model: str, extra_args: Optional[List[str]] = None):
    """THE boot path for every tool. Resolve `--profile` (env + eager + serve args),
    deploy (remote `--url` or local), yield (url, srv), and stash provenance on
    `args.prov` (incl. resolved `engine`/`profile`). `extra_args` = tool-specific
    serve flags (e.g. gsm8k's larger --max-model-len / --no-enable-prefix-caching).

    VERIFIES the engaged attention backend matches the request: vLLM silently falls
    back to a default if a backend is unavailable, which would mislabel results — so
    a mismatch aborts the run (local boot, where we can read the server log)."""
    p_eager, p_args, p_env = load_profile(getattr(args, "profile", None),
                                          getattr(args, "configs", None))
    eager = p_eager or getattr(args, "eager", False)
    if p_env:
        if getattr(args, "url", None):
            print(f"WARNING: profile env {list(p_env)} can't be pushed to a remote --url "
                  f"ML node; set it where the node is launched.", file=sys.stderr)
        os.environ.update(p_env)
    extra = list(p_args) + list(extra_args or [])
    url = getattr(args, "url", None)
    want_backend = _requested_backend(extra)
    with deploy(model, url=url, eager=eager, dtype=getattr(args, "dtype", "bfloat16"),
                extra_args=extra) as (base_url, srv):
        args.eager = eager  # reflect the resolved value back
        args.prov = {**env_info(),
                     **(vllm_config(srv) if srv else remote_config(url, want_backend, eager))}
        args.prov["engine"] = "eager" if eager else "cudagraph"
        args.prov["profile"] = getattr(args, "profile", None)
        args.prov["attention_requested"] = want_backend
        got = args.prov.get("attention_backend")
        if want_backend and srv is not None:
            if not got:
                raise RuntimeError(
                    f"requested --attention-backend {want_backend} but could not parse the "
                    f"engaged backend from the server log — cannot verify; aborting.")
            if got.upper() != want_backend.upper():
                raise RuntimeError(
                    f"attention backend mismatch: requested --attention-backend "
                    f"{want_backend} but the server engaged {got} (silent fallback). "
                    f"Aborting so results aren't mislabeled.")
        if want_backend and srv is None:
            print(f"NOTE: requested attention backend {want_backend} on a remote --url "
                  f"node; cannot auto-verify (no local log).", file=sys.stderr)
        yield base_url, srv


def request_generate(
    base_url: str,
    *,
    model: str,
    nonces: List[int],
    target: str = "vllm",
    block_hash: str = "deadbeef" * 8,
    public_key: str = "cafebabe" * 8,
    seq_len: int = 256,
    max_tokens: int = 0,
    k_dim: int = 12,
    batch_size: int = 32,
    enforced_k: Optional[Dict[int, List[int]]] = None,
    validation: Optional[List[Dict]] = None,
    dist_threshold: float = 0.02,
    p_mismatch: float = 0.1,
    fraud_threshold: float = 0.01,
    wait: bool = True,
    timeout: int = 900,
):
    """The one v2 /generate request builder. Returns (response_json, elapsed_sec).

    - generation:         leave enforced_k/validation None -> artifacts.
    - raw validation:     pass enforced_k -> artifacts carry n_sphere_mismatches.
    - verdict validation: pass enforced_k + validation -> run_validation runs and
      returns fraud_detected. max_tokens>0 selects the decode trajectory metric.
    """
    payload: Dict[str, Any] = {
        "block_hash": block_hash, "block_height": 100, "public_key": public_key,
        "node_id": 0, "node_count": 1, "nonces": nonces,
        "params": {"model": model, "seq_len": seq_len, "k_dim": k_dim, "max_tokens": max_tokens},
        "batch_size": batch_size, "wait": wait,
    }
    if enforced_k is not None:
        payload["enforced_k_steps"] = enforced_k
    if validation is not None:
        payload["validation"] = {"artifacts": validation}
        payload["stat_test"] = {"dist_threshold": dist_threshold,
                                "p_mismatch": p_mismatch, "fraud_threshold": fraud_threshold}
    t0 = time.perf_counter()
    resp = requests.post(f"{base_url}{POC_PREFIX[target]}/generate", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json(), time.perf_counter() - t0




def env_info() -> Dict[str, Any]:
    """Capture the environment for reproducibility: host, vLLM version, GPU."""
    info: Dict[str, Any] = {}
    try:
        import socket
        info["host"] = socket.gethostname()
    except Exception:
        pass
    try:
        from importlib.metadata import version
        info["vllm_version"] = version("vllm")
    except Exception:
        pass
    try:
        import subprocess
        root = str(Path(__file__).resolve().parents[2])
        out = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            info["vllm_commit"] = out.stdout.strip()[:12]
    except Exception:
        pass
    try:
        import subprocess
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            info["gpu"] = out.stdout.strip().splitlines()[0]
    except Exception:
        pass
    return info


def save_run(path: str, meta: Dict[str, Any], artifacts: List[Dict],
             results: Optional[Dict[str, Any]] = None) -> None:
    """Persist EVERYTHING needed to reproduce/replay a run: full config (meta:
    model, mode, seq_len, max_tokens, k_dim, block_hash, public_key, nonces,
    batch_size, stat_test), auto-captured env (timestamp, host, vLLM version, GPU),
    the artifacts (vector_b64 + k_points_steps + n_sphere_mismatches), and optional
    results (n_mismatch, fraud_detected, rate, timing)."""
    rec: Dict[str, Any] = {
        "meta": {"timestamp": datetime.now(timezone.utc).isoformat(), **env_info(), **meta},
        "artifacts": artifacts,
    }
    if results is not None:
        rec["results"] = results
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(rec, f, indent=2)


def load_run(path: str):
    """Load a saved run -> (meta, artifacts). Use the meta to reproduce or to drive
    a validation against precomputed artifacts without regenerating."""
    with open(path) as f:
        rec = json.load(f)
    return rec["meta"], rec["artifacts"]


def get_server_model(base_url: str, timeout: int = 10) -> str:
    """Fetch the first model name from /v1/models."""
    resp = requests.get(f"{base_url}/v1/models", timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data["data"][0]["id"]


def send_inference_request(
    base_url: str,
    block_hash: str,
    public_key: str,
    nonces: List[int],
    seq_len: int,
    max_tokens: int,
    model: str,
    timeout: int = 600,
) -> Dict[str, Any]:
    url = f"{base_url}/api/v1/pow/generate"
    payload = {
        "block_hash": block_hash,
        "block_height": 100,
        "public_key": public_key,
        "node_id": 0,
        "node_count": 1,
        "nonces": nonces,
        "params": {
            "model": model,
            "seq_len": seq_len,
            "k_dim": 12,
            "max_tokens": max_tokens,
        },
        "wait": True,
    }
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def send_validation_request(
    base_url: str,
    block_hash: str,
    public_key: str,
    nonces: List[int],
    seq_len: int,
    max_tokens: int,
    inference_sphere_k_steps: Dict[int, List[int]],
    model: str,
    timeout: int = 600,
) -> Dict[str, Any]:
    """Same as inference but includes inference_k_steps so the server tracks
    deviations without cascading."""
    url = f"{base_url}/api/v1/pow/generate"
    payload = {
        "block_hash": block_hash,
        "block_height": 100,
        "public_key": public_key,
        "node_id": 0,
        "node_count": 1,
        "nonces": nonces,
        "params": {
            "model": model,
            "seq_len": seq_len,
            "k_dim": 12,
            "max_tokens": max_tokens,
        },
        "wait": True,
        "enforced_k_steps": inference_sphere_k_steps,
    }
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def extract_artifact(response: Dict[str, Any], type: str="inference") -> Dict[str, Any]:
    all_nonces_artifacts = {}
    for art in response.get("artifacts", []):
        nonce = art.get("nonce")
        sphere_k_steps = art.get("k_points_steps")
        if nonce is not None and sphere_k_steps is not None:
            all_nonces_artifacts[nonce] = {}
            if type == "inference":
                all_nonces_artifacts[nonce] = sphere_k_steps
            elif type == "validation":
                n_sphere_mismatches = art.get("n_sphere_mismatches")
                all_nonces_artifacts[nonce]["sphere_k_steps"] = sphere_k_steps
                all_nonces_artifacts[nonce]["n_sphere_mismatches"] = n_sphere_mismatches
            else:
                raise ValueError(f"Unknown artifact type: {type}")
        else:
            raise ValueError(f"Nonce not found in response artifacts")

    return all_nonces_artifacts


def wait_for_server(url: str, timeout: int = 300, poll: float = 5.0):
    health_url = f"{url}/health"
    deadline = time.time() + timeout
    print(f"  Waiting for {url} ...", end="", flush=True)
    while time.time() < deadline:
        try:
            r = requests.get(health_url, timeout=5)
            if r.status_code == 200:
                print(" ready.")
                return
        except requests.RequestException:
            pass
        time.sleep(poll)
        print(".", end="", flush=True)
    print()
    raise TimeoutError(f"Server {url} did not become ready within {timeout}s")


def random_block_hash() -> str:
    return hashlib.sha256(os.urandom(32)).hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description="Compare PoC sphere_k_steps via inference+validation requests.")
    parser.add_argument("--inference", default="http://localhost:8000")
    parser.add_argument("--validation", default="http://localhost:8001")
    parser.add_argument("--n", type=int, default=1, help="Number of paired runs")
    parser.add_argument("--num-hashes", default=None, type=int)
    parser.add_argument("--block-hash", default="deadbeef" * 8)
    parser.add_argument("--public-key", default="cafebabe" * 8)
    parser.add_argument("--num-nonces", default=None, type=int)
    parser.add_argument("--nonce", type=int, default=42)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--output", default=None)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--no-wait", action="store_true")
    args = parser.parse_args()

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.results_dir) / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else run_dir / "artifacts.json"

    if not args.no_wait:
        print("Waiting for servers to be ready...")
        wait_for_server(args.inference)
        wait_for_server(args.validation)

    print("Fetching model names from servers...")
    inf_model = get_server_model(args.inference)
    val_model = get_server_model(args.validation)
    print(f"  inference  model: {inf_model}")
    print(f"  validation model: {val_model}")

    print(f"  inference  -> {args.inference}")
    print(f"  validation -> {args.validation}\n")

    runs: List[Dict[str, Any]] = []
    total_mismatches = 0

    if args.num_nonces is not None:
        nonces = list(range(args.num_nonces))
    else:
        nonces = [args.nonce]
    
    if args.num_hashes is not None:
        hashes = [random_block_hash() for _ in range(args.num_hashes)]
    else:
        hashes = [args.block_hash]

    for block_hash in hashes:
        for i in range(args.n):
            print(f"[{i+1}/{args.n}] Inference ...", end=" ", flush=True)
            t0 = time.perf_counter()
            resp_inf = send_inference_request(
                args.inference,
                block_hash,
                args.public_key,
                nonces,
                args.seq_len,
                args.max_tokens,
                inf_model,
            )
            t_inf = time.perf_counter() - t0
            inf_steps = extract_artifact(resp_inf)
            print(f"done ({t_inf:.2f}s)")

            print(f"[{i+1}/{args.n}] Validation ...", end=" ", flush=True)
            t0 = time.perf_counter()
            resp_val = send_validation_request(
                args.validation,
                block_hash,
                args.public_key,
                nonces,
                args.seq_len,
                args.max_tokens,
                inf_steps,
                val_model,
            )
            t_val = time.perf_counter() - t0
            val_art = extract_artifact(resp_val, type="validation")
            print(f"done ({t_val:.2f}s)")

            for nonce in nonces:
                n_mismatches_client = val_art.get(nonce, {}).get("n_sphere_mismatches", -1)
                total_mismatches += n_mismatches_client
                runs.append({
                    "run": i,
                    "nonce": nonce,
                    "block_hash": block_hash,
                    "inference_sphere_k_steps": inf_steps[nonce],
                    "validation_sphere_k_steps": val_art.get(nonce, {}).get("sphere_k_steps", []),
                    "inference_elapsed_s": t_inf,
                    "n_sphere_mismatches": n_mismatches_client,
                    "validation_elapsed_s": t_val,
                })

    print("=" * 60)
    print("SUMMARY")
    print(f"Total mismatches: {total_mismatches}")
    print(f"Total runs: {args.n*len(hashes)*len(nonces)}")

    all_inf_k = [k for r in runs for k in r["inference_sphere_k_steps"]]
    all_val_k = [k for r in runs for k in r["validation_sphere_k_steps"]]
    from collections import Counter
    print(f"\n  Inference k-distribution : {dict(sorted(Counter(all_inf_k).items()))}")
    print(f"  Validation k-distribution: {dict(sorted(Counter(all_val_k).items()))}")

    artifact = {
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_dir": str(run_dir),
            "inference_url": args.inference,
            "validation_url": args.validation,
            "max_tokens": args.max_tokens,
            "inference_model": inf_model,
            "validation_model": val_model,
        },
        "runs": runs,
    }

    with open(output_path, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"\nArtifacts saved to {output_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""collect.py — PoC data COLLECTION (client/server). One model, one mode, one file.

The tool's only job is to COLLECT data; the only difference is the mode:

  --mode generate            pure generation  -> k-trajectory artifacts
  --mode validate --ref F    validation with a supplied k-trajectory (from F) ->
                             teacher-forced n_sphere_mismatches

Both also collect perf (nonces/s) and FULL provenance (GPU, vLLM commit, cudagraph
mode, attention backend, dtype, quant, config). Deploys the model on a REMOTE ML
node (--url, the rented box runs node+vLLM) or boots vLLM locally for dev. Honest vs
fraud, the matrix, and A100-vs-H100 comparison are NOT done here — that's offline
analysis (analyze.py) over many collected files.

  collect.py --mode generate --model M               --save gen_M.json   [--url URL]
  collect.py --mode validate --model M --ref gen_X.json --save val_M_vs_X.json
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from poc_validation import (  # noqa: E402
    request_generate, save_run, load_run, add_engine_args, deploy_from_args,
)

KDIM = 12
BH, PK = "deadbeef" * 8, "cafebabe" * 8
try:  # frozen sphere codebook hash — the consensus reference the server asserts against
    from vllm.poc.sphere import EXPECTED_CODEBOOK_SHA256 as CODEBOOK_HASH
except Exception:  # pragma: no cover
    CODEBOOK_HASH = "unknown"


def cmd_generate(a):
    nonces = list(range(a.nonces))
    with deploy_from_args(a, a.model) as (url, srv):
        resp, secs = request_generate(url, target=a.target, model=a.model, nonces=nonces,
                                      block_hash=BH, public_key=PK, seq_len=a.seq_len,
                                      max_tokens=a.max_tokens, k_dim=KDIM)
    arts = resp["artifacts"]
    nps = len(nonces) / secs if secs else 0.0
    meta = {"role": "generate", "model": a.model, "seq_len": a.seq_len,
            "max_tokens": a.max_tokens, "k_dim": KDIM, "block_hash": BH, "public_key": PK,
            "codebook_hash": CODEBOOK_HASH, "nonces": nonces, "batch_size": 32, **a.prov}
    save_run(a.save, meta, arts,
             results={"nonces_per_s": round(nps, 3), "steps_per_s": round(nps * (a.max_tokens + 1), 1),
                      "elapsed_s": round(secs, 1)})
    print(f"generate {a.model}: {len(arts)} artifacts (traj {len(arts[0]['k_points_steps'])}), "
          f"{nps:.2f} nonces/s -> {a.save}")


def cmd_validate(a):
    rmeta, ref_arts = load_run(a.ref)
    nonces, mt, seq = rmeta["nonces"], rmeta["max_tokens"], rmeta["seq_len"]
    enforced = {x["nonce"]: x["k_points_steps"] for x in ref_arts}
    with deploy_from_args(a, a.model) as (url, srv):
        resp, secs = request_generate(url, target=a.target, model=a.model, nonces=nonces,
                                      block_hash=rmeta["block_hash"], public_key=rmeta["public_key"],
                                      seq_len=seq, max_tokens=mt, k_dim=rmeta["k_dim"],
                                      enforced_k=enforced, validation=ref_arts, p_mismatch=a.p_mismatch)
    rate = resp["n_mismatch"] / (len(nonces) * (mt + 1))
    nps = len(nonces) / secs if secs else 0.0
    honest = a.model == rmeta["model"]
    meta = {"role": "validate", "validator_model": a.model, "prover_model": rmeta["model"],
            "seq_len": seq, "max_tokens": mt, "k_dim": rmeta["k_dim"], "block_hash": rmeta["block_hash"],
            "public_key": rmeta["public_key"], "codebook_hash": rmeta.get("codebook_hash", CODEBOOK_HASH),
            "prover_gpu": rmeta.get("gpu"),  # HW the ref was generated on (cross-HW: != this validator's gpu)
            "nonces": nonces, "batch_size": 32, "ref": a.ref,
            "prover_engine": rmeta.get("engine"), "prover_profile": rmeta.get("profile"), **a.prov}
    save_run(a.save, meta, resp.get("artifacts", []),
             results={"validator_model": a.model, "prover_model": rmeta["model"], "honest": honest,
                      "rate": rate, "n_mismatch": resp["n_mismatch"], "fraud_detected": resp["fraud_detected"],
                      "per_nonce": resp.get("per_nonce", []),  # per-nonce mismatch counts (for charts)
                      "prover_gpu": rmeta.get("gpu"),  # prover HW (this run is the validator's HW)
                      "nonces_per_s": round(nps, 3), "elapsed_s": round(secs, 1)})
    print(f"validate {a.model} vs ref {rmeta['model']} ({'honest' if honest else 'fraud'}): "
          f"rate={rate*100:.3f}% fraud={resp['fraud_detected']} {nps:.2f} nonces/s -> {a.save}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["generate", "validate"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--ref", help="validate: generated file supplying the k-trajectory")
    add_engine_args(ap)
    # Defaults mirror production (vllm/config/cache.py): poc_max_batch_size=32,
    # poc_seq_len=256, poc_max_tokens=256. seq_len is PREFILL length only; decode
    # adds max_tokens on top, so the engine allocates seq_len+max_tokens KV upfront
    # (mixed_decode.py) and seq_len+max_tokens must stay <= --max-model-len (1024).
    # seq_len/max_tokens are artifact-defining: must match the deployed server config.
    ap.add_argument("--nonces", type=int, default=32)   # one production batch
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--p-mismatch", type=float, default=0.1)
    ap.add_argument("--save", required=True)
    a = ap.parse_args()
    if a.mode == "validate" and not a.ref:
        ap.error("--ref is required for --mode validate")
    (cmd_generate if a.mode == "generate" else cmd_validate)(a)


if __name__ == "__main__":
    main()

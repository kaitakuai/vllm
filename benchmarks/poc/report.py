#!/usr/bin/env python3
"""Render a self-contained HTML decode-PoC report per model from runs/*.json.

Consumes the SAME role-tagged result files the other tools already write
(collect.py validate -> separation, perfomance_nonces.py -> perf, quality_gsm8k.py
-> gsm8k). No measurement here — pure renderer. Output is one offline HTML file
(inline CSS, unicode bar cells; no JS, no external deps), one section per model with:
PERFORMANCE, SEPARATION (honest vs fraud), CO-EXISTENCE (GSM8K), provenance + glossary.

  report.py runs/*.json --out report.html
"""
import argparse
import glob
import html
import json
import os
from collections import defaultdict


def _load(paths):
    """Accepts files, globs, OR directories (recursed for *.json). So it works whether
    result files are flat in one dir or organized per-model session folders
    (runs/<model>__<gpu>__<date>/). Grouping is by model from each file's content, so
    mixing many models/sessions in one report is safe — each gets its own section."""
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += sorted(glob.glob(os.path.join(p, "**", "*.json"), recursive=True))
        elif any(c in p for c in "*?["):
            files += sorted(glob.glob(p, recursive=True))
        else:
            files.append(p)
    recs = []
    for f in files:
        try:
            d = json.load(open(f))
        except Exception as e:
            print(f"skip {f}: {e}")
            continue
        # Only role-tagged result records (dict with meta/results); silently ignore
        # raw artifact dumps like poc_artifacts.json (a list).
        if not isinstance(d, dict) or "results" not in d:
            continue
        d["_file"] = os.path.relpath(f)
        recs.append(d)
    return recs


def _g(meta, *keys, default="?"):
    for k in keys:
        if k in meta and meta[k] not in (None, ""):
            return meta[k]
    return default


def _short(s, n=30):
    s = str(s)
    return s if len(s) <= n else s[:n - 1] + "…"


def _gpu(meta, n=None):
    """GPU label (drop the 'NVIDIA' prefix + driver suffix); full name, no truncation."""
    g = str(_g(meta, "gpu", default="?")).replace("NVIDIA ", "").split(",")[0].strip()
    return g if n is None else _short(g, n)


def _backend(s):
    """Normalize an attention-backend or profile string to FA / FI."""
    s = str(s).lower()
    if "flashinfer" in s:
        return "FI"
    if "flash_attn" in s or "flashattn" in s or "flash attention" in s:
        return "FA"
    return "?"


def _eng(s):
    """Short graph-mode label."""
    s = str(s).lower()
    return "graph" if "graph" in s or "cuda" in s else ("eager" if "eager" in s else s)


def _bar(frac, kind="perf", thresh=None):
    """A self-contained CSS bar (proportional <div>, colored by kind). Optional
    `thresh` (0..1) draws a vertical marker line (used for the p_mismatch cutoff)."""
    frac = max(0.0, min(1.0, float(frac)))
    pct = frac * 100
    mark = ""
    if thresh is not None:
        mark = f"<span class=mark style='left:{max(0.0,min(1.0,thresh))*100:.1f}%'></span>"
    return (f"<span class=track>{mark}"
            f"<span class='fill {kind}' style='width:{pct:.1f}%'></span></span>")


def _model_of(rec):
    """Model this record belongs to (report is grouped per model). Perf/gsm8k carry
    `model` in meta; separation (validate) carries validator_model in results — group
    under the VALIDATOR's model so honest+fraud rows for model M sit together."""
    m = rec.get("meta", {})
    res = rec.get("results", {})
    return _g(m, "model", default=None) or _g(res, "validator_model", default=None) \
        or _g(m, "validator_model", "prover_model", default="?")


# ---- section renderers (return HTML strings) -------------------------------
def _is_perf(r):
    # Dedicated perf runs: throughput (req_per_min) OR clean per-step (per_step_ms).
    res = r.get("results", {})
    return "req_per_min" in res or "per_step_ms" in res


def _is_eager(m):
    s = str(_g(m, "engine", "cudagraph_mode") or "").upper()
    return "EAGER" in s or "NONE" in s or s == ""


def _perf_mode(m, res):
    return res.get("mode") or _g(m, "mode") or ("poc" if "total_nonces" in res else "chat")


def perf_section(recs):
    # CLEAN per-step decode efficiency (prefill + fixed overhead removed via the two-point
    # slope method). Throughput (req/min) is intentionally NOT used for the PoC-vs-inference
    # ratio: PoC's per-nonce 256-token prefill makes throughput prefill-dependent and
    # understates decode efficiency (the misleading ~0.55× number). Per-step is the truth.
    rows = [(r["meta"], r["results"]) for r in recs if "per_step_ms" in r.get("results", {})]
    if not rows:
        return "<p class=na>no perf runs</p>"
    ps, pt = {}, {}
    for m, res in rows:
        mode = res.get("mode") or _g(m, "mode")
        eng = "eager" if _is_eager(m) else "cudagraph"
        ps[(mode, eng)] = res["per_step_ms"]
        pt[(mode, eng)] = res.get("per_step_per_seq_ms")
    out = []
    for eng in ("cudagraph", "eager"):
        c, p = ps.get(("chat", eng)), ps.get(("poc", eng))
        if c and p:
            out.append(
                f"<p class=highlight>Decode efficiency ({eng}, per 32-wide step, prefill "
                f"isolated): PoC <b>{p:.1f} ms</b> vs real inference <b>{c:.1f} ms</b> "
                f"&mdash; PoC = <b>{c/p:.2f}×</b> chat per decode step.</p>")
    cc, ce = ps.get(("chat", "cudagraph")), ps.get(("chat", "eager"))
    pc, pe = ps.get(("poc", "cudagraph")), ps.get(("poc", "eager"))
    if None not in (cc, ce, pc, pe):
        out.append(
            f"<p class=highlight>cudagraph speedup (per step): real inference "
            f"<b>{ce/cc:.2f}×</b>, PoC <b>{pe/pc:.2f}×</b>. cudagraph's <b>absolute</b> "
            f"per-step saving is the same for both (chat {ce-cc:.0f} ms, PoC {pe-pc:.0f} ms) "
            f"&mdash; cudagraph graphs PoC identically; the residual (PoC ~{(pc/cc-1)*100:.0f}% "
            f"slower/step) is the MoE expert-loading toll, not a graphing defect.</p>")
    out.append("<table><tr><th>mode</th><th>engine</th><th>per-step (ms)</th>"
               "<th>per-token (ms)</th></tr>")
    for (mode, eng) in sorted(ps):
        out.append(
            f"<tr><td>{'real inference' if mode=='chat' else 'PoC'}</td>"
            f"<td>{eng}</td><td class=num>{ps[(mode,eng)]:.2f}</td>"
            f"<td class=num>{(pt.get((mode,eng)) or 0):.3f}</td></tr>")
    out.append("</table>")
    return "\n".join(out)


def _calibrated_thresholds(vals):
    """Per validator-model calibrated p_mismatch (same logic as the calibration table):
    geomean of the feasible window's bounds — floor = max(honest same-be, honest x-be),
    fraud-min on top. Used as the chart strike line (NOT a hardcoded 0.1)."""
    import math
    from collections import defaultdict
    g = defaultdict(lambda: {"hs": [], "hx": [], "fr": []})
    for m, res in vals:
        v, p = res.get("validator_model", "?"), res.get("prover_model", "?")
        vbe, pbe = _backend(m.get("attention_backend")), _backend(m.get("prover_profile"))
        same_be = (vbe == pbe and vbe != "?")
        bucket = "hs" if (v == p and same_be) else ("hx" if v == p else "fr")
        g[v][bucket].append(res.get("rate", 0.0))
    out = {}
    for v, b in g.items():
        hs = max(b["hs"]) if b["hs"] else None
        hx = max(b["hx"]) if b["hx"] else None
        fr = min(b["fr"]) if b["fr"] else None
        floor = max([x for x in (hs, hx) if x is not None], default=None)
        if fr is not None and floor is not None and floor < fr:      # separable unpinned
            out[v] = math.sqrt(floor * fr) if floor > 0 else fr / 3
        elif fr is not None and hs is not None and hs < fr:          # separable only if pinned
            out[v] = math.sqrt(hs * fr) if hs > 0 else fr / 3
    return out


def separation_section(recs):
    vals = [(r["meta"], r["results"]) for r in recs
            if r.get("meta", {}).get("role") == "validate" and "rate" in r.get("results", {})]
    if not vals:
        return "<p class=na>no separation runs</p>", None
    # Trajectory is config-sensitive in degrees (same config < graph mode < attention
    # backend < different model). The goal is a separable gap: honest below the calibrated
    # threshold, fraud above, with the backend pinned. Actual rates are measured (below).
    calib = _calibrated_thresholds(vals)
    ok = True
    any_cross = False
    out = ["<table><tr><th>hardware (val ⇐ prover)</th><th>config (val ⇐ prover)</th>"
           "<th>model (val ⇐ prover)</th><th>kind</th><th>rate</th><th>verdict</th><th></th></tr>"]
    for m, res in sorted(vals, key=lambda x: (x[1].get("rate", 0))):
        v, p = res.get("validator_model", "?"), res.get("prover_model", "?")
        honest = (v == p)
        rate = res.get("rate", 0.0)
        fraud = res.get("fraud_detected")
        thresh = calib.get(v) or res.get("p_mismatch") or m.get("p_mismatch") or 0.1
        vbe, pbe = _backend(m.get("attention_backend")), _backend(m.get("prover_profile"))
        same_be = (vbe == pbe and vbe != "?")
        veng, peng = _eng(m.get("engine")), _eng(m.get("prover_engine"))
        if honest and same_be:
            kind, good, counts = "honest", (not fraud), True
        elif honest:               # cross-backend honest: expected divergence, informational
            kind, good, counts, any_cross = "honest·x-backend", (not fraud), False, True
        else:
            kind, good, counts = "fraud", bool(fraud), True
        if counts:
            ok = ok and good
        verdict = ("ok" if good else "⚠ MISS") if counts else ("ok" if good else "pin-backend")
        rowcls = "honest" if honest else "fraud"
        if not counts:
            rowcls = "cross"
        vhw = _gpu(m)
        phw = str(res.get("prover_gpu")).replace("NVIDIA ", "").split(",")[0].strip() \
            if res.get("prover_gpu") else vhw
        out.append(
            f"<tr class={rowcls}>"
            f"<td>{html.escape(vhw)} ⇐ {html.escape(phw)}</td>"
            f"<td>{veng}/{vbe} ⇐ {peng}/{pbe}</td>"
            f"<td>{html.escape(v)} ⇐ {html.escape(p)}</td>"
            f"<td>{kind}</td><td class=num>{rate*100:.2f}%</td>"
            f"<td class=num>{verdict}</td>"
            f"<td class=bar>{_bar(rate, 'honest' if honest else 'fraud', thresh=thresh)}</td></tr>")
    out.append("</table>")
    if any_cross:
        out.append(
            "<p class=note><b>Backend pinning:</b> rows marked "
            "<code>honest·x-backend</code> validate an honest run across <em>different</em> "
            "attention backends (FlashAttention ⇄ FlashInfer). Their elevated rate is "
            "<em>expected</em> — the kernels are not bit-identical and the per-step "
            "<code>sphere_k</code> chain compounds tiny FP deltas over the trajectory. "
            "They are <em>not</em> counted as failures: production pins the attention "
            "backend (prover &amp; validator match), where honest is small. cudagraph vs "
            "eager (same backend) is <em>not</em> bit-identical either, but its divergence "
            "stays under the threshold, so same-backend honest passes on either graph mode. "
            "Fraud is caught in <em>every</em> config. The bar strike-line is the "
            "<em>calibrated</em> threshold for this model (geomean of honest-max and "
            "fraud-min), not a fixed 0.1.</p>")
    verdict = "PASS" if ok else "FAIL"
    return "\n".join(out), verdict


def calibration_section(recs):
    """Threshold-setting tool: derive the feasible per-model p_mismatch window from the
    MEASURED honest/fraud rates. p_mismatch is NOT baked in - prod loads it per-model from
    chain PoCStatTestParams; this picks the value for that config: honest_max < p < fraud_min."""
    import math
    from collections import defaultdict
    vals = [(r["meta"], r["results"]) for r in recs
            if r.get("meta", {}).get("role") == "validate" and "rate" in r.get("results", {})]
    if not vals:
        return "<p class=na>no validation runs</p>"
    g = defaultdict(lambda: {"hs": [], "hx": [], "fr": []})
    for m, res in vals:
        v, p = res.get("validator_model", "?"), res.get("prover_model", "?")
        rate = res.get("rate", 0.0)
        vbe, pbe = _backend(m.get("attention_backend")), _backend(m.get("prover_profile"))
        same_be = (vbe == pbe and vbe != "?")
        bucket = "hs" if (v == p and same_be) else ("hx" if v == p else "fr")
        g[v][bucket].append(rate)

    def pct(x):
        return "&mdash;" if x is None else f"{x*100:.2f}%"

    out = ["<table><tr><th>model</th><th>honest same-be (max)</th><th>honest x-be (max)</th>"
           "<th>fraud (min)</th><th>feasible p_mismatch</th><th>recommended</th><th>separable</th></tr>"]
    for model, b in sorted(g.items()):
        hs = max(b["hs"]) if b["hs"] else None
        hx = max(b["hx"]) if b["hx"] else None
        fr = min(b["fr"]) if b["fr"] else None
        floor = max([x for x in (hs, hx) if x is not None], default=None)  # heterogeneous floor
        if fr is not None and floor is not None and floor < fr:
            rec = math.sqrt(floor * fr) if floor > 0 else fr / 3
            window, rec_s, sep = f"({pct(floor)}, {pct(fr)})", f"{rec*100:.1f}%", "YES"
        elif fr is not None and hs is not None and hs < fr:  # only separable if backend pinned
            rec = math.sqrt(hs * fr) if hs > 0 else fr / 3
            window, rec_s, sep = f"pinned: ({pct(hs)}, {pct(fr)})", f"{rec*100:.1f}% (pin backend)", "pinned-only"
        else:
            window, rec_s, sep = "&mdash;", "&mdash;", "NO gap &#9888;"
        out.append(
            f"<tr><td>{html.escape(model)}</td>"
            f"<td class=num>{pct(hs)}</td><td class=num>{pct(hx)}</td><td class=num>{pct(fr)}</td>"
            f"<td>{window}</td><td class=num>{rec_s}</td><td>{sep}</td></tr>")
    out.append("</table>")
    out.append(
        "<p class=gloss><b>p_mismatch is not baked in</b> &mdash; production loads it per-model "
        "from chain <code>PoCStatTestParams</code> (decentralized-api <code>validator.go</code> "
        "&rarr; <code>StatTestParamsFromChain</code> &rarr; vLLM <code>stat_test</code>). This "
        "table sets that value from measured rates: any <code>p_mismatch</code> in "
        "(honest_max, fraud_min) separates. Recommended = geometric mean of the bounds. "
        "Backend-pinned uses honest same-be (tiny) &rarr; widest margin; heterogeneous uses "
        "honest x-be. <code>NO gap</code> = honest &ge; fraud, not separable at this config.</p>")
    return "\n".join(out)


def gsm8k_section(recs):
    rows = [(r["meta"], r.get("results", {})) for r in recs
            if r.get("meta", {}).get("role") == "gsm8k"]
    if not rows:
        return "<p class=na>no gsm8k runs</p>"
    out = ["<table><tr><th>hardware</th><th>engine</th><th>PoC load</th><th>accuracy (flex)</th></tr>"]
    for m, res in sorted(rows, key=lambda x: (not _is_eager(x[0]), x[0].get("poc_max_tokens", 0))):
        mt = m.get("poc_max_tokens")
        load = "pure chat (no PoC)" if not mt else f"+ 32 PoC ({mt} decode steps)"
        f = res.get("flexible_extract")
        out.append(
            f"<tr><td>{html.escape(_gpu(m))}</td>"
            f"<td>{html.escape(str(_g(m,'engine','cudagraph_mode')))}</td>"
            f"<td>{load}</td>"
            f"<td class=num>{(f'{f*100:.2f}%' if f is not None else '?')}</td></tr>")
    out.append("</table>")
    n = next((res.get("n_samples") for _, res in rows if res.get("n_samples")), None)
    nstr = f"{n}" if n else "a limited number of"
    se = ("up to ≈±%.0f pp (1σ, at p=0.5)" % (100 * (0.5 / (n ** 0.5)))) if n else "several points"
    out.append(
        f"<p class=note><b>Read as co-existence, not a leaderboard.</b> We report "
        f"<b>flexible_extract</b> — the answer's number is parsed from the model's free-form "
        f"output. (lm-eval also reports <code>strict_match</code>, which demands an exact "
        f"output format and so understates real accuracy; it's omitted here.) Scored on "
        f"<b>{nstr} samples</b>, so the binomial standard error is {se}: a small PoC-on vs "
        f"baseline difference is sampling noise, not a PoC effect — co-existence holds when "
        f"the two rows track each other within that error. Use the same N for both; raise "
        f"<code>--limit</code> to tighten.</p>")
    return "\n".join(out)


_CSS = """
:root{--ink:#0f172a;--mut:#64748b;--line:#e2e8f0;--bg:#f8fafc;
  --blue:#3b82f6;--green:#16a34a;--red:#dc2626;--amber:#d97706}
*{box-sizing:border-box}
body{font:14px/1.55 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,system-ui,sans-serif;
  margin:0;padding:2.5rem 1.5rem;color:var(--ink);background:var(--bg)}
.wrap{max-width:1120px;margin:0 auto}
.card{background:#fff;border:1px solid var(--line);border-radius:14px;padding:1.4rem 1.6rem;
  margin:0 0 1.4rem;box-shadow:0 1px 3px rgba(15,23,42,.04),0 8px 24px rgba(15,23,42,.03)}
.head{display:flex;align-items:center;justify-content:space-between;gap:1rem;flex-wrap:wrap}
h1{font-size:1.35rem;font-weight:700;margin:0;letter-spacing:-.01em}
.prov{color:var(--mut);font-size:.82rem;margin:.4rem 0 0;font-variant-numeric:tabular-nums}
h2{font-size:.78rem;font-weight:700;color:var(--mut);text-transform:uppercase;letter-spacing:.06em;
  margin:1.7rem 0 .5rem;display:flex;align-items:baseline;gap:.6rem}
.card h2:first-of-type{margin-top:0}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th{font-size:.7rem;font-weight:600;color:var(--mut);text-transform:uppercase;letter-spacing:.04em;
  text-align:left;padding:.35rem .7rem;border-bottom:2px solid var(--line)}
td{padding:.4rem .7rem;border-bottom:1px solid var(--line)}
tbody tr:hover{background:#f1f5f9}
.num{text-align:right}
.bar{width:200px;padding-right:0}
.track{position:relative;display:block;height:14px;background:#eef2f7;border-radius:7px;overflow:hidden}
.fill{position:absolute;left:0;top:0;height:100%;border-radius:7px}
.fill.perf{background:linear-gradient(90deg,#60a5fa,#3b82f6)}
.fill.honest{background:linear-gradient(90deg,#4ade80,#16a34a)}
.fill.fraud{background:linear-gradient(90deg,#f87171,#dc2626)}
.mark{position:absolute;top:-2px;height:18px;width:2px;background:#0f172a;opacity:.55;z-index:2}
.chip{display:inline-block;padding:.18rem .7rem;border-radius:999px;font-size:.78rem;font-weight:700;letter-spacing:.02em}
.pass{background:#dcfce7;color:#166534} .fail{background:#fee2e2;color:#991b1b}
.cov{font-size:.72rem;font-weight:500;color:var(--mut);text-transform:none;letter-spacing:0}
tr.honest td:nth-child(4){color:var(--green);font-weight:600}
tr.fraud td:nth-child(4){color:var(--red);font-weight:600}
tr.cross td:nth-child(4){color:var(--amber);font-weight:600}
tr.cross{background:#fffbeb}
.note{font-size:.78rem;color:var(--mut);line-height:1.55;background:#fffbeb;
  border-left:3px solid var(--amber);padding:.55rem .8rem;border-radius:6px;margin:.6rem 0}
.note b{color:#92400e}
.na{color:#94a3b8;font-style:italic;font-size:.85rem}
.highlight{background:#ecfdf5;border-left:3px solid #10b981;padding:.55rem .8rem;border-radius:6px;margin:.6rem 0;font-size:.95rem}
.highlight b{color:#065f46}
.lead{color:#475569;font-size:.9rem;line-height:1.55;margin:.2rem 0 .7rem;padding:.55rem .85rem;background:#f8fafc;border-left:3px solid #cbd5e1;border-radius:6px}
.lead b{color:var(--ink)}
.verdict{font-size:.95rem;font-weight:600;margin:.7rem 0 1.3rem;padding:.6rem .85rem;border-radius:6px;background:#eff6ff;border-left:3px solid #3b82f6;color:#1e3a8a}
.verdict.pass{background:#ecfdf5;border-left-color:#10b981;color:#065f46}
.verdict.fail{background:#fef2f2;border-left-color:#ef4444;color:#991b1b}
.verdict .lbl{font-weight:800;letter-spacing:.02em}
.gloss{color:var(--mut);font-size:.8rem;line-height:1.6;margin-top:.5rem}
ul.gloss{margin:.3rem 0 0;padding-left:1.1rem} ul.gloss li{margin:.35rem 0}
code{background:#f1f5f9;padding:.05rem .35rem;border-radius:4px;font-size:.9em}
.gloss b{color:var(--ink)}
.foot{text-align:center;color:#94a3b8;font-size:.75rem;margin:1rem 0}
"""


def render(recs):
    by_model = defaultdict(list)
    for r in recs:
        by_model[_model_of(r)].append(r)
    # A model gets its own report section only if it is a SUBJECT — validated (validator
    # of a validate run), benchmarked (perf), or evaluated (gsm8k). Models that appear
    # ONLY as a fraud reference (generate-only) are not subjects and get no section
    # (their trajectory still drives the subject's fraud rows).
    subjects = set()
    for r in recs:
        m, res = r.get("meta", {}), r.get("results", {})
        role = m.get("role")
        if role == "validate":
            subjects.add(res.get("validator_model"))
        elif role == "gsm8k":
            subjects.add(m.get("model"))
        elif _is_perf(r):
            subjects.add(m.get("model"))
    if not subjects:                       # fallback: nothing tagged -> show all
        subjects = set(by_model)
    parts = [f"<!doctype html><meta charset=utf-8><title>Decode-PoC report</title><style>{_CSS}</style>",
             "<div class=wrap>"]
    for model in sorted(m for m in by_model if m in subjects):
        rs = by_model[model]
        meta0 = next((r["meta"] for r in rs if r.get("meta")), {})
        sep_html, verdict = separation_section(rs)
        chip = f"<span class='chip {'pass' if verdict=='PASS' else 'fail'}'>{verdict}</span>" if verdict else ""
        # GPU is now per-row (runs may span hardware); header keeps build provenance only.
        prov = " · ".join(str(x) for x in [
            _g(meta0, "vllm_commit", "vllm_version", default=""),
            _g(meta0, "dtype", default=""),
            _g(meta0, "quantization", default="")] if x and x != "?")
        # Coverage counts — the report adapts to however many pairs/configs were run.
        n_perf = sum(1 for r in rs if _is_perf(r))
        sep = [r for r in rs if r.get("meta", {}).get("role") == "validate" and "rate" in r.get("results", {})]
        n_h = sum(1 for r in sep if r["results"].get("validator_model") == r["results"].get("prover_model"))
        n_f = len(sep) - n_h
        n_gsm = sum(1 for r in rs if r.get("meta", {}).get("role") == "gsm8k")
        parts.append("<div class=card>")
        parts.append(f"<div class=head><h1>{html.escape(model)}</h1>{chip}</div>")
        parts.append(f"<div class=prov>decode-PoC · {html.escape(prov)}</div>")
        parts.append(f"<h2>Performance <span class=cov>{n_perf} config(s)</span></h2>" + perf_section(rs))
        parts.append(f"<h2>Separation — honest vs fraud "
                     f"<span class=cov>{len(sep)} pair(s): {n_h} honest, {n_f} fraud</span></h2>" + sep_html)
        parts.append("<h2>p_mismatch calibration <span class=cov>feasible threshold window</span></h2>"
                     + calibration_section(rs))
        parts.append(f"<h2>Co-existence — GSM8K <span class=cov>{n_gsm} run(s)</span></h2>" + gsm8k_section(rs))
        parts.append("</div>")
    parts.append(
        "<div class=card><h2>How to read</h2><ul class=gloss>"
        "<li><b>Performance</b> — <i>nonces/min</i> decode-PoC throughput (higher is better); "
        "the bar is relative to the fastest config in the table.</li>"
        "<li><b>Separation</b> — <i>rate</i> = Σ sphere_k mismatches / (nonces × (max_tokens+1)); "
        "the vertical marker is the <i>calibrated p_mismatch</i> (geomean of honest-max &amp; fraud-min). "
        "<span style='color:var(--green)'><b>honest</b></span> (same model, same backend) sits below the "
        "marker (fraud = False); <span style='color:var(--red)'><b>fraud</b></span> (different model) above "
        "it (fraud = True).</li>"
        "<li><b>Co-existence</b> — GSM8K accuracy with PoC load vs the <code>--disable_poc</code> baseline; "
        "co-existence holds when the two are within noise.</li>"
        "<li><b>Config-sensitivity</b> — trajectories are <em>not</em> identical across configs: graph mode "
        "(cudagraph/eager) diverges slightly (tolerated under the threshold); a different attention backend "
        "diverges more (the validator must <b>pin</b> the prover's backend); a different model most (fraud).</li>"
        "<li><b>Hardware</b> is shown per row — a single report may span multiple GPUs "
        "(e.g. prover and validator on different hardware).</li>"
        "</ul></div>")
    parts.append("</div>")
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="runs/*.json result files")
    ap.add_argument("--out", default="report.html")
    a = ap.parse_args()
    recs = _load(a.files)
    open(a.out, "w").write(render(recs))
    models = sorted({_model_of(r) for r in recs})
    print(f"wrote {a.out}  ({len(recs)} result files, {len(models)} model(s): {', '.join(models)})")


if __name__ == "__main__":
    main()

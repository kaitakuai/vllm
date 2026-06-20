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
            d["_file"] = os.path.relpath(f)
            recs.append(d)
        except Exception as e:
            print(f"skip {f}: {e}")
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
    # Dedicated perf runs (perfomance_nonces.py) carry total_nonces; generate/validate
    # records also have nonces_per_s but are not perf measurements.
    return "total_nonces" in r.get("results", {})


def perf_section(recs):
    rows = [(r["meta"], r.get("results", {})) for r in recs if _is_perf(r)]
    if not rows:
        return "<p class=na>no perf runs</p>"
    mx = max(res.get("nonces_per_s", 0) for _, res in rows) or 1
    out = ["<table><tr><th>hardware</th><th>engine</th><th>attention</th><th>max_tok</th>"
           "<th>nonces/min</th><th>steps/s</th><th></th></tr>"]
    for m, res in sorted(rows, key=lambda x: -x[1].get("nonces_per_s", 0)):
        npm = res.get("nonces_per_s", 0) * 60
        out.append(
            f"<tr><td>{html.escape(_gpu(m))}</td>"
            f"<td>{html.escape(str(_g(m,'engine','cudagraph_mode')))}</td>"
            f"<td>{html.escape(str(_g(m,'attention_backend')))}</td>"
            f"<td>{_g(m,'max_tokens')}</td>"
            f"<td class=num>{npm:.0f}</td>"
            f"<td class=num>{res.get('steps_per_s',0):.0f}</td>"
            f"<td class=bar>{_bar(res.get('nonces_per_s',0)/mx, 'perf')}</td></tr>")
    out.append("</table>")
    return "\n".join(out)


def separation_section(recs):
    vals = [(r["meta"], r["results"]) for r in recs
            if r.get("meta", {}).get("role") == "validate" and "rate" in r.get("results", {})]
    if not vals:
        return "<p class=na>no separation runs</p>", None
    ok = True
    out = ["<table><tr><th>hardware (val ⇐ prover)</th><th>validator ⇐ prover</th>"
           "<th>kind</th><th>rate</th><th>fraud?</th><th></th></tr>"]
    for m, res in sorted(vals, key=lambda x: (x[1].get("rate", 0))):
        v, p = res.get("validator_model", "?"), res.get("prover_model", "?")
        honest = (v == p)
        rate = res.get("rate", 0.0)
        fraud = res.get("fraud_detected")
        good = (honest and not fraud) or ((not honest) and fraud)
        ok = ok and good
        kind = "honest" if honest else "fraud"
        flag = "" if good else " ⚠"
        # validator HW from this record's meta; prover HW from the ref (results.prover_gpu)
        vhw = _gpu(m)
        phw = str(res.get("prover_gpu")).replace("NVIDIA ", "").split(",")[0].strip() \
            if res.get("prover_gpu") else vhw
        out.append(
            f"<tr class={'honest' if honest else 'fraud'}>"
            f"<td>{html.escape(vhw)} ⇐ {html.escape(phw)}</td>"
            f"<td>{html.escape(_short(v,20))} ⇐ {html.escape(_short(p,20))}</td>"
            f"<td>{kind}</td><td class=num>{rate*100:.2f}%</td>"
            f"<td class=num>{fraud}{flag}</td>"
            f"<td class=bar>{_bar(rate, 'honest' if honest else 'fraud', thresh=res.get('p_mismatch', m.get('p_mismatch', 0.1)))}</td></tr>")
    out.append("</table>")
    verdict = "PASS" if ok else "FAIL"
    return "\n".join(out), verdict


def gsm8k_section(recs):
    rows = [(r["meta"], r.get("results", {})) for r in recs
            if r.get("meta", {}).get("role") == "gsm8k"]
    if not rows:
        return "<p class=na>no gsm8k runs</p>"
    out = ["<table><tr><th>hardware</th><th>engine</th><th>PoC load</th><th>strict</th><th>flex</th></tr>"]
    for m, res in sorted(rows, key=lambda x: x[0].get("poc_max_tokens", 0)):
        load = "off (baseline)" if not m.get("poc_max_tokens") else f"on (mt={m.get('poc_max_tokens')})"
        s, f = res.get("strict_match"), res.get("flexible_extract")
        out.append(
            f"<tr><td>{html.escape(_gpu(m))}</td>"
            f"<td>{html.escape(str(_g(m,'engine','cudagraph_mode')))}</td>"
            f"<td>{load}</td>"
            f"<td class=num>{(f'{s*100:.2f}%' if s is not None else '?')}</td>"
            f"<td class=num>{(f'{f*100:.2f}%' if f is not None else '?')}</td></tr>")
    out.append("</table>")
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
tr.honest td:nth-child(3){color:var(--green);font-weight:600}
tr.fraud td:nth-child(3){color:var(--red);font-weight:600}
.na{color:#94a3b8;font-style:italic;font-size:.85rem}
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
        parts.append(f"<div class=head><h1>{html.escape(_short(model,50))}</h1>{chip}</div>")
        parts.append(f"<div class=prov>decode-PoC · {html.escape(prov)}</div>")
        parts.append(f"<h2>Performance <span class=cov>{n_perf} config(s)</span></h2>" + perf_section(rs))
        parts.append(f"<h2>Separation — honest vs fraud "
                     f"<span class=cov>{len(sep)} pair(s): {n_h} honest, {n_f} fraud</span></h2>" + sep_html)
        parts.append(f"<h2>Co-existence — GSM8K <span class=cov>{n_gsm} run(s)</span></h2>" + gsm8k_section(rs))
        parts.append("</div>")
    parts.append(
        "<div class=card><h2>How to read</h2><ul class=gloss>"
        "<li><b>Performance</b> — <i>nonces/min</i> decode-PoC throughput (higher is better); "
        "the bar is relative to the fastest config in the table.</li>"
        "<li><b>Separation</b> — <i>rate</i> = Σ sphere_k mismatches / (nonces × (max_tokens+1)); "
        "the vertical marker on the bar is the <i>p_mismatch</i> cutoff. "
        "<span style='color:var(--green)'><b>honest</b></span> (same model) must be ≈ 0 and fraud = False; "
        "<span style='color:var(--red)'><b>fraud</b></span> (different model) must be above the cutoff and fraud = True.</li>"
        "<li><b>Co-existence</b> — GSM8K accuracy with PoC load vs the <code>--disable_poc</code> baseline; "
        "co-existence holds when the two are within noise.</li>"
        "<li><b>Config-invariance</b> — engine (eager / cudagraph) and attention (FlashAttention / FlashInfer) "
        "produce byte-identical trajectories, so separation does not depend on configuration.</li>"
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

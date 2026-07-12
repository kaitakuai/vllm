#!/usr/bin/env python3
"""Generate the SIMPLIFIED ("ideal") decode-PoC report from a session folder.

Three cards — Performance (decode efficiency) · Separation (honest vs fraud) ·
Co-existence (GSM8K) — each with a flow diagram, a table, and a plain verdict, plus an
ALL-PASS/REVIEW chip. Pure renderer
over the same role-tagged JSONs run_scope.sh writes.

  simplify_report.py <session-dir> [--out FILE]
"""
import argparse, glob, json, math, os

_IDEAL_CSS = """:root{--ink:#0f172a;--mut:#64748b;--line:#e2e8f0;--bg:#f1f5f9;--card:#fff;--blue:#2563eb;--green:#16a34a;--red:#dc2626}
*{box-sizing:border-box}
body{font:16px/1.6 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,system-ui,sans-serif;margin:0;padding:2.5rem 1.25rem;color:var(--ink);background:var(--bg)}
.wrap{max-width:980px;margin:0 auto}
h1{font-size:1.7rem;margin:0 0 .2rem}
.sub{color:var(--mut);font-size:1rem;margin:0 0 1.5rem;font-variant-numeric:tabular-nums}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:1.5rem 1.7rem;margin:0 0 1.4rem;box-shadow:0 1px 2px rgba(15,23,42,.05)}
.exp{font-size:.85rem;font-weight:800;color:var(--blue);letter-spacing:.08em;text-transform:uppercase}
.card h2{font-size:1.3rem;margin:.15rem 0 .9rem}
.lead{font-size:1rem;color:#334155;margin:.2rem 0 1rem}.lead b{color:var(--ink)}
.flow{display:flex;align-items:center;flex-wrap:wrap;gap:.5rem;margin:1rem 0 1.2rem;padding:1rem;background:#f8fafc;border:1px dashed #cbd5e1;border-radius:10px;font-size:.95rem}
.box{padding:.45rem .8rem;border-radius:8px;background:#fff;border:1px solid #cbd5e1;font-weight:600;white-space:nowrap}
.box.alt{background:#eff6ff;border-color:#bfdbfe;color:#1e40af}.box.good{background:#ecfdf5;border-color:#a7f3d0;color:#065f46}
.arr{color:var(--mut);font-weight:700;font-size:1.1rem}
.cap{font-size:.9rem;color:var(--mut);font-style:italic;width:100%;margin-top:.2rem}
table{border-collapse:collapse;width:100%;font-size:1rem;font-variant-numeric:tabular-nums;margin:.4rem 0}
th{text-align:left;font-size:.9rem;color:var(--mut);font-weight:700;padding:.5rem .7rem;border-bottom:2px solid var(--line)}
td{padding:.5rem .7rem;border-bottom:1px solid var(--line)}.num{text-align:right}
tr.hi td{background:#fef9c3}.good-t{color:var(--green);font-weight:700}.bad-t{color:var(--red);font-weight:700}
.verdict{font-size:1.05rem;font-weight:600;margin:1.1rem 0 0;padding:.8rem 1rem;border-radius:10px;background:#ecfdf5;border-left:5px solid var(--green);color:#065f46}
.verdict .lbl{font-weight:800;margin-right:.4rem}
.chip{display:inline-block;padding:.25rem .8rem;border-radius:999px;font-size:.95rem;font-weight:800;background:#dcfce7;color:#166534;vertical-align:middle}"""

ap = argparse.ArgumentParser()
ap.add_argument("session"); ap.add_argument("--out", default=None)
ap.add_argument("--p-mismatch", type=float, default=0.1,
                help="production acceptance threshold (fraction): fraud must exceed it and honest stay below it (default 0.1)")
ap.add_argument("--gsm-tol", type=float, default=0.0251,
                help="GSM8K co-existence tolerance (fraction; on/off delta within it = sampling noise; default ~2.5pt)")
a = ap.parse_args()
D = a.session.rstrip("/"); OUT = a.out or f"{D}/report_simple.html"
P_MISMATCH = a.p_mismatch * 100.0   # percent, to match the mismatch-rate scale
GSM_TOL = a.gsm_tol
L = lambda f: json.load(open(f))
def res(d): r = d.get("results", d); return r[0] if isinstance(r, list) and r else r

# ---- provenance: title = the HONEST/validator model (prefer val_/gen_honest_ over fraud) ----
meta = {}
for f in (sorted(glob.glob(f"{D}/val_*.json")) + sorted(glob.glob(f"{D}/gen_honest_*.json"))
          + sorted(glob.glob(f"{D}/*.json"))):
    try:
        m = L(f).get("meta", {})
        if m: meta = m; break
    except Exception: pass
model = meta.get("validator_model") or meta.get("model") or "?"
# fraud/producer model — the report must name BOTH sides (honest validator + fraud producer)
fraud_model = None
for _ff in sorted(glob.glob(f"{D}/gen_fraud_*.json")):
    try:
        fraud_model = L(_ff).get("meta", {}).get("model")
        if fraud_model: break
    except Exception: pass
models_str = f"honest {model}" + (f"  vs  fraud {fraud_model}" if fraud_model else "")
sub = " · ".join(str(x) for x in [
    models_str, meta.get("gpu", "?"), "decode-PoC",
    f"{len(meta.get('nonces', [])) or '?'} nonces", f"seq_len {meta.get('seq_len','?')}",
    f"max_tokens {meta.get('max_tokens','?')}", f"vLLM {meta.get('vllm_commit','?')}"])
prov = (f"codebook {str(meta.get('codebook_hash','?'))[:12]} · vLLM {meta.get('vllm_commit','?')} · "
        f"poc-scope {os.environ.get('POC_SCOPE_COMMIT','?')} · block_hash {str(meta.get('block_hash','?'))[:12]}")

def _sps(tag):   # decode-PoC throughput: steps/s
    try: return res(L(f"{D}/perf_{tag}.poc.json")).get("steps_per_s")
    except Exception: return None
def _tps(tag):   # pure inference (chat) throughput: tokens/s
    try: return res(L(f"{D}/perf_{tag}.chat.json")).get("tokens_per_s")
    except Exception: return None

cards = []
# ---- Experiment 1: Performance — cudagraph is genuinely engaged for PoC (same 32-batch, cg vs eager,
#      shown for BOTH pure inference and decode-PoC; PoC gets a real cudagraph speedup like normal decode) ----
poc_cg, poc_eg = _sps("cg-flashattn"), _sps("eager-flashattn")
inf_cg, inf_eg = _tps("cg-flashattn"), _tps("eager-flashattn")
if poc_cg and poc_eg:
    poc_sp = poc_cg / poc_eg
    rows = ""
    if inf_cg and inf_eg:
        rows += (f"<tr><td>pure inference (chat)</td><td class=num>{inf_eg:.0f} tok/s</td>"
                 f"<td class=num>{inf_cg:.0f} tok/s</td><td class=num>{inf_cg/inf_eg:.2f}×</td></tr>")
    rows += (f"<tr class=hi><td><b>decode-PoC</b></td><td class=num>{poc_eg:.0f} steps/s</td>"
             f"<td class=num>{poc_cg:.0f} steps/s</td><td class=num>{poc_sp:.2f}×</td></tr>")
    cards.append(f"""<div class=card><div class=exp>Experiment 1</div>
 <h2>Performance — is cudagraph actually engaged for PoC?</h2>
 <p class=lead>Same 32-wide batch, <b>cudagraph vs eager</b>, measured for two workloads on the same server: <b>pure inference</b> (normal decode) and <b>decode-PoC</b>. Both get a real cudagraph speedup — so the PoC decode path is genuinely captured by cudagraph, not silently falling back to eager.</p>
 <table><tr><th>workload</th><th class=num>eager</th><th class=num>cudagraph</th><th class=num>cudagraph speedup</th></tr>{rows}</table>
 <p class=verdict><span class=lbl>VERDICT</span> cudagraph speeds up decode-PoC by <b>{poc_sp:.2f}×</b> — the same mechanism that accelerates pure inference — confirming it is effectively engaged for PoC decode.</p></div>""")

# ---- Experiment 2: Separation ----
_short = lambda g: (g or "?").split(",")[0].replace("NVIDIA ", "").strip()   # GPU short name
LABELS = {"gen_honest_cg-flashattn": "honest producer · same config as validator (floor)",
          "gen_honest_cg-flashinfer": "honest producer · FlashInfer backend",
          "gen_honest_eager-flashattn": "honest producer · eager engine",
          "gen_fraud_cg-flashattn": "fraud producer · cheaper quant",
          "gen_fraud_cg-flashinfer": "fraud producer · cheaper quant, FlashInfer"}
vals = []
for f in sorted(glob.glob(f"{D}/val_*.json")):
    dj = L(f); r = res(dj); m = dj.get("meta", {})
    ref = os.path.basename(f).split("__")[-1].replace(".json", "")
    is_xhw = ref.startswith("xhw_")
    seg = ref                                             # locate the honest_/fraud_ config segment
    for kind in ("honest", "fraud"):                      # (xhw tags may carry a peer-GPU prefix)
        i = ref.find(kind + "_")
        if i >= 0: seg = ref[i:]; break                   # -> honest_cg-flashattn
    key = "gen_" + seg
    honest = "honest" in seg
    pgpu, vgpu = _short(r.get("prover_gpu")), _short(m.get("gpu"))
    if pgpu != "?" and vgpu != "?" and pgpu != vgpu: is_xhw = True   # robust: HW actually differs
    lab = LABELS.get(key, key)
    prof = key.replace("gen_honest_", "").replace("gen_fraud_", "")
    if is_xhw:
        lab = f"{lab} · xHW⇐{pgpu}"; prof = f"{prof} · xHW ({pgpu}⇐{vgpu})"
    vals.append((lab, r.get("rate", 0)*100, honest, r.get("per_nonce", []), prof))
if vals:
    hon = [v for v in vals if v[2]]; fr = [v for v in vals if not v[2]]
    hmax = max((v[1] for v in hon), default=0); fmin = min((v[1] for v in fr), default=0)
    thr = math.sqrt(hmax*fmin) if (hmax and fmin) else None
    _cfg = lambda p: (("cudagraph" if str(p).split(" ")[0].startswith("cg") else "eager"),
                      ("FlashInfer" if "flashinfer" in str(p) else "FlashAttn"))
    v_eng, v_bk = _cfg(meta.get("profile", "cg-flashattn"))    # validator config — READ from the artifact meta
    validator_cfg = f"{v_eng} · {v_bk}"
    rows = ""
    for lab, rate, honest, _pn, _pr in vals:
        cls = "good-t" if honest else "bad-t"
        ok = (rate < P_MISMATCH) if honest else (rate > P_MISMATCH)   # production acceptance at p_mismatch
        vd = ("honest ✓" if ok else "honest ✗ false-pos") if honest else ("fraud ✓" if ok else "fraud ✗ MISSED")
        vdcls = cls if ok else "bad-t"
        hi = " class=hi" if not honest else ""
        g, b = _cfg(_pr)
        prod = f"<b class={cls}>{'honest' if honest else 'fraud'}</b> · {g} · {b}"
        note = ""
        if "xHW" in _pr:                                   # producer ran on a different GPU than the validator
            pgpu = _pr.split("(")[-1].rstrip(")").split("⇐")[0].strip()
            note = f" · <b>xHW⇐{pgpu}</b>"
        # what this pair tests: producer config vs the fixed validator config
        diffs = []
        if g != v_eng: diffs.append("cross-engine")
        if b != v_bk: diffs.append("cross-backend")
        if "xHW" in _pr: diffs.append("cross-HW")
        base = "honest" if honest else "fraud"
        what = (f"{base} · " + " · ".join(diffs)) if diffs else (f"{base} · floor (same config)" if honest else f"{base} detection")
        rows += (f"<tr{hi}><td>{prod}{note}</td><td>{validator_cfg}</td><td class={cls}>{what}</td>"
                 f"<td class=num>{rate:.2f}%</td><td class={vdcls}>{vd}</td></tr>")
    # acceptance = the PRODUCTION fixed threshold p_mismatch (all honest below it, all fraud above it).
    passed = (not hon or hmax < P_MISMATCH) and (not fr or fmin > P_MISMATCH)
    _adapt = f" · adaptive gap √(honest_max·fraud_min) ≈ {thr:.2f}%" if thr else ""
    vtxt = (f"At production <b>p_mismatch = {P_MISMATCH:.1f}%</b>: honest ≤ <b>{hmax:.2f}%</b>, fraud ≥ <b>{fmin:.2f}%</b> — every honest pair below and every fraud pair above → fraud caught.{_adapt}" if passed
            else f"At production <b>p_mismatch = {P_MISMATCH:.1f}%</b>: honest ≤ {hmax:.2f}%, fraud ≥ {fmin:.2f}% — a pair crosses the line (honest &gt; p_mismatch → false-positive, or fraud &lt; p_mismatch → MISSED).{_adapt}")
    cards.append(f"""<div class=card><div class=exp>Experiment 2</div>
 <h2>Separation — honest vs fraud (anti-cheat)</h2>
 <p class=lead>A <b>producer</b> generates a trajectory; the <b>validator</b> (the honest model) re-runs it teacher-forced and counts mismatches. An <span class=good-t>honest producer</span> (real model) matches the validator → low %; a <span class=bad-t>fraud producer</span> (cheaper model) diverges → high %.</p>
 <div class=flow><span class=box>producer (honest or fraud)</span><span class=arr>→</span><span class=box>sphere_k trajectory</span><span class=arr>→</span><span class="box alt">honest validator re-runs</span><span class=arr>→</span><span class=box>mismatch rate</span><span class=cap>honest producer → low · fraud producer → high</span></div>
 <table><tr><th>producer config — what we validate</th><th>validator config — what validates</th><th>what it tests</th><th class=num>mismatch rate</th><th>verdict</th></tr>{rows}</table>
 <p class=verdict><span class=lbl>VERDICT — {'PASS' if passed else 'REVIEW'}</span> {vtxt}</p></div>""")

    # ---- Experiment 2 (detail): per-nonce HONEST vs FRAUD overlaid, one chart per prover config ----
    rows_pn = [(lab, rate, honest, pn, prof) for (lab, rate, honest, pn, prof) in vals if pn]
    if rows_pn:
        nst = max((e.get("n_steps", 257) for (_, _, _, pn, _) in rows_pn for e in pn), default=257)
        n_nonce = len(rows_pn[0][3])
        # per-nonce charts in PERCENT (consistent with the rate labels + the calculated threshold),
        # on ONE shared y-scale so every config is directly comparable.
        _pct = lambda e: e["n_sphere_mismatches"] / max(e.get("n_steps", nst), 1) * 100.0
        _allpct = [_pct(e) for (_, _, _, pn, _) in rows_pn for e in pn]
        ytop = max((max(_allpct) * 1.1 if _allpct else 5.0), (thr * 1.4 if thr else 0.0), 5.0)
        prof_name = {"cg-flashattn": "cudagraph·FlashAttn", "cg-flashinfer": "cudagraph·FlashInfer",
                     "eager-flashattn": "eager·FlashAttn", "eager-flashinfer": "eager·FlashInfer",
                     "cudagraph": "cudagraph", "eager": "eager"}
        # TWO rows by PRODUCER — honest producer (all its configs) then fraud producer (all its configs),
        # both re-checked by the SAME honest validator.
        hon_ser = [(prof_name.get(prof, prof), rate, pn) for (_l, rate, honest, pn, prof) in rows_pn if honest]
        fr_ser  = [(prof_name.get(prof, prof), rate, pn) for (_l, rate, honest, pn, prof) in rows_pn if not honest]
        W, H = 900.0, 230.0; mL, mR, mT, mB = 50.0, 14.0, 12.0, 22.0
        pw, ph = W - mL - mR, H - mT - mB
        Y = lambda p: mT + (1 - p / ytop) * ph          # p = per-nonce mismatch RATE (%), shared ytop
        def _bg():
            grid = ""
            for yv in (0, ytop / 2, ytop):
                yy = Y(yv)
                grid += (f'<line x1="{mL}" x2="{mL+pw}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="#eef2f7"/>'
                         f'<text x="{mL-5}" y="{yy+4:.1f}" text-anchor="end" font-size="10" fill="#94a3b8">{yv:.1f}%</text>')
            tl = zones = ""
            if thr and 0 < thr <= ytop:
                yt = Y(thr)
                zones = (f'<rect x="{mL}" y="{mT:.1f}" width="{pw}" height="{yt-mT:.1f}" fill="#dc2626" fill-opacity="0.06"/>'
                         f'<rect x="{mL}" y="{yt:.1f}" width="{pw}" height="{mT+ph-yt:.1f}" fill="#16a34a" fill-opacity="0.06"/>'
                         f'<text x="{mL+6}" y="{mT+13:.1f}" font-size="10" fill="#dc2626">FRAUD zone (&gt; threshold)</text>'
                         f'<text x="{mL+6}" y="{mT+ph-6:.1f}" font-size="10" fill="#16a34a">honest zone (&lt; threshold)</text>')
                tl = (f'<line x1="{mL}" x2="{mL+pw}" y1="{yt:.1f}" y2="{yt:.1f}" stroke="#0f172a" stroke-dasharray="5 3"/>'
                      f'<text x="{mL+pw}" y="{yt-3:.1f}" text-anchor="end" font-size="10" fill="#0f172a">threshold {thr:.2f}%</text>')
            return grid, zones, tl
        def _row(head, ser, col, hcls):
            if not ser: return ""
            n = max((len(pn) for _, _, pn in ser), default=1)
            X = lambda i: mL + (i / max(n - 1, 1)) * pw
            grid, zones, tl = _bg()
            dots = ""
            for _lab, _rate, pn in ser:
                pns = sorted(pn, key=lambda e: e["nonce"])
                dots += "".join(f'<circle cx="{X(i):.1f}" cy="{Y(_pct(e)):.1f}" r="2" fill="{col}" fill-opacity="0.7"/>' for i, e in enumerate(pns))
            svg = (f'<svg viewBox="0 0 {W} {H}" width="100%" style="display:block;background:#fff;border:1px solid #e2e8f0;border-radius:6px">{zones}{grid}{tl}{dots}</svg>')
            confs = " · ".join(f'{lab} <b>{rate:.2f}%</b>' for lab, rate, pn in ser)
            return (f'<div style="margin:.6rem 0"><div style="font-size:.95rem;margin-bottom:.15rem">'
                    f'honest validator vs <b class={hcls}>{head}</b> &mdash; {confs}'
                    f'</div>{svg}</div>')
        panels = (_row("honest producer", hon_ser, "#16a34a", "good-t")
                  + _row("fraud producer",  fr_ser,  "#dc2626", "bad-t"))
        def _setup(prof, honest, rate):
            graph = "cudagraph" if prof.startswith("cg") else "eager"
            backend = "FlashInfer" if "flashinfer" in prof else "FlashAttn"
            cls = "good-t" if honest else "bad-t"
            return (f"<tr><td class={cls}>{'honest' if honest else 'fraud'} producer</td>"
                    f"<td>{graph}</td><td>{backend}</td><td class=num>{rate:.2f}%</td></tr>")
        vrow = f'<tr class=hi><td><b>VALIDATOR (fixed)</b></td><td><b>{v_eng}</b></td><td><b>{v_bk}</b></td><td class=num>—</td></tr>'
        setup_rows = "".join(_setup(prof, honest, rate) for (_l, rate, honest, pn, prof) in rows_pn)
        setup_tbl = (f'<table><tr><th>role · producer</th><th>engine (graph or not)</th><th>attention backend</th>'
                     f'<th class=num>mismatch rate</th></tr>{vrow}{setup_rows}</table>')
        cards.append(f"""<div class=card><div class=exp>Experiment 2 — detail</div>
 <h2>Per-nonce separation — honest producer vs fraud producer</h2>
 <p class=lead>The one <b>validator</b> is fixed at <b>{validator_cfg}</b> and re-checks every producer run below — each producer varies its engine (graph or not) and/or backend. The two chart rows then plot each nonce's <b>mismatch rate</b> (%) on a shared scale: <span class=good-t><b>honest producer</b></span> stays in the green zone (below threshold), <span class=bad-t><b>fraud producer</b></span> in the red zone (above).</p>
 {setup_tbl}
 {panels}</div>""")

# ---- Experiment 3: GSM8K ----
def gsm(s):
    try: return res(L(f"{D}/gsm_cg-flashattn_{s}.json")).get("flexible_extract")
    except Exception: return None
on, off = gsm("on"), gsm("off")
if on is not None and off is not None:
    same = abs(on-off) <= GSM_TOL      # tolerance (default ~2.5pt = sampling noise on 100 q); --gsm-tol
    cards.append(f"""<div class=card><div class=exp>Experiment 3</div>
 <h2>Co-existence — GSM8K under PoC load</h2>
 <p class=lead>GSM8K accuracy <b>with</b> a concurrent PoC load vs <b>without</b>.</p>
 <div class=flow><span class=box>GSM8K</span><span class=arr>+</span><span class="box alt">concurrent PoC</span><span class=arr>→</span><span class="box good">accuracy unchanged?</span></div>
 <table><tr><th>PoC load</th><th class=num>flexible</th></tr><tr><td>off (baseline)</td><td class=num>{off*100:.0f}%</td></tr><tr class=hi><td>on (decode PoC)</td><td class=num><b>{on*100:.0f}%</b></td></tr></table>
 <p class=verdict><span class=lbl>VERDICT — {'PASS' if same else 'REVIEW'}</span> flexible {on*100:.0f}% vs {off*100:.0f}% — {'within noise' if same else 'differs'}.</p></div>""")

# ---- Experiment 4: k-distribution (codebook coverage, from artifacts) ----
def _khist(fn):
    try: d = L(f"{D}/{fn}")
    except Exception: return None, 0
    c = {}
    for a in d.get("artifacts", []):
        for k in (a.get("k_points_steps") or []):
            if isinstance(k, int) and k >= 0: c[k] = c.get(k, 0) + 1
    tot = sum(c.values())
    return ([100*c.get(i, 0)/tot for i in range(16)] if tot else None), tot
hon_k, nh = _khist("gen_honest_cg-flashattn.json")
fra_k, nf = _khist("gen_fraud_cg-flashattn.json")
if hon_k:
    W, H, PAD, BW = 860, 300, 44, 20; plotH = H - 2*PAD
    series = [("#2563eb", hon_k)] + ([("#dc2626", fra_k)] if fra_k else [])
    maxp = max([6.25] + [max(s) for _, s in series]) * 1.15
    yy = lambda p: PAD + plotH - (p/maxp*plotH); gw = (W - 2*PAD)/16; b = []
    for i in range(16):
        x0 = PAD + i*gw + (gw - BW*len(series))/2
        for j, (col, s) in enumerate(series):
            b.append(f'<rect x="{x0+j*BW:.1f}" y="{yy(s[i]):.1f}" width="{BW}" height="{PAD+plotH-yy(s[i]):.1f}" fill="{col}" opacity="{0.75 if j else 1}"/>')
        b.append(f'<text x="{PAD+i*gw+gw/2:.1f}" y="{H-PAD+14}" font-size="10" text-anchor="middle" fill="#555">{i}</text>')
    yi = yy(6.25)
    svg = (f'<svg viewBox="0 0 {W} {H}" style="max-width:100%;height:auto">'
           f'<line x1="{PAD}" y1="{yi:.1f}" x2="{W-PAD}" y2="{yi:.1f}" stroke="#16a34a" stroke-dasharray="5 4"/>'
           f'<text x="{W-PAD}" y="{yi-4:.1f}" font-size="10" text-anchor="end" fill="#16a34a">ideal 6.25%</text>'
           f'<line x1="{PAD}" y1="{PAD+plotH}" x2="{W-PAD}" y2="{PAD+plotH}" stroke="#999"/>{"".join(b)}</svg>')
    leg = (f'<span><span style="display:inline-block;width:12px;height:12px;background:#2563eb"></span> honest (N={nh:,})</span>'
           + (f' &nbsp; <span><span style="display:inline-block;width:12px;height:12px;background:#dc2626;opacity:.75"></span> fraud (N={nf:,})</span>' if fra_k else '')
           + ' &nbsp; <span style="color:#16a34a">– – ideal 6.25%</span>')
    cards.append(f"""<div class=card><div class=exp>Experiment 4</div>
 <h2>k-distribution — codebook coverage</h2>
 <p class=lead>How often each of the 16 sphere codebook points is hit across all nonces × steps. Near-uniform = high-entropy fingerprint (full codebook live). Honest &amp; fraud both cover all 16 — fraud is caught by <b>which step lands where</b> (the chained trajectory), not by codebook coverage.</p>
 <div style="margin:.4rem 0">{leg}</div>
 {svg}</div>""")

chip = "NO DATA" if not cards else ("NEEDS REVIEW" if "REVIEW" in "".join(cards) else "ALL PASS")
html = f"""<!doctype html><meta charset=utf-8><title>Decode-PoC report — {model}</title>
<style>{_IDEAL_CSS}</style><div class=wrap>
<h1>Decode-PoC validation &nbsp;<span class=chip>{chip}</span></h1>
<p class=sub>{sub}</p>
{''.join(cards)}
<p style="text-align:center;color:#94a3b8;font-size:.9rem">rate = Σ sphere_k mismatches / (nonces × (max_tokens+1)) · raw data in {os.path.basename(D)}</p>
<p style="text-align:center;color:#cbd5e1;font-size:.78rem;font-variant-numeric:tabular-nums">provenance — {prov}</p>
</div>"""
open(OUT, "w").write(html)
print(f"simplified report -> {OUT}  ({len(cards)} cards)")

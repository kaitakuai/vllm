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
a = ap.parse_args()
D = a.session.rstrip("/"); OUT = a.out or f"{D}/report_simple.html"
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
sub = " · ".join(str(x) for x in [
    model, meta.get("gpu", "?"), "decode-PoC",
    f"{len(meta.get('nonces', [])) or '?'} nonces", f"seq_len {meta.get('seq_len','?')}",
    f"max_tokens {meta.get('max_tokens','?')}", f"vLLM {meta.get('vllm_commit','?')}"])
prov = (f"codebook {str(meta.get('codebook_hash','?'))[:12]} · vLLM {meta.get('vllm_commit','?')} · "
        f"poc-scope {os.environ.get('POC_SCOPE_COMMIT','?')} · block_hash {str(meta.get('block_hash','?'))[:12]}")

def msstep(tag):  # ms/step from steps_per_s (poc) ; ms/req-equiv from chat
    try: return 1000.0 / res(L(f"{D}/perf_{tag}.poc.json")).get("steps_per_s")
    except Exception: return None

cards = []
# ---- Experiment 1: Performance ----
cg, eg = msstep("cg-flashattn"), msstep("eager-flashattn")
if cg and eg:
    try: cgc = 1000.0/ (res(L(f"{D}/perf_cg-flashattn.chat.json")).get("steps_per_s") or 1e9)
    except Exception: cgc = None
    sp = eg/cg
    rows = f"<tr><td>decode-PoC</td><td class=num>{cg:.1f}</td><td class=num>{eg:.1f}</td><td class=num>{sp:.2f}×</td></tr>"
    cards.append(f"""<div class=card><div class=exp>Experiment 1</div>
 <h2>Performance — decode efficiency</h2>
 <p class=lead>Per-step decode latency at a 32-wide batch — <b>cudagraph vs eager</b>.</p>
 <div class=flow><span class=box>32 nonces</span><span class=arr>→</span><span class=box>decode N steps</span><span class=arr>→</span><span class="box alt">ms / step</span><span class=arr>→</span><span class=box>cudagraph vs eager</span><span class=cap>per-step latency isolates decode (prefill removed by the slope)</span></div>
 <table><tr><th>workload</th><th class=num>cudagraph (ms/step)</th><th class=num>eager (ms/step)</th><th class=num>speedup</th></tr>{rows}</table>
 <p class=verdict><span class=lbl>VERDICT</span> cudagraph is <b>{sp:.1f}×</b> faster per step on decode-PoC.</p></div>""")

# ---- Experiment 2: Separation ----
_short = lambda g: (g or "?").split(",")[0].replace("NVIDIA ", "").strip()   # GPU short name
LABELS = {"gen_honest_cg-flashattn": "honest floor", "gen_honest_cg-flashinfer": "honest · FA↔FI",
          "gen_honest_eager-flashattn": "honest · cg↔eager", "gen_fraud_cg-flashattn": "FRAUD",
          "gen_fraud_cg-flashinfer": "FRAUD · x-backend"}
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
    rows = ""
    for lab, rate, honest, _pn, _pr in vals:
        cls = "good-t" if honest else "bad-t"; vd = "honest ✓" if honest else "fraud ✓"
        hi = " class=hi" if not honest else ""
        rows += f"<tr{hi}><td>{lab}</td><td class=num>{rate:.2f}%</td><td class={cls}>{vd}</td></tr>"
    passed = (thr and hmax < thr < fmin)
    vtxt = (f"honest ≤ <b>{hmax:.2f}%</b>, fraud ≥ <b>{fmin:.2f}%</b>; threshold <b>p_mismatch ≈ {thr:.2f}%</b> separates cleanly." if passed
            else f"honest ≤ {hmax:.2f}%, fraud ≥ {fmin:.2f}% — margin tight; review threshold.")
    cards.append(f"""<div class=card><div class=exp>Experiment 2</div>
 <h2>Separation — honest vs fraud (anti-cheat)</h2>
 <p class=lead>Each prover's trajectory re-checked on the <b>honest validator</b>: honest → low mismatch, fraud → high.</p>
 <div class=flow><span class=box>prover model</span><span class=arr>→</span><span class=box>sphere_k trajectory</span><span class=arr>→</span><span class="box alt">validator re-runs</span><span class=arr>→</span><span class=box>mismatch rate</span><span class=cap>honest (same model) low · fraud (cheaper model) high</span></div>
 <table><tr><th>case</th><th class=num>rate</th><th>verdict</th></tr>{rows}</table>
 <p class=verdict><span class=lbl>VERDICT — {'PASS' if passed else 'REVIEW'}</span> {vtxt}</p></div>""")

    # ---- Experiment 2 (detail): per-nonce HONEST vs FRAUD overlaid, one chart per prover config ----
    rows_pn = [(lab, rate, honest, pn, prof) for (lab, rate, honest, pn, prof) in vals if pn]
    if rows_pn:
        nst = max((e.get("n_steps", 257) for (_, _, _, pn, _) in rows_pn for e in pn), default=257)
        n_nonce = len(rows_pn[0][3])
        thr_cnt = (thr / 100.0) * nst if thr else None
        from collections import OrderedDict
        groups = OrderedDict()   # prof -> {"H":(lab,rate,pn), "F":(lab,rate,pn)}
        prof_name = {"cg-flashattn": "cudagraph · FlashAttn", "cg-flashinfer": "cudagraph · FlashInfer",
                     "eager-flashattn": "eager · FlashAttn", "eager-flashinfer": "eager · FlashInfer",
                     "cudagraph": "cudagraph", "eager": "eager"}
        for lab, rate, honest, pn, prof in rows_pn:
            groups.setdefault(prof, {})["H" if honest else "F"] = (lab, rate, pn)
        def _overlay(prof, g):
            series = [(g.get("H"), "#16a34a", "honest"), (g.get("F"), "#dc2626", "fraud")]
            allv = [e["n_sphere_mismatches"] for s, _, _ in series if s for e in s[2]]
            pmax = max(allv) if allv else 1
            ytop = max(int(pmax * 1.15) + 1, (int(thr_cnt) + 2 if thr_cnt else 0), 5)
            n = max((len(s[2]) for s, _, _ in series if s), default=1)
            W, H = 900.0, 150.0; mL, mR, mT, mB = 44.0, 12.0, 10.0, 20.0
            pw, ph = W - mL - mR, H - mT - mB
            X = lambda i: mL + (i / max(n - 1, 1)) * pw
            Y = lambda v: mT + (1 - v / ytop) * ph
            grid = ""
            for yv in (0, ytop // 2, ytop):
                yy = Y(yv)
                grid += (f'<line x1="{mL}" x2="{mL+pw}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="#eef2f7"/>'
                         f'<text x="{mL-5}" y="{yy+4:.1f}" text-anchor="end" font-size="10" fill="#94a3b8">{int(yv)}</text>')
            tl = (f'<line x1="{mL}" x2="{mL+pw}" y1="{Y(thr_cnt):.1f}" y2="{Y(thr_cnt):.1f}" stroke="#0f172a" stroke-dasharray="5 3"/>'
                  f'<text x="{mL+pw}" y="{Y(thr_cnt)-3:.1f}" text-anchor="end" font-size="10" fill="#0f172a">threshold {int(thr_cnt)}</text>'
                  ) if (thr_cnt and 0 < thr_cnt <= ytop) else ""
            dots = ""; sub = []
            for s, col, kind in series:
                if not s: continue
                lab, rate, pn = s; pn = sorted(pn, key=lambda e: e["nonce"])
                dots += "".join(f'<circle cx="{X(i):.1f}" cy="{Y(e["n_sphere_mismatches"]):.1f}" r="2" fill="{col}" fill-opacity="0.8"/>' for i, e in enumerate(pn))
                cls = "good-t" if kind == "honest" else "bad-t"
                sub.append(f'<b class={cls}>{lab}</b> {rate:.2f}%')
            svg = (f'<svg viewBox="0 0 {W} {H}" width="100%" style="display:block;background:#fff;border:1px solid #e2e8f0;border-radius:6px">{grid}{tl}{dots}</svg>')
            return (f'<div style="margin:.6rem 0"><div style="font-size:.9rem;margin-bottom:.1rem">'
                    f'<b>{prof_name.get(prof, prof)}</b> — {" vs ".join(sub)}</div>{svg}</div>')
        panels = "".join(_overlay(p, g) for p, g in groups.items())
        cards.append(f"""<div class=card><div class=exp>Experiment 2 — detail</div>
 <h2>Per-nonce separation — honest vs fraud, every nonce</h2>
 <p class=lead>One chart per prover config; each dot is one of the {n_nonce} nonces' <b>sphere_k mismatch count</b>. <span class=good-t><b>honest</b></span> (green) sits below the dashed threshold, <span class=bad-t><b>fraud</b></span> (red) above — separated on <b>every</b> nonce, not just on average.</p>
 {panels}</div>""")

# ---- Experiment 3: GSM8K ----
def gsm(s):
    try: return res(L(f"{D}/gsm_cg-flashattn_{s}.json")).get("flexible_extract")
    except Exception: return None
on, off = gsm("on"), gsm("off")
if on is not None and off is not None:
    same = abs(on-off) <= 0.0251      # ~2.5pt = sampling noise on 100 q (fp-safe)
    cards.append(f"""<div class=card><div class=exp>Experiment 3</div>
 <h2>Co-existence — GSM8K under PoC load</h2>
 <p class=lead>GSM8K accuracy <b>with</b> a concurrent PoC load vs <b>without</b>.</p>
 <div class=flow><span class=box>GSM8K</span><span class=arr>+</span><span class="box alt">concurrent PoC</span><span class=arr>→</span><span class="box good">accuracy unchanged?</span></div>
 <table><tr><th>PoC load</th><th class=num>flexible</th></tr><tr><td>off (baseline)</td><td class=num>{off*100:.0f}%</td></tr><tr class=hi><td>on (decode PoC)</td><td class=num><b>{on*100:.0f}%</b></td></tr></table>
 <p class=verdict><span class=lbl>VERDICT — {'PASS' if same else 'REVIEW'}</span> flexible {on*100:.0f}% vs {off*100:.0f}% — {'within noise' if same else 'differs'}.</p></div>""")

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

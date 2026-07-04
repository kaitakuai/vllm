#!/usr/bin/env python3
"""CUSTOM, NON-GIT post-processor. Inject an 'Artifacts & Reproduce (S3)' section into a
decode-PoC report.html, listing the public Supabase URLs of every artifact in the session
folder + the keyless pull-and-open command. Keeps the in-git report.py a pure renderer.

  inject_s3.py <report.html> <session-name> [public-base-url]
"""
import os, sys, html

report = sys.argv[1]
name = sys.argv[2]
PUB = sys.argv[3] if len(sys.argv) > 3 else \
    "https://wsegqlqqkkuzrlppdbcx.supabase.co/storage/v1/object/public/gonka-artifacts"
base = f"{PUB}/reports/{name}"
d = os.path.dirname(os.path.abspath(report))

files = sorted(
    os.path.relpath(os.path.join(r, f), d)
    for r, _, fs in os.walk(d) for f in fs
    if not f.startswith(".") and f != os.path.basename(report)
)
rows = "\n".join(
    f'<li><a href="{base}/{html.escape(f)}">{html.escape(f)}</a></li>' for f in files
)
section = f"""
<section style="margin-top:2em;padding:1em;border:2px solid #2a6;border-radius:8px;font-size:1.05em">
<h2 style="margin-top:0">📦 Artifacts &amp; Reproduce (S3)</h2>
<p>All inputs/outputs for this report are stored in the public <code>gonka-artifacts</code>
bucket (read-only, no key needed) —
<a href="{base}/report.html">open report ↗</a> ·
<a href="{base}/REPRODUCE.md">reproduce ↗</a></p>
<details><summary>{len(files)} artifact file(s)</summary><ul>{rows}</ul></details>
</section>
"""

with open(report, encoding="utf-8") as f:
    doc = f.read()
doc = doc.replace("</body>", section + "</body>") if "</body>" in doc else doc + section
with open(report, "w", encoding="utf-8") as f:
    f.write(doc)
print(f"injected S3 section ({len(files)} files) -> {report}")

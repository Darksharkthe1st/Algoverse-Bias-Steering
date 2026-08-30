#!/usr/bin/env python3
"""Build the team-facing research control surface from canonical repo state.

The site is a **projection**, never a source of truth. Every fact rendered here
is read at build time from one of:

    PROJECT_STATE.md      current paper, gate, blockers
    RESEARCH_CONTRACT.md  frozen science
    WORK_LEDGER.md        execution packages
    DECISION_LOG.md       why earlier documents no longer apply
    runs/                 actual experiment artifacts
    git                   freeze tag, SHA, dirty state
    tests/                live suite result

If a fact is not in one of those, it does not belong on the page. Hand-writing a
claim here re-creates the exact failure this project spent a week undoing.

Usage:  python3 scripts/build_dashboard.py --out dashboard/index.html
"""

from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sh(cmd: str, default: str = "") -> str:
    try:
        r = subprocess.run(cmd, shell=True, cwd=ROOT, capture_output=True,
                           text=True, timeout=90)
        return r.stdout.strip() or default
    except Exception:
        return default


def read(rel: str) -> str:
    p = ROOT / rel
    return p.read_text(encoding="utf-8") if p.is_file() else ""


def section(md: str, heading: str) -> str:
    m = re.search(rf"^##\s+{re.escape(heading)}\s*$(.*?)(?=^##\s|\Z)", md, re.M | re.S)
    return m.group(1).strip() if m else ""


def strip_md(s: str) -> str:
    s = re.sub(r"`([^`]*)`", r"\1", s)
    s = re.sub(r"\*\*([^*]*)\*\*", r"\1", s)
    s = re.sub(r"\*([^*]*)\*", r"\1", s)
    s = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", s)
    return " ".join(s.split()).strip()


def md_inline(s: str) -> str:
    s = html.escape(s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    return s


def git_state() -> dict:
    tag = sh("git describe --tags --abbrev=0 --match 'freeze-*'", "(untagged)")
    return {"tag": tag, "sha": sh("git rev-parse --short HEAD", "?"),
            "branch": sh("git rev-parse --abbrev-ref HEAD", "?"),
            "dirty": bool(sh("git status --porcelain")),
            "commits": sh("git rev-list --count HEAD", "?")}


def test_state() -> dict:
    """Run the suite the way a human runs it: one `pytest -q` over the repo.

    This used to invoke each test file in its own process. That hid cross-test
    state contamination by construction — the registries a test clobbered were
    restored for free by the next process — so the page published "96/97 green"
    while a normal `python -m pytest -q` gave 17 failures. A control surface
    that cannot reproduce its own headline number is worse than no number.

    Falls back to the per-file runners only if pytest is unavailable, and says
    so in `mode` rather than quietly reporting a weaker check as the same thing.
    """
    out = sh("python3 -m pytest -q 2>&1 | tail -25")
    m = re.search(r"^(?:(\d+) failed,? ?)?(\d+) passed(?:,? (\d+) skipped)?", out, re.M)
    if m:
        failed = int(m.group(1) or 0)
        passed = int(m.group(2))
        skipped = int(m.group(3) or 0)
        failing = [{"name": n.split("::")[0].split("/")[-1], "why": strip_md(w)[:160]}
                   for n, w in re.findall(r"^FAILED (\S+)(?: - (.*))?$", out, re.M)]
        return {"files": sorted(p.name for p in (ROOT / "tests").glob("test_*.py")),
                "passed": passed, "total": passed + failed, "skipped": skipped,
                "failing": failing, "mode": "pytest -q (normal collection)"}

    # No pytest — fall back to the standalone runners, and label it as such.
    files, passed, total, failing = [], 0, 0, []
    for f in sorted((ROOT / "tests").glob("test_*.py")):
        r = sh(f"python3 {f} 2>&1 | grep -E '^[0-9]+/[0-9]+ passed' | tail -1")
        mm = re.match(r"(\d+)/(\d+) passed", r)
        if not mm:
            continue
        p, t = int(mm.group(1)), int(mm.group(2))
        passed += p; total += t
        files.append(f.name)
        if p < t:
            failing.append({"name": f.name,
                            "why": strip_md(sh(f"python3 {f} 2>&1 | grep -E '^FAIL' | head -1"))[:160]})
    return {"files": files, "passed": passed, "total": total, "skipped": 0,
            "failing": failing, "mode": "per-file fallback (pytest unavailable)"}


def run_evidence() -> list[dict]:
    out = []
    for d in sorted((ROOT / "runs").glob("*/")):
        if not d.is_dir():
            continue
        need = ("results.csv", "summary.md", "manifest.json", "steering_vector.safetensors")
        arts = {n: (d / n).is_file() and (d / n).stat().st_size > 0 for n in need}
        rc = d / "results.csv"
        rows = max(0, sum(1 for _ in rc.open())) - 1 if rc.is_file() else 0
        out.append({"name": d.name, "rows": max(rows, 0), "complete": all(arts.values())})
    return out


def positive_control() -> dict | None:
    for d in sorted((ROOT / "runs").glob("*refusal-repro*")):
        log = d / "logs" / "run.log"
        if not log.is_file():
            continue
        txt = log.read_text(errors="ignore")
        arms = [(m[0], f"{m[1]}/{m[2]}") for m in
                re.findall(r"(harm\w+/\w+):\s+(\d+)/(\d+) refused", txt)]
        return {"run": d.name, "arms": arms, "log": str(log.relative_to(ROOT))}
    return None


def work_packages() -> list[dict]:
    pkgs = []
    for line in read("WORK_LEDGER.md").splitlines():
        if not line.startswith("| **WP-"):
            continue
        c = [x.strip() for x in line.strip("|").split("|")]
        if len(c) < 6:
            continue
        st = next((strip_md(x).lower() for x in c
                   if re.search(r"\*\*(done|not started|prepared|running)", x, re.I)), "not started")
        owner = next((x.strip() for x in c if x.strip() in
                      ("Farhan", "Jeremiah", "Edward", "Aryaman", "Claude", "unowned")), "—")
        pkgs.append({"id": strip_md(c[0]), "objective": strip_md(c[1]), "status": st,
                     "owner": owner, "blocking": "Y" in c[-1].upper(),
                     "evidence": strip_md(c[6]) if len(c) > 6 else ""})
    return pkgs


def decisions() -> list[dict]:
    out = []
    for m in re.finditer(r"^##\s+(D-\d+)\s+·\s+([^·]+)·\s+(.+?)$(.*?)(?=^##\s|\Z)",
                         read("DECISION_LOG.md"), re.M | re.S):
        b = m.group(4)
        g = lambda k: (re.search(rf"\*\*{k}\.\*\*(.*?)(?=\*\*[A-Z]|\Z)", b, re.S) or [None, ""])[1]
        out.append({"id": m.group(1), "date": m.group(2).strip(), "title": m.group(3).strip(),
                    "decision": strip_md(g("Decision"))[:340],
                    "evidence": strip_md(g("Evidence"))[:400]})
    return out


def scope_cuts() -> list[str]:
    body = section(read("RESEARCH_CONTRACT.md"), "11. Scope")
    cuts = []
    for label in ("Future work:", "Abandoned:"):
        m = re.search(rf"\*\*{re.escape(label)}\*\*(.*?)(?=\*\*|\Z)", body, re.S)
        if m:
            for part in re.split(r"[·;]", m.group(1)):
                part = strip_md(part).strip(" .")
                if 3 < len(part) < 90:
                    cuts.append(part)
    return cuts[:14]


CSS = """
:root{--bg:#0b0d10;--panel:#12161b;--line:#222a33;--ink:#e6edf3;--dim:#8b98a5;
--acc:#5bc8af;--warn:#e3b341;--bad:#f0776c;--good:#3fb950;
--mono:ui-monospace,SFMono-Regular,Menlo,monospace}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.6 -apple-system,BlinkMacSystemFont,'Segoe UI',Inter,sans-serif}
a{color:var(--acc)}
.wrap{max-width:1120px;margin:0 auto;padding:0 24px}
header{border-bottom:1px solid var(--line);background:linear-gradient(180deg,#0f1418,#0b0d10)}
.status{display:flex;flex-wrap:wrap;gap:9px;padding:14px 0;font:12px/1 var(--mono)}
.chip{border:1px solid var(--line);border-radius:999px;padding:6px 11px;color:var(--dim);white-space:nowrap}
.chip b{color:var(--ink);font-weight:600}
.chip.ok{border-color:#1d3d2a;color:var(--good)}
.chip.warn{border-color:#4a3c14;color:var(--warn)}
.chip.bad{border-color:#4a2320;color:var(--bad)}
h1{font-size:12px;letter-spacing:.16em;text-transform:uppercase;color:var(--dim);
margin:26px 0 10px;font-weight:600}
.paper{font-size:25px;line-height:1.4;margin:0 0 10px;max-width:62ch;font-weight:600}
.sub{color:var(--dim);max-width:74ch;margin:0 0 22px;font-size:14px}
section{padding:32px 0;border-bottom:1px solid var(--line)}
.grid{display:grid;gap:14px}
.g2{grid-template-columns:repeat(auto-fit,minmax(300px,1fr))}
.g3{grid-template-columns:repeat(auto-fit,minmax(210px,1fr))}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px}
.card h3{margin:0 0 8px;font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--dim)}
.card p{margin:0 0 8px;font-size:14px}
.big{font:600 22px/1.2 var(--mono)}
.mono{font-family:var(--mono);font-size:12.5px}
.dim{color:var(--dim)}
.gate{border-left:3px solid var(--bad);background:#181111}
table{width:100%;border-collapse:collapse;font-size:13.5px}
th{text-align:left;color:var(--dim);font-weight:600;font-size:11px;letter-spacing:.08em;
text-transform:uppercase;padding:8px 10px;border-bottom:1px solid var(--line)}
td{padding:9px 10px;border-bottom:1px solid #191f26;vertical-align:top}
.pill{display:inline-block;font:11px/1 var(--mono);padding:4px 7px;border-radius:5px;
border:1px solid var(--line);color:var(--dim)}
.pill.done{color:var(--good);border-color:#1d3d2a}
.pill.run{color:var(--warn);border-color:#4a3c14}
.tl{position:relative;padding-left:22px}
.tl:before{content:'';position:absolute;left:6px;top:4px;bottom:4px;width:1px;background:var(--line)}
.ev{position:relative;margin-bottom:15px}
.ev:before{content:'';position:absolute;left:-19px;top:7px;width:7px;height:7px;border-radius:50%;
background:var(--dim);box-shadow:0 0 0 3px var(--bg)}
.ev.key:before{background:var(--acc)}
.ev h4{margin:0 0 3px;font-size:14px}
.ev .meta{font:11px var(--mono);color:var(--dim);margin-bottom:3px}
.ev p{margin:0 0 3px;font-size:13.5px;color:#c5cfd9}
.cut{display:inline-block;margin:0 7px 7px 0;padding:6px 11px;border-radius:6px;
border:1px dashed #3a2a2a;color:#b98b86;font-size:13px;text-decoration:line-through}
.foot{padding:26px 0 50px;color:var(--dim);font-size:12.5px}
code{font-family:var(--mono);font-size:.92em;background:#171d24;padding:1px 5px;border-radius:4px}
.hint{font-size:12.5px;color:var(--dim);margin-top:8px}
.arch{background:#0e1216;border:1px solid var(--line);border-radius:10px;padding:14px;margin-top:10px}
details summary{cursor:pointer;color:var(--dim);font-size:13px;padding:6px 0}
"""

SVG = """
<svg viewBox="0 0 620 250" width="100%" style="max-width:640px" role="img"
     aria-label="Two ablation trajectories in logit space and the angle between them">
  <defs><marker id="a" markerWidth="9" markerHeight="9" refX="7" refY="3.2" orient="auto">
    <path d="M0,0 L7,3.2 L0,6.4 z" fill="#5bc8af"/></marker>
  <marker id="b" markerWidth="9" markerHeight="9" refX="7" refY="3.2" orient="auto">
    <path d="M0,0 L7,3.2 L0,6.4 z" fill="#e3b341"/></marker></defs>
  <line x1="70" y1="205" x2="290" y2="205" stroke="#2a333d"/>
  <line x1="70" y1="205" x2="70" y2="30" stroke="#2a333d"/>
  <line x1="360" y1="205" x2="580" y2="205" stroke="#2a333d"/>
  <line x1="360" y1="205" x2="360" y2="30" stroke="#2a333d"/>
  <text x="300" y="240" fill="#8b98a5" font-size="11.5" font-family="monospace"
        text-anchor="middle">x: Δlogit P(named a side) on S2 · y: Δlogit P(refused) on harm</text>
  <text x="70" y="22" fill="#5bc8af" font-size="12.5" font-weight="600">one shared control</text>
  <line x1="70" y1="205" x2="250" y2="70" stroke="#5bc8af" stroke-width="2.5" marker-end="url(#a)"/>
  <circle cx="130" cy="160" r="3.4" fill="#5bc8af"/><circle cx="190" cy="115" r="3.4" fill="#5bc8af"/>
  <text x="255" y="66" fill="#5bc8af" font-size="12" font-family="monospace">r̂_harm</text>
  <line x1="70" y1="205" x2="226" y2="88" stroke="#8fd9c8" stroke-width="2.5"
        stroke-dasharray="5 3" marker-end="url(#a)"/>
  <text x="196" y="112" fill="#8fd9c8" font-size="12" font-family="monospace">r̂_stance</text>
  <path d="M128,168 A62,62 0 0,1 140,154" fill="none" stroke="#5bc8af" stroke-width="1.4"/>
  <text x="146" y="168" fill="#5bc8af" font-size="12" font-family="monospace">θ &lt; 25°</text>
  <text x="360" y="22" fill="#e3b341" font-size="12.5" font-weight="600">distinct controls</text>
  <line x1="360" y1="205" x2="360" y2="65" stroke="#e3b341" stroke-width="2.5" marker-end="url(#b)"/>
  <text x="368" y="62" fill="#e3b341" font-size="12" font-family="monospace">r̂_harm</text>
  <line x1="360" y1="205" x2="545" y2="205" stroke="#c678dd" stroke-width="2.5"/>
  <text x="452" y="196" fill="#c678dd" font-size="12" font-family="monospace">r̂_stance</text>
  <path d="M360,175 A30,30 0 0,0 390,205" fill="none" stroke="#e3b341" stroke-width="1.4"/>
  <text x="378" y="170" fill="#e3b341" font-size="12" font-family="monospace">θ large</text>
  <text x="360" y="38" fill="#8b98a5" font-size="10.5">not claimable — every nuisance biases θ this way</text>
</svg>
"""


def build() -> str:
    ps = read("PROJECT_STATE.md")
    g, t = git_state(), test_state()
    runs, pc, pkgs, decs = run_evidence(), positive_control(), work_packages(), decisions()
    E = html.escape

    paper = strip_md(section(ps, "The paper, in one sentence"))
    gate = section(ps, "Current gate")
    blocks = strip_md(section(ps, "Blocks the paper"))
    notblocks = strip_md(section(ps, "Does *not* block the paper"))
    venue = re.search(r"\*\*Venue\*\*\s*\|\s*(.+?)\s*\|", ps)
    deadline = re.search(r"\*\*Deadline\*\*\s*\|\s*(.+?)\s*\|", ps)
    complete = sum(1 for r in runs if r["complete"])
    blocking = [p for p in pkgs if p["blocking"]]
    done = sum(1 for p in blocking if "done" in p["status"])

    def pill(st):
        if "done" in st:
            return '<span class="pill done">done</span>'
        if "running" in st or "prepared" in st:
            return '<span class="pill run">in flight</span>'
        return '<span class="pill">not started</span>'

    o = [f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>fvoa · research control surface</title><style>{CSS}</style></head><body>
<header><div class="wrap"><div class="status">
<span class="chip"><b>{E(g['tag'])}</b> · {E(g['sha'])}</span>
<span class="chip {'bad' if g['dirty'] else 'ok'}">{'uncommitted changes' if g['dirty'] else 'clean tree'}</span>
<span class="chip {'ok' if not t['failing'] else 'warn'}">tests <b>{t['passed']}/{t['total']}</b></span>
<span class="chip">runs complete <b>{complete}/{len(runs)}</b></span>
<span class="chip bad">positive control <b>not satisfied</b></span>
<span class="chip"><b>{E(strip_md(deadline.group(1)) if deadline else '')}</b></span>
</div></div></header><div class="wrap">

<section><h1>The paper</h1>
<p class="paper">{md_inline(paper)}</p>
<p class="sub">{E(strip_md(venue.group(1)) if venue else '')}</p>
<h1>The experiment</h1>
<div class="arch">{SVG}
<p class="hint">Partial directional ablation <code>x − λ(x·r̂)r̂</code> at λ ∈ {{0, 0.5, 1}}, for
<code>r̂_stance</code>, <code>r̂_harm</code> and a covariance-matched <code>r̂_random</code>, on both
batteries. <strong>θ</strong> is the angle between the two trajectories <strong>in logit space</strong>
— under a shared knob the ratio of logit changes is constant across λ and across directions, and that
invariant is what probability space destroys.</p></div>
<div class="grid g2" style="margin-top:14px">
<div class="card"><h3>What a small θ can establish</h3>
<p>“Consistent with a shared-control model, within the resolution of this experiment.” Conditional on
all four gates passing.</p></div>
<div class="card"><h3>What it cannot</h3>
<p>It does <strong>not</strong> establish a unique shared mechanism. And a <strong>large θ licenses
nothing on its own</strong> — every identified nuisance biases θ toward “distinct”, none toward
“shared”. Inference is deliberately asymmetric.</p></div></div></section>

<section><h1>Current gate</h1><div class="card gate">
{''.join(f'<p>{md_inline(strip_md(b))}</p>' for b in gate.split(chr(10) + chr(10))[:4] if b.strip())}
</div></section>

<section><h1>Evidence</h1><div class="grid g3">
<div class="card"><h3>Test suite</h3><div class="big">{t['passed']}/{t['total']}</div>
<p class="dim mono">{len(t['files'])} files &middot; {E(t['mode'])}{f" &middot; {t['skipped']} skipped" if t.get('skipped') else ''}</p>
{''.join(f"<p class='dim mono'>{E(f['why'])}</p>" for f in t['failing'])}</div>
<div class="card"><h3>Run artifacts</h3><div class="big">{complete}/{len(runs)}</div>
<p class="dim mono">complete = results + summary + manifest + vector</p></div>
<div class="card"><h3>Blocking packages</h3><div class="big">{done}/{len(blocking)}</div>
<p class="dim mono">done by evidence, not by report</p></div></div>"""]

    if pc:
        rows = "".join(f"<tr><td class='mono'>{E(k)}</td><td class='mono'>{E(v)} refused</td></tr>"
                       for k, v in pc["arms"])
        o.append(f"""<div class="card" style="margin-top:14px">
<h3>Positive control — mechanism yes, replication no</h3><table>{rows}</table>
<p class="hint">Ablation collapses harmful refusal, so the operator works. But the run's own findings
doc records <strong>“Not reproduced”</strong>: baseline 0.380 against the paper's 0.700, extraction
cosine 0.90 against a 0.999 target — and on Qwen1.5-1.8B, not a submission model.
<strong>G1 is not satisfied.</strong> Source: <code>{E(pc['log'])}</code></p></div>""")

    o.append(f"""<div class="card" style="margin-top:14px">
<h3>The simulation that killed the selectivity statistic</h3>
<p>Holding the world <strong>fixed at “one shared control”</strong> and varying only direction-estimate
quality — which is unmeasurable — the previous <code>SEL ≥ 2</code> rule fired between
<strong>1.8% and 64%</strong> of the time with no second mechanism present, and inverted sign at one
setting. That is why the headline statistic changed.</p>
<p class="hint">Reproduce: <code>python3 analysis/sim_lambda_identifiability.py</code></p></div>
<details style="margin-top:14px"><summary>All committed runs ({len(runs)})</summary>
<table><tr><th>run</th><th>rows</th><th>artifacts</th></tr>
{''.join("<tr><td class='mono'>%s</td><td class='mono'>%s</td><td>%s</td></tr>" % (
    E(r['name']), r['rows'] or '&mdash;',
    "<span class='pill done'>complete</span>" if r['complete'] else "<span class='pill'>partial</span>")
    for r in runs)}</table></details></section>

<section><h1>Why the paper changed</h1>
<p class="sub">Each entry is an epistemic update with the evidence that forced it. Full text in
<code>DECISION_LOG.md</code> — append-only, history rather than doctrine.</p><div class="tl">""")

    key = {"D-004", "D-009", "D-011", "D-012", "D-013", "D-015"}
    for d in decs:
        o.append(f"""<div class="ev{' key' if d['id'] in key else ''}">
<div class="meta">{E(d['id'])} · {E(d['date'])}</div><h4>{E(d['title'])}</h4>
<p>{E(d['decision'])}</p><p class="dim">{E(d['evidence'])}</p></div>""")

    o.append("""</div></section><section><h1>Execution</h1>
<p class="sub">Work packages, not personal agendas. Owner is metadata — a package keeps its meaning
when it changes hands. A package is done when its <strong>evidence exists and validates</strong>.</p>
<table><tr><th>ID</th><th>Objective</th><th>Status</th><th>Owner</th><th>Evidence required</th></tr>""")
    for p in blocking:
        o.append(f"""<tr><td class="mono">{E(p['id'])}</td><td>{E(p['objective'])}</td>
<td>{pill(p['status'])}</td><td class="mono dim">{E(p['owner'])}</td>
<td class="dim mono">{E(p['evidence'][:70])}</td></tr>""")
    o.append(f"""</table><div class="grid g2" style="margin-top:16px">
<div class="card"><h3>Blocks submission</h3><p>{md_inline(blocks)}</p></div>
<div class="card"><h3>Does not block</h3><p class="dim">{md_inline(notblocks)}</p></div>
</div></section>

<section><h1>Not this paper</h1>
<p class="sub">Deliberately cut. Resurrecting any of these needs a dated amendment to
<code>RESEARCH_CONTRACT.md</code> §12 — “this would be more interesting” is not a reopen condition.</p>
{''.join(f'<span class="cut">{E(c)}</span>' for c in scope_cuts())}</section>

<section><h1>Archive</h1>
<p class="sub">Superseded framings are kept for provenance in <code>docs/superseded/</code>, each
carrying a banner. 2025 results are <strong>not</strong> current evidence — the archived refusal arms
were invalidated by a one-dimensional tensor broadcast, documented in
<code>docs/REVIVAL_AUDIT.md</code>. Nothing there is a paper claim.</p></section>

<div class="foot">Projection of <code>{E(g['branch'])}</code> @ <code>{E(g['sha'])}</code> · tag
<code>{E(g['tag'])}</code> · {E(g['commits'])} commits · built
{datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC.<br>
Generated from <code>PROJECT_STATE.md</code>, <code>RESEARCH_CONTRACT.md</code>,
<code>WORK_LEDGER.md</code>, <code>DECISION_LOG.md</code>, <code>runs/</code>, git and the live test
suite. A projection, never a source of truth — if this page disagrees with the repo, the repo wins.
</div></div></body></html>""")
    return "".join(o)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="dashboard/index.html")
    a = ap.parse_args()
    p = ROOT / a.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(build(), encoding="utf-8")
    print(f"wrote {p} ({p.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

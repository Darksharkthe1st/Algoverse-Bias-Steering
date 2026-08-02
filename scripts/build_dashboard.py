"""Generate a self-contained HTML status dashboard for the bias-steering revival.

Built so a newcomer or returning member can drop in cold and get oriented in
under a minute: what we're building, why it was revived, where we are, the
headline (and honest-negative) results, decisions, the task board, and what's
next. Pure stdlib; renders the results tables directly from committed CSV
artifacts under experiments/past_logs/. Narrative constants at the top are
hand-maintained — edit them via PR (see .claude/skills/dashboard-update/).

    python3 -m scripts.build_dashboard --out dashboard/index.html
"""
import os
import csv
import html
import argparse
import datetime

# Committed result artifacts (do not point these at uncommitted files)
BATCHED_CSV = "experiments/past_logs/methodology_experiments/batched_tests/Batched_Gen.csv"
CROWS_CSV = "experiments/past_logs/crows_experiments/Crows_Opin_Tests/Crows_Opin.csv"

# ---- editable narrative state -------------------------------------------------
TAGLINE = "Finding — and steering — the 'soft refusal' direction: whether an open-weight LLM takes a side at all, as distinct from refusing harm and from which side it takes."

OVERVIEW = [
    ("What we're building",
     "An activation-space account of <b>soft refusal</b> — the behavior where a model declines to take sides on controversial-but-not-harmful prompts. We extract steering vectors/subspaces with TransformerLens (difference-in-means and 2026-grade upgrades), steer open-weight models toward opinionation or neutrality, and test how soft refusal relates to the Arditi <i>hard</i>-refusal direction."),
    ("Why it was revived",
     "The 2025 Algoverse run got a real result — <b>bidirectional in-distribution steering with coherence preserved</b> — but stalled on OOD transfer and ran out of team. Revived <b>Aug 2026</b> by a three-person team. The frontier moved (refusal is now cones/subspaces, not single directions; labs ship persona vectors and even-handedness evals), yet the exact soft-refusal claim is <b>still unclaimed</b>."),
    ("Where we are today",
     "<b>REVIVAL SPRINT — week 0.</b> Post-mortem + frontier scan + fused sprint plan are in <code>docs/</code>. Thesis recommendation: <b>\"One Knob or Two?\"</b> — the soft-refusal vs harm-refusal dissociation, pending team sign-off Tue. Venue <b>verified</b>: Interpretability for Discovery @ NeurIPS 2026, deadline <b>Aug 29 AoE</b> (5pp, non-archival, double-blind). Week 1 = judge v2 + unified re-extraction; Gate 1 Sunday."),
]

RESEARCH_Q = ("The research question",
    "Is the <b>soft-refusal direction</b> — controlling <i>whether</i> the model takes a side on contested-but-benign prompts — causally and geometrically dissociable from the Arditi <b>hard-refusal</b> direction, or are they one shared knob? A 2×2 cross-steering grid (steer each; measure opinion benchmarks AND safety benchmarks) answers it either way: dissociation contradicts the 'one shared refusal knob' finding (QCRI, arXiv:2602.02132); entanglement mechanistically explains why refusal-direction surgery shifts unrelated dispositions (arXiv:2607.17427). Full plan: docs/2026-08-01_sprint_plan.md.")

PHASE = "REVIVAL SPRINT week 0→1 — sign off thesis Tue; judge-v2 rubric + unified Arditi-convention re-extraction; Gate 1 (kappa ≥ 0.7 + archived stance-shift ≥ 10pp) end of week 1. Target: Interp4Discovery @ NeurIPS 2026, Aug 29 AoE."

RUNS = [  # (label, state) state in {running, done, queued}
    ("2025 · In-distribution bidirectional steering, 9 models (Batched_Gen)", "done"),
    ("2025 · Zero-vector ablation control (effect is the vector, not the hook)", "done"),
    ("2025 · Per-model coefficient sweeps (0–20)", "done"),
    ("2025 · BBQ transfer — semi-failed", "done"),
    ("2025 · CrowS-Pairs transfer — FAILED (honest negative, kept)", "done"),
    ("2025 · Refusal-vector ↔ opinion-vector cross-application — failed both ways", "done"),
    ("2025 · Synthetic-neutral-output steering — failed (Dec 2025, last commit)", "done"),
    ("2026 · Judge v2 (two-axis: stance-taking × hedging; kappa ≥ 0.7 gate) + re-judge archived outputs", "queued"),
    ("2026 · Unified Arditi-convention re-extraction, both direction families, 4 models", "queued"),
    ("2026 · Gate 2: reproduce bidirectional steering + Arditi bypass with re-extracted vectors", "queued"),
    ("2026 · 2×2 cross-steering grid: 4 models × 5 conditions × 2 batteries (~30k gens)", "queued"),
]

DECISIONS = [
    ("Core method (baseline)", "Difference-in-means steering at all layers/positions",
     "The 2025 recipe. Now the BASELINE, not the method — upgrades (affine/ACE, capping, conditional) are sprint work, per the frontier scan."),
    ("Judge", "UNDER REPLACEMENT — GPT-4o-mini binary judge is retired",
     "2025 rubric scored any clear stance 'opinionated', even factual ones — conflating decisiveness with bias. This predicts the CrowS transfer failure. Judge v2 = graded scale + separated axes; the dormant farhan-opinion-spectrum branch has a 5-point seed."),
    ("Models", "Qwen1.5-7B · gemma-2b-it · Llama-3-8B · Qwen2.5-7B (sprint set)",
     "Three families + one modern checkpoint answers the model-vintage objection; Qwen1.5-14B and Yi dropped (budget, redundancy). All fit one A100 in bf16."),
    ("Evals", "Opinion: held-out comparisons + IssueBench subset + Paired Prompts · Safety: XSTest + JailbreakBench · Capability: MMLU slice",
     "The 2×2 grid needs BOTH batteries per condition; BBQ/CrowS-Pairs retired to historical context (CrowS widely criticized)."),
    ("Venue", "Interp4Discovery @ NeurIPS 2026 — Aug 29 AoE (VERIFIED)",
     "5pp + refs, non-archival, double-blind; backup AI4GOOD (same deadline); slip path ICLR 2027. docs/2026-08-01_venue_scan.md."),
    ("Scope", "Open-weight models only; no cone-fitting; no ideology-direction experiment this cycle",
     "Steering needs residual-stream access. Scope cuts are doctrine — see sprint plan §6."),
]

TASKS = [  # who, track, status (active|blocked|queued|done), next action
    ("Farhan", "Research lead · pipeline", "active",
     "Sign off (or red-line) PAPER_FRAMING.md + sprint plan Tue. Pipeline walkthrough for the team. Week-1 technical task: unified re-extraction — port opinionation extraction to Arditi conventions (post-instruction positions), extract harm-refusal directions, all 4 models. Recover Slack graphs if the archive lands."),
    ("Edward", "Measurement · evals · geometry · infra", "active",
     "Team kit + analysis + sprint plan shipped (this PR). Week 1: two-axis judge rubric → kappa ≥ 0.7 against ~150 gold labels (annotation with Jeremiah); re-judge archived logs (~$20-50 API); first per-layer cosine figure by Friday. Week 2: benchmark harness (agent-assisted) — IssueBench subset, Paired Prompts, XSTest, JailbreakBench. Week 3: judging + aggregation as the grid streams."),
    ("Jeremiah", "Onboarding → audits + writing", "active",
     "Onboarding path in HANDOFF_JEREMIAH.md (Arditi paper, analysis doc, 3B1B, TransformerLens demo). Week 1: help hand-label the ~150 gold examples — the fastest way to learn the construct. Week 2: assist the eval harness + reproduce one archived experiment. Weeks 3-4: MMLU audit runs, repro pass, distribution figures, writing."),
]

HOW_WE_WORK = [
    ("Cadence", "Tue / Thu / Sat <b>9pm ET</b> syncs · Slack/Discord between meetings"),
    ("Compute", "Lambda Cloud (~$377 credits, Farhan) · possibly a small GPU cluster (Edward) · archived 2025 outputs cover judge/eval work with <b>zero GPU</b>"),
    ("Code", "This repo only — PRs, no direct pushes to main; runs land as raw CSVs + pickles under <code>experiments/</code>"),
    ("Agents", "Coding agents are first-class: they read <code>CLAUDE.md</code>/<code>AGENTS.md</code> and <code>PAPER_FRAMING.md</code>; doctrine disagreements are PRs to those files, not forked narratives"),
]

MILESTONES = [
    ("done", "2025 core result: bidirectional in-distribution steering, 9 models, ablation-controlled"),
    ("done", "Post-mortem + frontier scan + fused sprint plan + venue verified (docs/)"),
    ("done", "Team kit: dashboard · framing doctrine · agent instructions · handoffs"),
    ("now", "Week 1 → Gate 1: thesis sign-off (Tue) · judge v2 (kappa ≥ 0.7) · re-judge archived outputs · unified re-extraction"),
    ("next", "Week 2 → Gate 2: reproduce steering + Arditi with re-extracted vectors · geometry package · eval harness"),
    ("then", "Week 3: the 2×2 grid (~30k gens) · Week 4: robustness + 5pp draft → SUBMIT Aug 29 AoE"),
]

PATH_TO_SUBMISSION = [
    ("done", "Edward", "Post-mortem, frontier scan, team kit, sprint plan, venue verification committed."),
    ("now", "Team", "Tue: sign off thesis + sprint plan (PAPER_FRAMING.md, docs/2026-08-01_sprint_plan.md)."),
    ("now", "Edward", "Judge v2 rubric → kappa ≥ 0.7 vs ~150 gold labels (Jeremiah co-annotates) → re-judge archived outputs. GATE 1(a)+(b) end of week 1."),
    ("now", "Farhan", "Unified Arditi-convention re-extraction of both direction families, 4 models (~20 GPU-hrs)."),
    ("next", "Farhan", "Gate 2: reproduce bidirectional steering (≥20pp, ≥3/4 models) + Arditi bypass/induction (≥30pp) with re-extracted vectors."),
    ("next", "Edward", "Benchmark harness (agent-assisted): IssueBench subset · Paired Prompts · XSTest · JailbreakBench + geometry package (per-layer cosines, principal angles)."),
    ("then", "Farhan", "Week 3 crunch: 4×5×2 cross-steering grid (~110-130 GPU-hrs); degradation order pre-committed — never shrink the safety battery."),
    ("then", "Edward+Jeremiah", "Judging/aggregation + MMLU audit as results stream; per-example distribution figures."),
    ("then", "Team", "Freeze Aug 26 · red-team read Aug 27 · SUBMIT Interp4Discovery @ NeurIPS 2026, Aug 29 AoE."),
]

LINKS = [
    ("★ The Correct Problem (read first)", "https://github.com/Darksharkthe1st/Algoverse-Bias-Steering/blob/main/docs/THE_CORRECT_PROBLEM.md"),
    ("Framing doctrine (PAPER_FRAMING.md)", "https://github.com/Darksharkthe1st/Algoverse-Bias-Steering/blob/main/PAPER_FRAMING.md"),
    ("Sprint plan (thesis · gates · weeks)", "https://github.com/Darksharkthe1st/Algoverse-Bias-Steering/blob/main/docs/2026-08-01_sprint_plan.md"),
    ("Post-mortem + frontier scan", "https://github.com/Darksharkthe1st/Algoverse-Bias-Steering/blob/main/docs/2026-08-01_project_analysis.md"),
    ("Venue scan", "https://github.com/Darksharkthe1st/Algoverse-Bias-Steering/blob/main/docs/2026-08-01_venue_scan.md"),
    ("Interp4Discovery CFP (target venue)", "https://interpretability4discovery.github.io/cfp.html"),
    ("Repo", "https://github.com/Darksharkthe1st/Algoverse-Bias-Steering"),
    ("2025 paper outline (rough)", "https://www.overleaf.com/4514258212zmrztmsxptvy#2adb04"),
    ("Arditi refusal-direction paper", "https://arxiv.org/abs/2406.11717"),
    ("Anthropic even-handedness eval (open-source)", "https://github.com/anthropics/political-neutrality-eval"),
]

# Frontier reading list — mirrors PAPER_FRAMING.md (doctrine lives there; this
# renders it). action in {must-cite, nice, watch}. From the 2026-08-01 scan.
FRONTIER = [
    ("must-cite", "Refusal is mediated by a single direction — Arditi et al. (arXiv:2406.11717)",
     "The method we build on; our soft-refusal construct is defined against its hard-refusal one."),
    ("must-cite", "There Is More to Refusal than a Single Direction — QCRI (arXiv:2602.02132)",
     "Eleven refusal flavors collapse onto one behavioral knob. Our separability question is posed directly against this — we must show soft refusal is or isn't that knob."),
    ("must-cite", "The Geometry of Refusal: Concept Cones — Wollschläger et al., ICML 2025 (arXiv:2502.17420)",
     "Single-direction claims are out; cone/subspace language and their independence criteria are the standard we report against."),
    ("must-cite", "Refusal Steering — Multiverse Computing (arXiv:2512.16602)",
     "Closest competitor: DiM-family political-refusal control at 80B with safety preserved. One paragraph must differentiate: they steer censorship/refusal, we factorize opinionation."),
    ("must-cite", "AxBench — Wu et al., ICML 2025 (arXiv:2501.17148)",
     "Prompting is the mandatory baseline; naive DiM usually loses. Every steering result needs the system-prompt comparison."),
    ("must-cite", "Abliteration Is Not a Scalpel — Fafuła (arXiv:2607.17427)",
     "Refusal-direction removal shifts opinionation as a side effect; explicitly requests our dissociation experiment. Cite as motivation."),
    ("must-cite", "Steering Towards Fairness — Nadeem et al. (arXiv:2508.08846)",
     "Steers WHICH side (ideology axes). We steer WHETHER a side is taken; the factorization must be stated against this line."),
    ("nice", "Persona Vectors — Anthropic (arXiv:2507.21509) · Assistant Axis (arXiv:2601.10387)",
     "Industrialized trait-vector pipeline + deployed activation capping; capping > constant coefficients."),
    ("nice", "CAST conditional steering — IBM, ICLR 2025 (arXiv:2409.05907) · ACE affine editing (arXiv:2411.09003)",
     "The two cheap recipe upgrades over add-everywhere DiM."),
    ("nice", "IssueBench — Röttger et al., TACL 2026 (arXiv:2502.08395)",
     "The 2026-standard political-bias eval; reviewers will ask why BBQ/CrowS alone."),
    ("watch", "Manifold steering — Wurgaft et al. (arXiv:2605.05115)",
     "If the geometry track opens up: on-manifold steering beats linear; the license for geometry-aware interventions."),
]

BULLETPROOFING = [
    ("done", "2025 ablation control: zero-coefficient steering → 99% nonsense; the vector does the work."),
    ("done", "Transfer failure documented honestly (CrowS: several models show literally no effect)."),
    ("todo", "Judge v2: current construct conflates decisiveness with bias — every downstream claim inherits this until fixed."),
    ("todo", "Prompting baseline: no steering result is reportable without the system-prompt comparison (AxBench bar)."),
    ("todo", "Per-example distributions: aggregate judge percentages hide bimodal steering effects."),
    ("todo", "Side-effect audits: capability (MMLU slice) + safety (XSTest/JailbreakBench) on any final intervention."),
    ("done", "Deadline verified 2026-08-01: Interp4Discovery @ NeurIPS 2026, Aug 29 AoE (5pp, non-archival, double-blind); backup AI4GOOD same day."),
    ("todo", "Extraction-convention confound: 2025 vectors are mean-pooled over all tokens — headline figures must use the unified Arditi-convention re-extraction only."),
    ("idea", "Judge-v2 on archived outputs may RE-DATE the 2025 headline numbers — treat old percentages as provisional until re-judged."),
]
# ------------------------------------------------------------------------------


def _read_rows(path):
    try:
        with open(path) as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def _short(name):
    return name.split("/", 1)[-1]


def _b(v):
    cls = "b-pos" if v >= 10 else ("b-neg" if v <= -10 else "b-zero")
    return f"<span class='b {cls}'>{v:+d}</span>"


def steering_table(path, note, empty_note):
    rows = _read_rows(path)
    if not rows:
        return f"<div class='card'><p class='muted'>{empty_note}</p></div>"
    body = ""
    for r in rows:
        try:
            io, inu = int(r["Init->Opin"]), int(r["Init->Neut"])
            oo = int(r["Opin->Opin"])
            nn = int(r["Neut->Neut"])
            nons = int(r["Init->Nons"]) + int(r["Opin->Nons"]) + int(r["Neut->Nons"])
        except (KeyError, ValueError):
            continue
        d_op, d_nu = oo - io, nn - inu
        body += (f"<tr><td>{html.escape(_short(r['Model name']))}</td>"
                 f"<td class='mono'>{io} / {inu}</td>"
                 f"<td class='mono'>{oo} {_b(d_op)}</td>"
                 f"<td class='mono'>{nn} {_b(d_nu)}</td>"
                 f"<td class='mono'>{nons}</td></tr>")
    return f"""
<div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--r);overflow:hidden">
  <table class="data-table">
    <tr><th>Model</th><th>baseline Opin/Neut</th><th>steered → opinion (Δ)</th><th>steered → neutral (Δ)</th><th>nonsense</th></tr>
    {body}
  </table>
  <div style="padding:10px 16px 14px;font-size:12px;color:var(--ink3);border-top:1px solid var(--border)">{note}</div>
</div>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dashboard/index.html")
    ap.add_argument("--stamp", default=None)
    args = ap.parse_args()

    stamp = args.stamp or datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    def _gpill(state):
        cls = {"done": "p-done", "running": "p-running", "active": "p-active",
               "blocked": "p-blocked", "todo": "p-todo", "queued": "p-todo",
               "now": "p-running", "next": "p-next", "then": "p-then",
               "idea": "p-idea"}.get(state, "p-todo")
        return f"<span class='pill {cls}'>{state}</span>"

    # Deadline countdown (Interp4Discovery @ NeurIPS 2026, Aug 29 AoE)
    try:
        _today = datetime.date.fromisoformat(stamp[:10])
    except ValueError:
        _today = datetime.date.today()
    _deadline = datetime.date(2026, 8, 29)
    _days_left = (_deadline - _today).days
    if _days_left > 1:
        deadline_chip = f"<span class='chip chip-red' title='Interpretability for Discovery @ NeurIPS 2026 — verified on the CFP.'>⏳ {_days_left} days to submission — Aug 29 AoE</span>"
    elif _days_left >= 0:
        deadline_chip = "<span class='chip chip-red'>🔥 SUBMISSION WINDOW — Aug 29 AoE</span>"
    else:
        deadline_chip = "<span class='chip chip-gray'>Aug 29 passed — slip path: ICLR 2027 (Sept 25)</span>"

    # Which sprint week are we in? (sprint: Aug 3 → Aug 29, 2026)
    _wk_starts = [datetime.date(2026, 8, 3), datetime.date(2026, 8, 10),
                  datetime.date(2026, 8, 17), datetime.date(2026, 8, 24)]
    def _wk_state(i):
        if _today < _wk_starts[i]:
            return "next"
        if i == 3:
            return "now" if _today <= _deadline else "done"
        return "now" if _today < _wk_starts[i + 1] else "done"

    _weeks = [
        ("Week 1", "Aug 3–9", "Judge v2 rubric · ~150 gold labels · re-judge archive · unified re-extraction", "GATE 1: kappa ≥ 0.7 + stance-shift ≥ 10pp"),
        ("Week 2", "Aug 10–16", "Reproduce steering + Arditi · geometry package · eval harness", "GATE 2: ≥20pp steer, ≥30pp refusal"),
        ("Week 3", "Aug 17–23", "THE GRID — 4 models × 5 conditions × 2 batteries (~30k gens)", "degradation order pre-committed"),
        ("Week 4", "Aug 24–29", "Robustness · distribution figures · 5-page draft", "freeze 26 · red-team 27 · SUBMIT 29 AoE"),
    ]
    timeline_html = "".join(
        f"<div class='wk {_wk_state(i)}'>"
        f"<div class='wk-head'>{_gpill(_wk_state(i))}<span class='wk-name'>{w}</span><span class='wk-dates'>{d}</span></div>"
        f"<div class='wk-body'>{html.escape(t)}</div>"
        f"<div class='wk-gate'>{html.escape(g)}</div></div>"
        for i, (w, d, t, g) in enumerate(_weeks))

    experiment_html = """
<div class="x22-wrap">
  <div class="x22-title">The experiment at a glance — the 2×2 cross-steering grid</div>
  <table class="x22">
    <tr><th class="x22-corner">intervene ↓ &nbsp;·&nbsp; measure →</th>
        <th>Opinion battery<br><span class="x22-sub">IssueBench · Paired Prompts · comparisons</span></th>
        <th>Safety battery<br><span class="x22-sub">XSTest · JailbreakBench</span></th></tr>
    <tr><td class="x22-row">Steer <b>soft-refusal</b> direction ±</td>
        <td class="x22-diag">large Δ expected<br><span class="x22-sub">the direction works</span></td>
        <td class="x22-off">≈ 0 if two knobs<br><span class="x22-sub">moves → entangled</span></td></tr>
    <tr><td class="x22-row">Ablate <b>hard-refusal</b> direction</td>
        <td class="x22-off">≈ 0 if two knobs<br><span class="x22-sub">moves → explains Fafuła side effects</span></td>
        <td class="x22-diag">large Δ expected<br><span class="x22-sub">Arditi replication</span></td></tr>
  </table>
  <div class="x22-legend"><b>Two knobs</b> (diagonal only) contradicts the shared-refusal-knob finding (arXiv:2602.02132) · <b>One knob</b> (off-diagonals move) mechanistically explains abliteration side effects (arXiv:2607.17427) · <b>Either outcome is the paper.</b> Plus per-layer cosines/principal angles between the two directions, a system-prompt baseline, and an MMLU capability audit.</div>
</div>"""

    overview_html = "".join(
        f"<div class='ov-card'><h3>{t}</h3><p>{b}</p></div>" for t, b in OVERVIEW)
    runs_html = "".join(f"<li>{_gpill(s)} {html.escape(l)}</li>" for l, s in RUNS)
    dec_html = "".join(
        f"<tr><td>{k}</td><td>{v}</td><td class='dim'>{html.escape(w)}</td></tr>"
        for k, v, w in DECISIONS)
    tasks_html = "".join(
        f"<tr><td><div class='task-who'>{who}</div><div class='task-track'>{track}</div></td>"
        f"<td>{_gpill(st)}</td><td class='task-action'>{nx}</td></tr>"
        for who, track, st, nx in TASKS)
    how_html = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in HOW_WE_WORK)
    bullets_html = "".join(f"<li>{_gpill(s)} {html.escape(t)}</li>" for s, t in BULLETPROOFING)
    miles_html = "".join(
        f"<div class='mile {'done' if s == 'done' else ('now' if s in ('now', 'running') else '')}'>"
        f"{_gpill(s)}<div class='mile-text'>{html.escape(t)}</div></div>"
        for s, t in MILESTONES)
    path_html = "".join(
        f"<li>{_gpill(s)}<div class='ptext'><span class='pwho'>{who}:</span> {t}</div></li>"
        for s, who, t in PATH_TO_SUBMISSION)
    links_html = "".join(f"<a href='{u}' target='_blank'>{html.escape(n)} ↗</a>" for n, u in LINKS)
    frontier_html = "".join(
        f"<tr><td><span class='pill {'p-done' if a == 'must-cite' else ('p-running' if a == 'nice' else 'p-todo')}'>{a}</span></td>"
        f"<td><b>{html.escape(t)}</b></td><td class=small style='font-size:12.5px;color:var(--ink3)'>{html.escape(r)}</td></tr>"
        for a, t, r in FRONTIER)

    batched_html = steering_table(
        BATCHED_CSV,
        "Counts out of ~100 held-out synthetic comparison prompts per condition, GPT-4o-mini judge (2025 binary rubric — provisional until judge v2 re-judges). "
        "Steering works bidirectionally in-distribution with near-zero nonsense. Artifact: <code>" + BATCHED_CSV + "</code>",
        "Batched_Gen.csv not found — run from repo root.")
    crows_html = steering_table(
        CROWS_CSV,
        "Same vectors applied to CrowS-Pairs prompts: deltas near zero for most models — the 2025 'neutrality direction' does NOT transfer to real "
        "social-bias benchmarks. This honest negative is a core motivation for the revival. Artifact: <code>" + CROWS_CSV + "</code>",
        "Crows_Opin.csv not found — run from repo root.")

    head = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bias Steering — project dashboard</title>
<style>
:root{--bg:#0a0c10;--bg2:#0d1117;--surface:#111620;--surface2:#161c28;--surface3:#1c2436;--border:#1f2a3e;--border2:#2a374f;--ink:#e8edf5;--ink2:#a8b8cc;--ink3:#5e7490;--accent:#4a7cf5;--accent-dim:rgba(74,124,245,.10);--accent-glow:rgba(74,124,245,.04);--green:#2ea86a;--green-dim:rgba(46,168,106,.10);--red:#d95f5f;--red-dim:rgba(217,95,95,.10);--yellow:#c4972a;--yellow-dim:rgba(196,151,42,.10);--purple:#8b74d4;--purple-dim:rgba(139,116,212,.10);--r:8px;--font:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;--mono:'JetBrains Mono','SF Mono','Cascadia Code',Consolas,monospace}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth}
body{font-family:var(--font);font-size:14px;line-height:1.65;color:var(--ink2);background:var(--bg);-webkit-font-smoothing:antialiased}
::-webkit-scrollbar{width:6px}::-webkit-scrollbar-track{background:var(--bg)}::-webkit-scrollbar-thumb{background:var(--border2);border-radius:3px}
nav{position:sticky;top:0;z-index:100;height:52px;display:flex;align-items:center;gap:4px;padding:0 28px;background:rgba(10,12,16,.85);backdrop-filter:blur(20px) saturate(180%);border-bottom:1px solid var(--border)}
.nav-brand{font-weight:700;font-size:15px;color:var(--ink);letter-spacing:-.3px;margin-right:20px;white-space:nowrap;display:flex;align-items:center;gap:8px}
.nav-dot{width:7px;height:7px;border-radius:50%;background:var(--green);opacity:.85}
.nav-links{display:flex;gap:2px}
.nav-links a{color:var(--ink3);font-size:13px;text-decoration:none;padding:5px 10px;border-radius:6px;transition:color .15s,background .15s}
.nav-links a:hover{color:var(--ink);background:var(--surface2)}
.nav-right{margin-left:auto;display:flex;align-items:center;gap:12px}
.nav-meta{font-size:12px;color:var(--ink3)}
.nav-gh{font-size:12.5px;color:var(--ink3);text-decoration:none;border:1px solid var(--border2);padding:4px 12px;border-radius:6px;transition:all .15s;font-weight:500}
.nav-gh:hover{color:var(--ink);border-color:var(--accent)}
.hero{position:relative;overflow:hidden;padding:44px 36px 36px;border-bottom:1px solid var(--border);background:var(--bg2)}
.hero-inner{max-width:1120px;margin:0 auto}
.hero-eyebrow{font-size:11px;font-weight:600;letter-spacing:1.4px;text-transform:uppercase;color:var(--ink3);margin-bottom:12px}
.hero h1{font-size:26px;font-weight:700;color:var(--ink);letter-spacing:-.3px;line-height:1.25;margin-bottom:10px}
.hero-sub{font-size:15px;color:var(--ink2);max-width:640px;line-height:1.7;margin-bottom:24px}
.hero-chips{display:flex;flex-wrap:wrap;gap:8px}
[title]{cursor:help}
.chip{display:inline-flex;align-items:center;gap:5px;padding:5px 12px;border-radius:20px;font-size:12px;font-weight:600;letter-spacing:.2px;border:1px solid transparent}
.chip-green{background:var(--green-dim);color:var(--green);border-color:rgba(46,168,106,.25)}
.chip-blue{background:var(--accent-dim);color:var(--accent);border-color:rgba(74,124,245,.25)}
.chip-yellow{background:var(--yellow-dim);color:var(--yellow);border-color:rgba(196,151,42,.25)}
.chip-gray{background:var(--surface2);color:var(--ink3);border-color:var(--border2)}
.wrap{max-width:1120px;margin:0 auto;padding:0 28px 80px}
.g2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.g3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px}
.section-label{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:var(--ink3);display:flex;align-items:center;gap:12px;margin:44px 0 18px}
.section-label::after{content:'';flex:1;height:1px;background:var(--border)}
.card{background:var(--surface);border:1px solid var(--border);border-radius:var(--r);padding:20px 22px}
.card-title{font-size:13.5px;font-weight:600;color:var(--ink);margin-bottom:6px}
.ov-card{background:var(--surface);border:1px solid var(--border);border-top:2px solid var(--accent);border-radius:var(--r);padding:20px 22px}
.ov-card h3{font-size:13.5px;font-weight:600;color:var(--ink);margin-bottom:10px}
.ov-card p{font-size:13px;color:var(--ink2);line-height:1.65}
.callout{background:var(--accent-glow);border:1px solid rgba(74,124,245,.2);border-left:3px solid var(--accent);border-radius:var(--r);padding:18px 22px}
.callout h3{font-size:13px;font-weight:600;color:var(--accent);margin-bottom:8px}
.callout p{font-size:13.5px;color:var(--ink2);line-height:1.7}
.callout b{color:var(--ink)}
.insight{background:rgba(46,168,106,.05);border:1px solid rgba(46,168,106,.15);border-left:3px solid var(--green);border-radius:var(--r);padding:16px 20px;font-size:13px;color:var(--ink2);line-height:1.7}
.insight strong{color:var(--green)}
.warn{background:var(--red-dim);border:1px solid rgba(217,95,95,.2);border-left:3px solid var(--red);border-radius:8px;padding:12px 16px;font-size:12.5px;color:#f4a0a0;line-height:1.6;margin-top:14px}
.warn strong{color:var(--red)}
.b{display:inline-block;padding:2px 8px;border-radius:10px;font-size:12px;font-weight:600;font-family:var(--mono)}
.b-pos{background:var(--green-dim);color:var(--green)}
.b-neg{background:var(--red-dim);color:var(--red)}
.b-zero{background:var(--surface2);color:var(--ink3)}
.data-table{width:100%;border-collapse:collapse;font-size:13px}
.data-table th{text-align:left;padding:10px 16px;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.6px;color:var(--ink3);border-bottom:1px solid var(--border);background:var(--surface2)}
.data-table td{padding:11px 16px;color:var(--ink2);border-bottom:1px solid var(--border);vertical-align:middle}
.data-table tr:last-child td{border-bottom:none}
.data-table tr:hover td{background:rgba(255,255,255,.025)}
.data-table td.mono{font-family:var(--mono);font-size:13px}
.pill{display:inline-flex;align-items:center;gap:4px;padding:3px 9px;border-radius:20px;font-size:11px;font-weight:700;letter-spacing:.4px;text-transform:uppercase;white-space:nowrap;border:1px solid transparent}
.p-done{background:var(--green-dim);color:var(--green);border-color:rgba(46,168,106,.25)}
.p-running{background:var(--yellow-dim);color:var(--yellow);border-color:rgba(196,151,42,.25)}
.p-todo,.p-next,.p-then{background:var(--surface2);color:var(--ink3);border-color:var(--border2)}
.p-idea{background:var(--purple-dim);color:var(--purple);border-color:rgba(139,116,212,.25)}
.p-active{background:var(--accent-dim);color:var(--accent);border-color:rgba(74,124,245,.25)}
.p-blocked{background:var(--red-dim);color:var(--red);border-color:rgba(217,95,95,.25)}
.task-table{width:100%;border-collapse:collapse}
.task-table th{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.6px;color:var(--ink3);text-align:left;padding:10px 18px;border-bottom:1px solid var(--border);background:var(--surface2)}
.task-table td{padding:14px 18px;border-bottom:1px solid var(--border);vertical-align:top}
.task-table tr:last-child td{border-bottom:none}
.task-table tr:hover td{background:rgba(255,255,255,.018)}
.task-who{font-weight:600;color:var(--ink);white-space:nowrap;font-size:13.5px}
.task-track{font-size:12.5px;color:var(--ink3)}
.task-action{font-size:13px;color:var(--ink2);line-height:1.6}
.milestones{display:grid;grid-template-columns:1fr 1fr;gap:10px;align-items:start}
.mile{background:var(--surface2);border:1px solid var(--border);border-radius:8px;padding:12px 14px;display:flex;flex-direction:column;gap:6px}
.mile.done{border-color:rgba(46,168,106,.25)}
.mile.now{border-color:rgba(196,151,42,.25);background:rgba(196,151,42,.04)}
.mile-text{font-size:12.5px;color:var(--ink2);line-height:1.5}
.path{list-style:none;padding:0;margin:0}
.path li{display:flex;align-items:flex-start;gap:12px;padding:9px 0;border-bottom:1px solid var(--border)}
.path li:last-child{border-bottom:none}
.path .ptext{font-size:13.5px;color:var(--ink2);line-height:1.5}
.path .pwho{font-weight:600;color:var(--ink)}
.links{display:flex;flex-wrap:wrap;gap:10px;margin-top:4px}
.links a{display:inline-flex;align-items:center;gap:6px;font-size:13px;border:1px solid var(--border2);padding:8px 14px;border-radius:8px;color:var(--ink);background:var(--surface2)}
.links a:hover{border-color:var(--accent);text-decoration:none}
.status-list{list-style:none}
.status-list li{display:flex;align-items:flex-start;gap:10px;padding:6px 0;font-size:13.5px;color:var(--ink2)}
.status-list li:not(:last-child){border-bottom:1px solid var(--border)}
.kv-table{width:100%;border-collapse:collapse}
.kv-table td{padding:12px 0;border-bottom:1px solid var(--border);font-size:13.5px;vertical-align:top}
.kv-table tr:last-child td{border-bottom:none}
.kv-table td:first-child{color:var(--ink);font-weight:600;width:110px;white-space:nowrap;padding-right:24px}
code{font-family:var(--mono);font-size:12.5px;background:var(--surface3);padding:2px 6px;border-radius:4px;color:var(--accent)}
a{color:var(--accent);text-decoration:none}
a:hover{text-decoration:underline}
.dec-table{width:100%;border-collapse:collapse}
.dec-table th{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.6px;color:var(--ink3);text-align:left;padding:10px 16px;border-bottom:1px solid var(--border);background:var(--surface2)}
.dec-table td{padding:13px 16px;border-bottom:1px solid var(--border);font-size:13px;color:var(--ink2);vertical-align:middle}
.dec-table tr:last-child td{border-bottom:none}
.dec-table td:first-child{color:var(--ink);font-weight:600;white-space:nowrap}
.dec-table td.dim{color:var(--ink3);font-size:12.5px;line-height:1.55}
.muted{color:var(--ink3)}
.chip-red{background:var(--red-dim);color:#f4a0a0;border-color:rgba(217,95,95,.3);font-family:var(--mono)}
.timeline{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}
.wk{background:var(--surface);border:1px solid var(--border);border-radius:var(--r);padding:14px 16px;display:flex;flex-direction:column;gap:8px;position:relative}
.wk.now{border-color:rgba(196,151,42,.45);background:linear-gradient(180deg,rgba(196,151,42,.06),var(--surface))}
.wk.done{border-color:rgba(46,168,106,.3)}
.wk-head{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.wk-name{font-weight:700;color:var(--ink);font-size:13.5px}
.wk-dates{font-family:var(--mono);font-size:11.5px;color:var(--ink3);margin-left:auto}
.wk-body{font-size:12.5px;color:var(--ink2);line-height:1.55}
.wk-gate{font-family:var(--mono);font-size:11px;color:var(--yellow);border-top:1px dashed var(--border);padding-top:8px;letter-spacing:.2px}
.wk.done .wk-gate{color:var(--green)}
.x22-wrap{background:var(--surface);border:1px solid rgba(74,124,245,.35);border-radius:var(--r);overflow:hidden;box-shadow:0 0 0 1px rgba(74,124,245,.08),0 8px 32px rgba(74,124,245,.05);margin-top:16px}
.x22-title{padding:14px 20px;font-size:13.5px;font-weight:600;color:var(--ink);background:linear-gradient(135deg,rgba(74,124,245,.08),var(--surface2));border-bottom:1px solid var(--border)}
.x22{width:100%;border-collapse:collapse}
.x22 th{padding:12px 16px;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.5px;color:var(--ink2);background:var(--surface2);border-bottom:1px solid var(--border);text-align:center;line-height:1.5}
.x22 th.x22-corner{text-align:left;color:var(--ink3);text-transform:none;letter-spacing:0;font-family:var(--mono);font-weight:500;width:30%}
.x22 td{padding:16px;border-bottom:1px solid var(--border);border-right:1px solid var(--border);text-align:center;font-size:13px;line-height:1.5}
.x22 td:last-child{border-right:none}
.x22 tr:last-child td{border-bottom:none}
.x22 td.x22-row{text-align:left;color:var(--ink);font-size:13px;background:var(--surface2)}
.x22 td.x22-diag{color:var(--green);font-weight:600;background:rgba(46,168,106,.05)}
.x22 td.x22-off{color:var(--yellow);font-weight:600;background:rgba(196,151,42,.04)}
.x22-sub{display:block;font-size:11px;font-weight:400;color:var(--ink3);text-transform:none;letter-spacing:0;margin-top:2px}
.x22-legend{padding:12px 20px 14px;font-size:12px;color:var(--ink3);line-height:1.7;border-top:1px solid var(--border)}
.x22-legend b{color:var(--ink2)}
@media(max-width:960px){.g3{grid-template-columns:1fr 1fr}.timeline{grid-template-columns:1fr 1fr}}
@media(max-width:640px){.g2,.g3,.timeline{grid-template-columns:1fr}}
</style>
</head>"""

    htmldoc = head + f"""
<body>
<nav>
  <span class="nav-brand"><span class="nav-dot"></span>Bias Steering</span>
  <div class="nav-links">
    <a href="#overview">Overview</a>
    <a href="#timeline">Timeline</a>
    <a href="#results">Results</a>
    <a href="#decisions">Decisions</a>
    <a href="#tasks">Tasks</a>
    <a href="#path">Path</a>
    <a href="#frontier">Frontier</a>
  </div>
  <div class="nav-right">
    <span class="nav-meta">updated {stamp}</span>
    <a class="nav-gh" href="https://github.com/Darksharkthe1st/Algoverse-Bias-Steering" target="_blank">GitHub ↗</a>
  </div>
</nav>

<div class="hero">
  <div class="hero-inner">
    <div class="hero-eyebrow">Algoverse · Revival Sprint · Aug 2026</div>
    <h1>Soft Refusal — the geometry of taking a side</h1>
    <p class="hero-sub">{html.escape(TAGLINE)}</p>
    <div class="hero-chips">
      {deadline_chip}
      <span class="chip chip-green" title="Bidirectional steering on 9 models, ablation-controlled, coherence preserved (2025).">In-distribution steering verified</span>
      <span class="chip chip-yellow" title="Vectors trained on synthetic prompts do not transfer to CrowS-Pairs; semi-fail on BBQ. Kept honest.">Honest negative: no OOD transfer</span>
      <span class="chip chip-blue" title="No published paper names a soft-refusal direction as of Aug 2026; nearest neighbors differentiated in PAPER_FRAMING.md.">'Soft refusal' still unclaimed</span>
      <span class="chip chip-gray">Darksharkthe1st/Algoverse-Bias-Steering</span>
    </div>
  </div>
</div>

<div class="wrap">

<div class="section-label" id="overview">Project overview</div>
<div class="g3">{overview_html}</div>

<div class="callout" style="margin-top:16px">
  <h3>{RESEARCH_Q[0]}</h3>
  <p>{RESEARCH_Q[1]}</p>
</div>

{experiment_html}

<div class="section-label" id="timeline">Sprint timeline — Aug 3 → Aug 29</div>
<div class="timeline">{timeline_html}</div>

<div class="section-label" id="results">2025 results — what we inherit</div>
<p style="font-size:13px;color:var(--ink3);margin-bottom:16px">Both tables render live from committed CSVs. Δ = steered minus baseline count (≈percentage points).</p>
<div class="card-title" style="margin-bottom:8px">Headline: in-distribution bidirectional control (synthetic comparison prompts)</div>
{batched_html}
<div class="card-title" style="margin:20px 0 8px">Honest negative: the same vectors on CrowS-Pairs</div>
{crows_html}
<div class="warn"><strong>Provisional numbers.</strong> All 2025 percentages were produced by the retired binary judge, whose rubric scored factual decisiveness as opinionation. Judge v2 re-judging of the archived outputs may move every number above. Do not quote them in new text without the caveat.</div>

<div class="section-label" id="decisions">Decisions (locked + under review)</div>
<div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--r);overflow:hidden">
  <table class="dec-table">
    <tr><th>Item</th><th>Choice</th><th>Why</th></tr>
    {dec_html}
  </table>
</div>

<div class="section-label">Status &amp; roadmap</div>
<div class="g2">
  <div class="card">
    <div class="card-title" style="margin-bottom:14px">Experiment status</div>
    <ul class="status-list">{runs_html}</ul>
  </div>
  <div class="card">
    <div class="card-title" style="margin-bottom:14px">Milestones</div>
    <div class="milestones">{miles_html}</div>
  </div>
</div>

<div class="section-label" id="tasks">Task board</div>
<div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--r);overflow:hidden">
  <table class="task-table">
    <tr><th>Who</th><th>Status</th><th>Next action</th></tr>
    {tasks_html}
  </table>
</div>

<div class="section-label">How we work</div>
<div class="card">
  <table class="kv-table">{how_html}</table>
</div>

<div class="section-label" id="path">Path to submission</div>
<div class="card"><ul class="path">{path_html}</ul></div>

<div class="section-label">Docs &amp; links</div>
<div class="card"><div class="links">{links_html}</div></div>

<div class="section-label" id="frontier">Frontier — related papers &amp; positioning (mirrors PAPER_FRAMING.md)</div>
<div class="card"><table class="data-table"><tr><th>action</th><th>paper</th><th>why it matters to US</th></tr>{frontier_html}</table></div>

<div class="section-label">Bulletproofing &amp; open items</div>
<div class="card"><ul class="status-list">{bullets_html}</ul></div>

</div>
</body>
</html>"""

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(htmldoc)
    print(f"wrote {args.out} ({len(htmldoc)} bytes)")


if __name__ == "__main__":
    main()

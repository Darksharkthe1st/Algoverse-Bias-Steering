# HANDOFF — Edward (infra · agents · measurement)

> Read `PAPER_FRAMING.md` before writing any paper/summary text. Disagree → PR
> that file, don't fork the narrative.

## Shipped (this PR)

- Team kit: dashboard (`scripts/build_dashboard.py` → `dashboard/index.html`,
  CI rebuild), `PAPER_FRAMING.md` doctrine, `CLAUDE.md`/`AGENTS.md`,
  `dashboard-update` skill, per-person handoffs.
- `docs/2026-08-01_project_analysis.md` — post-mortem + frontier scan.

## Week 1 (sprint plan §3; Gate 1 owner)

1. **Judge v2 rubric** (with Farhan's failure-mode list): two-axis —
   stance-taking (yes/no) × hedging register — seeded from the
   `farhan-opinion-spectrum` branch. Annotate ~150 gold labels with Jeremiah;
   iterate to Cohen's kappa ≥ 0.7 (max two iterations); then re-judge
   the archived logs (~$20–50 API). Gate 1: kappa AND archived vectors moved
   stance-taking ≥ 10pp — bring both numbers to the Sunday gate decision.
2. **First per-layer cosine figure by Friday** — second task, deliberately on
   the caching code (bus-factor onboarding: if Farhan is out in week 3, you
   run the degraded grid).
3. **Compute wiring**: Lambda launch recipe (SSH + notebook, per Farhan's 2025
   flow) documented in `docs/`; check cluster access as backup.

## Week 2–3 (evals track — agent-assisted)

- Benchmark ingestion harness (`experiments/eval_harness/`): IssueBench
  stratified subset, Anthropic Paired Prompts, XSTest, JailbreakBench,
  pluggable into `src/data.py`'s loaders; Jeremiah assists, Farhan reviews.
- Geometry package: per-layer cosines + principal angles, all 4 models.
- Week 3: judging + metric aggregation as grid cells stream in (pinned
  GPT-4o-mini, ~$75–125 cash total).

Done already: venue verified (Interp4Discovery @ NeurIPS 2026, Aug 29 AoE —
docs/2026-08-01_venue_scan.md).

## Standing

- Keep the dashboard honest (`dashboard-update` skill); doctrine + dashboard
  change in the same PR when framing moves.
- Agent infra: multi-agent research/verification runs on demand; results
  always land as committed artifacts, never as chat-only claims.

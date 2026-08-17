> ## 📓 NON-CANONICAL — personal working view
>
> This is a scratchpad. It may hold commands, local notes and handoff context.
> It may **not** redefine the paper, an experiment, a metric, a model set, a
> deadline, a rubric, a claim, or a task's definition of done.
>
> Canonical: **`PROJECT_STATE.md`** · **`RESEARCH_CONTRACT.md`** ·
> **`WORK_LEDGER.md`** · **`docs/PREREG.md`**.
> If this file disagrees with those, this file is stale.

# RUNBOOK — Edward (measurement · evals · infrastructure)

> Facts live in their owners — see `docs/SOURCES_OF_TRUTH.md`. This file holds
> only what I'm doing, never what is true.

## Done

- Two independent verification passes on the archive; the retraction and the
  norm-profile finding (`docs/VERIFICATION_2026-08-07.md`).
- Prior-art scan and the venue conflict (`docs/PRIOR_ART_2026-08-07.md`).
- Model-set refresh, verified against HF configs and the TransformerLens
  registry (`docs/MODEL_SET_2026-08-07.md`).
- Team kit: dashboard (live at https://bias-steering.exe.xyz), framing
  doctrine, agent instructions, runbooks, fact registry.
- Rubric reconciliation — four competing versions merged into
  `docs/RUBRIC_v2.md`.
- NotebookLM pack regenerated clean of the retracted claim.
- PR #1 open, awaiting Farhan's review.

## Now — Gate 1 owner (blocks everything downstream)

1. **File `docs/PREREG.md`** once the rubric freezes. The skeleton is there;
   the rubric commit hash is the field that matters.
2. **Gold-set annotation with Jeremiah** — ~150 archived responses, blind to
   arm, independent passes, per-category κ ≥ 0.70. My job is the harness and
   the agreement statistics; the rubric itself is a team decision, not mine.
3. **Re-judge the archived outputs** under the frozen rubric, with the scaffold
   and control tokens normalized out first. Report per-example distributions.
4. **Gate 1 call** — kappa met, and did the 2025 vectors move *stance-taking*
   (not just hedging register) by ≥10pp? Bring both numbers to the meeting.

## Next — evals and geometry support

- Benchmark harness (agent-assisted): IssueBench subset, Anthropic paired
  prompts, XSTest, JailbreakBench, pluggable into the existing loaders.
- Geometry support for Jeremiah's workstream 2 — he owns the analysis; I own
  making the numbers cheap to produce.
- Judging and aggregation as grid cells land.

## Standing

- Keep the claim ledger honest. It only works if it is the thing people edit,
  not a summary of things edited elsewhere — that is how four rubrics happened
  under a ledger built to prevent exactly that.
- Daily scoop watch: refusal direction × tokenization/perturbation ×
  identifiability. Pre-committed response is in
  `docs/PRIOR_ART_2026-08-07.md`.
- Dashboard rebuild + redeploy on any state change:
  `python3 scripts/build_dashboard.py --out dashboard/index.html` then
  `scp dashboard/index.html bias-steering.exe.xyz:/var/www/html/index.html`.

## Not mine

Merging PR #1 — that is Farhan's, as repo owner, first author, and the person
whose code the audit is about. Rubric content, the venue choice, and the
scope call are team decisions.

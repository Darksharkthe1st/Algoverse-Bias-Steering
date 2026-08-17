# PROJECT_STATE.md

**Front door. Two minutes. If this disagrees with anything else, this and
`RESEARCH_CONTRACT.md` win.**

| | |
|---|---|
| **Venue** | Interpretability as a Science @ NeurIPS 2026 (Sydney), long track ≤9pp, non-archival |
| **Deadline** | **2026-08-28 AoE** · numbers freeze **2026-08-24** · red team **2026-08-26** |
| **Next decision** | **Tue 2026-08-19** — ratify the frozen contract, or amend it |

## The paper, in one sentence

CoCoNot built contrastive counterparts for its refusal categories but explicitly
not for indeterminate/subjective ones — we build that missing contrast set
(benign comparative questions that *do* have a right answer) and test whether
hedging on them is controlled by the same mechanism as harm refusal.

## The experiment

Partial directional ablation `x − λ(x·r̂)r̂` at λ ∈ {0, 0.5, 1}, for
`r̂_stance` and `r̂_harm`, measured on both batteries. Primary statistic **θ**, the
angle between the two directions' trajectories **in logit space**.

- **θ small (CI_hi < 25°)** → one shared control
- **θ large (CI_lo > 25°)** → distinct controls
- **otherwise** → inconclusive, reported as such

Four gates must pass first or no claim is made: positive control, direction
precision, random-direction specificity, coherence.

## Hypothesis space

Shared control (θ≈0–6°) · oblique/partial (≈51°) · nested (≈66°) · distinct (≈90°).
Two worlds *mimic* "shared" and are caught only by gates: generic rank-1 damage
and positive-control failure.

## Current gate

**G1 — the positive control has never been run.** Ablating `r̂_harm` must suppress
harm refusal by ≥0.15 on the primary model. No 2026-model generation exists yet;
every number in every plan so far is simulated or assumed. **If G1 fails there is
no paper.**

## Decided most recently (2026-08-17)

- **"Soft refusal" retired.** The behaviour is *hedging* (arXiv:2502.19463); the
  failure mode is *over-abstention on answerable items* (AbstentionBench).
  Naming is not a contribution.
- **The 296-item battery is a three-way mixture** — 28 items are CoCoNot
  Humanizing, ~75 are Indeterminate–Subjective, and Joad et al. already have
  directions for both. Only the ~193 privileged-answer items (**S2**) are the
  primary battery. Fitting one direction over all 296 would have made the central
  claim unidentifiable.
- **Selectivity ratio dropped; logit-space trajectory angle adopted.** Simulation:
  the ratio rule fires 1.8%→64% with no second mechanism present. Under a shared
  knob `Δlogit P_harm / Δlogit P_stance` is constant across λ *and* directions —
  that invariant is the identification, and probability space destroys it.
- **λ grid shrunk to {0, 0.5, 1}.** Denser grids cost power at matched budget.
- **Qwen 27B trio cut** — no `-Base` checkpoints, so no control.

## Running now

Final adversarial freeze review (novelty · identifiability · statistics ·
execution). **Nothing else should start until the contract is ratified.**

## Blocks the paper

Positive control (G1) · ablation operator (does not exist) · extraction protocol
(current `generate_with_cache()` substrate is invalid) · stratum labels on the 296
items · DV extractor + ternary validation · preregistration hash.

## Does *not* block the paper

Second model · S3 appropriate-hedging arm · post-training trajectory · SAEs ·
ACE/cone/gradient methods · bias taxonomy · the forensic reconstruction of the
2025 scalar-broadcast bug · dashboard.

## Canonical evidence

`RESEARCH_CONTRACT.md` (science, frozen) · `WORK_LEDGER.md` (execution) ·
`docs/PREREG.md` (hash before any off-target read) · `docs/REVIVAL_AUDIT.md`
(why the 2025 numbers are not evidence) · `analysis/sim_lambda_*.py` (the
simulations behind the decision rule) · `runs/` (12 recovered campaign runs).

## Standing rules

Personal runbooks are scratchpads. They may record commands and handoff notes.
They may **not** redefine the paper, an experiment, a metric, a deadline, a model
set, a rubric, a claim, or a definition of done. If one disagrees with this file
or the contract, it is stale.

A work package is done when its **evidence exists and validates** — not when
someone reports that it ran.

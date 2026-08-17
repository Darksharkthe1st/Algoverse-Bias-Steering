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

Partial directional ablation `x − λ(x·r̂)r̂` at λ ∈ {0, 0.5, 1}, for `r̂_stance`,
`r̂_harm` and `r̂_random`, on both batteries. Statistic **θ** = angle between the
two directions' trajectories **in logit space**, 90% CI by item bootstrap.

**We claim only the equivalence direction.** `CI_hi < 25°` → consistent with the
shared non-compliance control. Otherwise we report **θ̂ with its CI as a
preregistered bound**, plus the n needed to resolve it — because at k=1 (forced by
greedy decoding) the 296-item asset cannot separate shared from nested, and every
nuisance biases θ toward "distinct". A "distinct control" headline is not
available and is not being sought.

Four gates must pass or no claim is made: positive control, direction precision,
random-direction specificity, coherence.

## Hypothesis space

Shared control (θ≈0–6°) · oblique/partial (≈51°) · nested (≈66°) · distinct (≈90°).
Two worlds *mimic* "shared" and are caught only by gates: generic rank-1 damage
and positive-control failure.

## Current gate

**G0 — merge, Tue Aug 18 12:00.** `origin/fk/init-refusal-rewrite` and
`fix/steering-shape-guard` into `main`. Both are clean; the first is a
fast-forward. This retires three supposed blockers as merge commits.

**G1 is NOT satisfied.** `runs/20260816-011914_refusal-repro_qwen-1.8b` shows the
*mechanism* — harmful 38/100 → **0/100** under ablation — but the run's own
findings doc records **"Not reproduced"**: baseline 0.380 vs the paper's 0.700
(Δ −0.32), extraction cosine **0.90** vs a 0.999 target, on Qwen1.5-1.8B, which is
not a submission model. It proves the operator works. It is not the gate.

**G1 now requires, on the submission model:** ΔP_harm ≤ −0.15 **and** cosine ≥ 0.95
**and** baseline within ±0.05 of reference.

**Then G2, Thu Aug 20** — pilot at λ=1, ~600 generations, ~6 min GPU, needs
`z_stance ≥ 4`. If it fails, no achievable n rescues the identification question
and we publish the bound.

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

Nothing. The freeze review is complete and its verdict is in the contract (§0,
amendments A1–A3). **Execute, validate, write.**

## Blocks the paper

Merge (G0) · partial-λ on the existing ablation operator (~15 LOC — the operator
itself already exists) · fp32 projection · stratum labels on the 296 items · DV
extractor + ternary audit (n=120, 2 annotators) · preregistration hash · the two
statistical fixes (delete whitening; match bootstrap map).

## Does *not* block the paper

Second model · S3 appropriate-hedging arm · post-training trajectory · SAEs ·
ACE/cone/gradient methods · bias taxonomy · the forensic reconstruction of the
2025 scalar-broadcast bug · dashboard.

## Freeze

**Tag `freeze-2026-08-17` · SHA `aed0141`.** Any change
to the science after this point needs a dated amendment in
`RESEARCH_CONTRACT.md` §12 and an entry in `DECISION_LOG.md`.

Superseded doctrine lives in `docs/superseded/`, each file carrying a banner. It
is retained for provenance and **does not govern**. `RUNBOOK_*` and `HANDOFF_*`
are personal scratchpads, marked non-canonical.

## Canonical evidence

`RESEARCH_CONTRACT.md` (science, frozen) · `WORK_LEDGER.md` (execution) ·
`docs/PREREG.md` (hash before any off-target read) · `docs/REVIVAL_AUDIT.md`
(why the 2025 numbers are not evidence) · `analysis/sim_lambda_*.py` (the
simulations behind the decision rule) · `runs/` (12 recovered campaign runs +
the refusal repro) · `DECISION_LOG.md` (why earlier documents no longer apply).

## Standing rules

Personal runbooks are scratchpads. They may record commands and handoff notes.
They may **not** redefine the paper, an experiment, a metric, a deadline, a model
set, a rubric, a claim, or a definition of done. If one disagrees with this file
or the contract, it is stale.

A work package is done when its **evidence exists and validates** — not when
someone reports that it ran.

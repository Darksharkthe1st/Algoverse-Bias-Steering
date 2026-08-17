# PROJECT_STATE.md

**Front door. Two minutes. If this disagrees with anything else, this and
`RESEARCH_CONTRACT.md` win.**

| | |
|---|---|
| **As of** | **Mon 2026-08-17** · 11 days to deadline |
| **Venue** | Interpretability as a Science @ NeurIPS 2026 (Sydney), long track ≤9pp, non-archival |
| **Deadline** | **Fri 2026-08-28 AoE** · numbers freeze **Mon 2026-08-24** · red team **Wed 2026-08-26** |
| **Next decision** | **Tue 2026-08-18** (team call) — G1 result on a submission model. Contract is frozen; only §12 reopens it |

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

**G1 is the only current gate, it is NOT satisfied, and it has never been run on
a submission model.** It is now *evaluable* on one, which it was not before.

**G1 requires all three legs**, on `Qwen/Qwen3-8B` @ `b968826d9c46`, computed
from the model's own activations (contract §12 **A6**, PREREG §7a): **G1a** the
direction is estimable — split-half cosine beats its label-permutation null and
clears 0.68; **G1b** full ablation cuts refusal by **≥ 0.15** on held-out
`harmful_test`; **G1c** permuted and covariance-matched random directions each
move it by **< 0.05** and sit ≥ 4 SE below `r̂_harm`. All three, or no claim.

**Why it changed.** The frozen form demanded a cosine and a baseline *against
Arditi's reference*. He publishes those for five models and `Qwen3-8B` is not
one, so on the frozen primary both criteria were **undefined** — G1 could be
neither passed nor failed. The gate was wrong, so the gate changed. **We did not
switch models to rescue it**: letting a third-party file's availability pick the
submission model is how a project ends up reporting on whatever was convenient.

**Blocked on hardware.** `transformer_lens` is not importable outside the Lambda
box, so G1 and every torch-touching change can only run there. That serialises
the critical path onto whoever has box access — the single-engineer risk, now
verified rather than assumed. Run it from `docs/HANDOFF_G1.md`.

**Why the existing run does not count.**
`runs/20260816-011914_refusal-repro_qwen-1.8b` shows the *mechanism* — harmful
38/100 → **0/100** under ablation — but its own findings doc records **"Not
reproduced"**: baseline 0.380 vs the paper's 0.700 (Δ −0.32), extraction cosine
**0.90** vs a 0.999 target, on Qwen1.5-1.8B. It is kept as historical mechanism
evidence and is not the gate. Contract §12 **A3** once called it "PASSING";
**A5 withdrew that**.

### Already closed

**G0 — reconciliation: DONE.** `fk/init-refusal-rewrite`, `team-kit` and
`fix/steering-shape-guard` are merged into `main` at `aed0141`; those branches are
deleted. Suite green. The ablation operator, prompt-position extraction and the
deterministic refusal judge are all on `main` — they were merges, not builds.

### Next after G1

**G2 — Thu 2026-08-20.** Pilot at λ=1, ~600 generations, ~6 min GPU, needs
`z_stance ≥ 4`. Not current: it cannot be read until G1 passes, because a failed
positive control makes every downstream number uninterpretable. If G2 then fails,
no achievable n rescues the identification question and we publish the bound.

## Decided most recently (2026-08-17)

- **"Soft refusal" retired.** The behaviour is *hedging* (arXiv:2502.19463). The
  failure mode is over-abstention on an **answerable** item — and that is the
  gap, not a citation: **AbstentionBench (arXiv:2506.09038) targets *unanswerable*
  questions.** Our S2 items are answerable, so AbstentionBench names the family
  but does not cover our case. Naming is not a contribution.
- **The battery is a three-way mixture** — 296 rows dedup to **293 unique items**
  (3 exact duplicates removed). 28 are CoCoNot Humanizing (**S1**, excluded by
  deterministic regex) and the rest split S2/S3, with Joad et al. already holding
  directions for the S1/S3 behaviours. Only the privileged-answer stratum
  (**S2**) is the primary battery; **the S2/S3 split is set by blind three-way
  human adjudication, ties → S3** (PREREG §3), so the exact S2 count is an
  output, not an assumption. Fitting one direction over the unstratified set
  would have made the central claim unidentifiable.
- **Selectivity ratio dropped; logit-space trajectory angle adopted.** Simulation:
  the ratio rule fires 1.8%→64% with no second mechanism present. Under a shared
  knob `Δlogit P_harm / Δlogit P_stance` is constant across λ *and* directions —
  that invariant is the identification, and probability space destroys it.
- **λ grid shrunk to {0, 0.5, 1}.** Denser grids cost power at matched budget.
- **Qwen 27B trio cut** — no `-Base` checkpoints, so no control.

## Running now

Nothing. The freeze review is complete and its verdict is in the contract (§0,
amendments A1–A5). **Execute, validate, write.**

## Blocks the paper

**G1 on the submission model** (needs the Lambda box) · partial-λ on the existing
ablation operator (~15 LOC — the operator itself already exists) · fp32
projection · stratum labels on the 293 items · DV extractor + ternary audit
(n=120, 2 annotators) · preregistration hash · the two statistical fixes (delete
whitening; match bootstrap map).

## Does *not* block the paper

Second model · S3 appropriate-hedging arm · post-training trajectory · SAEs ·
ACE/cone/gradient methods · bias taxonomy · the forensic reconstruction of the
2025 scalar-broadcast bug · dashboard.

## Freeze

| Tag | SHA | What it marks |
|---|---|---|
| `freeze-2026-08-17` | `aed0141` | **The science.** Every pre-outcome choice in PREREG §1–§10. Never moves. |
| `freeze-2026-08-17-a1` | `25a18c7` | **Current.** Adds Amendment **A4** — model revisions pinned to immutable SHAs (PREREG §3b). No hypothesis, statistic, threshold, gate or model set changed. |

Amendments live in `RESEARCH_CONTRACT.md` §12: **A1** identification claim cut ·
**A2** two statistical defects · ~~A3~~ **withdrawn** · **A4** model pin ·
**A5** withdraws A3 (G1 is not satisfied) · **A6** G1 redefined model-internally
(protocol validity — the frozen form was undefined on the frozen primary). Any
change to the science after this point needs a dated amendment there and an
entry in `DECISION_LOG.md`.

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

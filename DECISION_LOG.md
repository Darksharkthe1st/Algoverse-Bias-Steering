# DECISION_LOG.md — append-only

**Historical reasoning, not current doctrine.** Nothing here governs the project;
`PROJECT_STATE.md`, `RESEARCH_CONTRACT.md`, `WORK_LEDGER.md` and `docs/PREREG.md`
do. This file exists so that a newcomer — human or agent — can see *why* earlier
documents no longer apply, and so nobody re-derives a cut idea in week three.

Append only. Never rewrite an entry; supersede it with a later one.

Evidence is cited by commit, run directory, file, or primary source. Where a
meeting transcript conflicts with committed evidence, **the discrepancy is
recorded rather than silently resolved.**

---

## D-001 · 2025 (original cohort) · The founding hypothesis
**Decision.** An LLM's tendency to be neutral rather than opinionated on
contested-but-benign prompts is mediated by a linear residual-stream direction,
extractable by difference-in-means.
**Evidence.** Original Algoverse cohort, Jun–Dec 2025. Paper never finished;
Overleaf draft and Slack archive both lost.
**Status.** Superseded by D-004 and D-009.

## D-002 · 2026-08-04 · The asymmetry that founded the revival
**Decision.** Treat as the project's central finding that the bias vector
transferred to refusal data while the refusal vector did not transfer to bias
data — implying bias is the more general construct.
**Evidence.** Farhan, Aug 4 meeting, from memory: *"The refusal vector was not
very useful for our bias data set. But the bias vector did work for the refusal
data set."* Coefficient behaviour: refusal gibberish above 1, bias usable at 10–15.
**Status.** **Withdrawn as evidence** by D-004.

## D-003 · 2026-08-08 · Capability checks before claims
**Decision.** Run capability/coherence benchmarks early so steering cannot be
shown to have lobotomised the model.
**Evidence.** Aug 8 meeting. Edward argued against GSM8K as the only skill check.
**Status.** Survives, folded into gate G4.

## D-004 · 2026-08-16 · The 2025 refusal arms are not evidence
**Decision.** Withdraw all 2025 refusal-arm results, including D-002's asymmetry.
**Evidence.** `docs/REVIVAL_AUDIT.md`: archived `.pt` files load as
**one-dimensional** tensors, and `layered_generation` indexed them per layer —
which on a 1-D tensor yields a **scalar** — then added that scalar uniformly
across the residual width. A DC offset, not a direction. Audit verdict: *"a
pipeline-failure case study, not intervention evidence."* Separately, one headline
run saved a `Qwen-1_5-1_8B` payload under a Llama label.
**Consequence.** The reported symptom in D-002 (gibberish above coefficient 1) is
exactly what a uniform DC offset produces. The founding asymmetry may never have
existed.
**Status.** Governing. The bug now appears only as a methods correction — never as
a contribution. A single unpublished student project is not a finding about the
literature.

## D-005 · 2026-08-16 · Documentation had forked into two universes
**Decision.** Reconcile `main` and `team-kit` rather than continuing on both.
**Evidence.** The branches had diverged 19 commits each way with **disjoint**
`docs/` trees. `RUBRIC_v2.md` — the canonical rubric — did not exist on the branch
the pipeline owner worked from. The file marked *"This file is doctrine"*
described a byte-fallback perturbation paper that **no meeting had ever agreed**;
it was raised once on Aug 1, parked the same call, and never returned across four
subsequent meetings.
**Status.** Executed 2026-08-17 (D-013).

## D-006 · 2026-08-16 · Independence in the annotation instrument
**Decision.** One tab per annotator instead of three visible label columns.
**Evidence.** `RUBRIC_v2.md` requires independent labelling; three adjacent
columns make peeking the default.
**Status.** Superseded by D-010 — the eight-way instrument itself was cut.

## D-007 · 2026-08-16 · Bibliography verified; one citation retracted
**Decision.** Remove Fafuła (arXiv:2607.17427) from the motivation.
**Evidence.** 19/19 inherited arXiv IDs are **real** — zero fabricated, control
test `2607.99999` → 404. But Fafuła's text contains no "soft refusal", no
"neutrality", no "benign", has no future-work section, and states its task
(21,600 stock picks on 60 equities) *elicits no refusals*. The claim that it
"explicitly requests this experiment" was false.
**Status.** Governing. Earlier plans — including ones written in this project's
own chat — cited it as the strongest acceptance argument. It was not.

## D-008 · 2026-08-17 · Geometry is not sufficient evidence
**Decision.** Do not use cosine or principal angle between DiM directions as
evidence of functional separation.
**Evidence.** Wollschläger et al. (arXiv:2502.17420) define *representational
independence* precisely because orthogonality does not imply independence under
intervention. Joad et al. (arXiv:2602.02132) find eleven refusal categories
geometrically distinct (cos 0.4–0.6) yet behaviourally collapsed onto *"a shared
one-dimensional control knob."*
**Status.** Governing.

## D-009 · 2026-08-17 · The selectivity ratio cannot identify mechanisms
**Decision.** Drop the single-point full-ablation 2×2 and its `SEL ≥ 2` rule.
**Evidence.** `analysis/sim_lambda_identifiability.py`. Holding the world **fixed
at shared** and varying only direction-estimate quality — which is unmeasurable —
the rule fires with probability **1.8% → 64%**, and at ρ_stance=0.85/ρ_harm=0.55
it silently **inverts**.
**Status.** Governing.

## D-010 · 2026-08-17 · The eight-way cascade fails as an instrument
**Decision.** Replace it with a ternary judgement (COMMITTED /
ENGAGED-DID-NOT-COMMIT / UNUSABLE) plus a deterministic primary DV.
**Evidence.** Simulation: five of eight categories have expected count <7 at
n=150; hard refusal has a 53% chance of landing below 5 items. **Not a
sample-size problem** — at n=2400 median per-category κ for soft refusal is still
0.52, because the ceiling is set by the 4↔5 and 6↔7 confusions. "Per-category
Cohen's κ" is also undefined for four raters. Splitting validity out as its own
facet additionally dissolves the 2025 "garbage judged neutral" pathology, which
was a **label-space defect, not a judge defect**.
**Status.** Governing.

## D-011 · 2026-08-17 · Logit space is the identification
**Decision.** Primary statistic is the angle between ablation trajectories in
**logit** space, not probability space; λ ∈ {0, 0.5, 1}.
**Evidence.** Under a shared knob, `Δlogit P_harm / Δlogit P_stance` is exactly
**constant** across λ *and* across directions (2.250000 in simulation) — invariant
to direction efficacy and to the differing thresholds and baselines that break the
ratio rule. Probability space destroys the invariant: null bias 11.6° vs 3.6°,
and 24.7° vs 8.2° under a 0.99 ceiling. Denser λ grids *cost* power at matched
budget — `{0,1}` at n=1000 gives 0.942 versus 0.525 for a five-point grid at
n=333.
**Status.** Governing.

## D-012 · 2026-08-17 · The battery is a three-way mixture
**Decision.** Stratify. S2 (privileged answer) is the primary battery; S3 is the
appropriate-hedging control; S1 is excluded.
**Evidence.** Of 296 items, 28 are first-person preference (CoCoNot
**Humanizing**) and ~75 are peer-subjective (CoCoNot
**Indeterminate–Subjective**). **Joad et al. already have directions for both.**
A single direction fit over the unstratified set is a mixture, and if that mixture
looked distinct from harm refusal, the mixture would be a sufficient alternative
explanation.
**Positioning gained.** CoCoNot states verbatim that it built contrast sets only
for incomplete, unsupported and safety categories — *"Not all the categories in
our taxonomy have the potential of having contrastive counterparts."* Table 1:
Indeterminate and Humanizing have **none**. The S2 battery *is* that missing
contrast set. Verified against the CoCoNot full text, not a summary.
**Status.** Governing. **"Soft refusal" is retired** — the behaviour is *hedging*
(arXiv:2502.19463); the failure mode is over-abstention on an *answerable* item,
where AbstentionBench (arXiv:2506.09038) targets *unanswerable* ones.

## D-013 · 2026-08-17 · The identification claim is cut
**Decision.** Keep the machinery; reduce the headline to the measurement
contribution plus a preregistered bound. Claim only the equivalence direction.
**Evidence.** Three independent critics, each verified in code.
1. **Greedy decoding makes k>1 fictional.** The protocol mandates greedy for
   Arditi parity, so repeated generations are byte-identical: measured z_stance
   3.01→3.16 across greedy k=1→5, versus 3.11→4.54 sampled. k=1 is forced, and at
   k=1 the 296-item asset gives correct-verdict rates **0.11 / 0.14 / 0.72 / 0.02**.
2. **Reaching the claim needs 4.7× the asset** — n_ben=1400, n_harm=700, ~10,500
   generations, ~2,100 audit judgements, against 293 items and 11 days.
   `n_harm` binds, not `n_ben`.
3. **Every nuisance biases θ upward, toward "distinct"**; none toward "shared". So
   the design can confirm Joad et al. but cannot credibly refute them.
**Status.** Governing. This fired the contract's own stop rule §12.5.

## D-014 · 2026-08-17 · Two statistical defects fixed before any θ
**Decision.** Delete the axis whitening; make the bootstrap use the same map as
the statistic.
**Evidence.** Whitening ties the estimand to the benign/harm budget split — θ_eq
calibrated at 1:1 becomes 33° at 14:1, inside the bound meant to exclude *nested*.
Mismatched bootstrap maps give measured coverage **0.22** under nested, erring
**78%** toward falsely declaring "shared" — the paper's own null.
**Status.** Governing. Both are an afternoon of work and must land before any θ is
computed.

## D-015 · 2026-08-17 · The positive control is not yet satisfied
**Decision.** G1 requires a run on a **submission** model meeting three
conditions, not just the effect size.
**Evidence.** `runs/20260816-011914_refusal-repro_qwen-1.8b` shows harmful
38/100 → **0/100** under ablation. But the run's own findings doc records
**"Not reproduced"**: baseline 0.380 vs the paper's 0.700 (Δ −0.32) and extraction
cosine **0.90** against a 0.999 target.
**Discrepancy recorded.** Earlier planning in this project — including agent-written
plans — reported this as "G1 already passes". It does not. The *mechanism*
reproduces; the *replication* misses its own bar, on a non-submission model.
**Status.** Governing. G1 now also requires cosine ≥ 0.95 and baseline within
±0.05 of reference.

## D-016 · 2026-08-17 · The campaign data was never lost
**Decision.** Do not re-run the Aug 9 campaign.
**Evidence.** 12 of 13 run directories appeared to hold only a 167-byte log. The
artifacts **were** committed at phase boundaries (`b95ded9` and siblings) and were
later dropped from HEAD by a checkpoint commit. **67 artifacts across 12 runs
recovered from history**; `results.csv` files are 301 rows (100 items × 3
conditions). Only `anchor-qwen-14b` has no vector commit — it died before the
vector phase, 14B, OOM the obvious hypothesis.
**Incidental.** Every recovered vector is `(24, 2048)` — 2-D and correctly shaped.
The 1-D failure of D-004 is confined to the 2025 `experiments/*_vecs` archive and
never entered the refactored pipeline's own outputs.
**Status.** Governing.

## D-017 · 2026-08-17 · Venue
**Decision.** Interpretability as a Science @ NeurIPS 2026, Sydney. ≤9pp long
track, non-archival. Deadline **2026-08-28 AoE**.
**Evidence.** Verified from primary CFPs. Its scope is standards for measurement,
causal claims and falsifiability — which is what this paper now is. The
alternative, Interp4Discovery (Aug 29, ≤5pp), frames around domain knowledge
discovery, which fits worse, and 5pp cannot hold the controls. **InterpScience
forbids concurrent workshop review — one venue only.**
**Status.** Governing.

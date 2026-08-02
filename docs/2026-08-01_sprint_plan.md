# Sprint plan — fused recommendation (2026-08-01)

*Produced by a multi-agent planning pass (5 lens-diverse proposals → 3
adversarial judges → fusion), then edited for team assignments. **Status:
recommendation — needs team sign-off at the Tue meeting.** The thesis and
experiment list here are the working content of `PAPER_FRAMING.md`; if the
team changes them, change both files in the same PR.*

## 1. Recommended thesis (one decisive call)

**Run the soft-refusal vs. harm-refusal dissociation paper — the strongest
proposal's science on the feasibility-first budget discipline, with a zero-GPU
construct gate as Week 1.**

**Working title:** *"One Knob or Two? Dissociating Soft Refusal (Opinion
Suppression) from Harm Refusal in Chat LLMs"*

**Thesis:** In instruction-tuned chat LLMs across 3+ model families, the
residual-stream direction mediating soft refusal (declining to take a side on
contested-but-benign prompts) is causally and geometrically dissociable from
the Arditi harm-refusal direction — or measurably entangled with it, in which
case the entanglement geometry (per-layer cosines, principal angles)
mechanistically explains Fafuła's abliteration side effects. Both directions
are extracted under a **single unified Arditi-convention pipeline**
(post-instruction token positions), eliminating the extraction-convention
confound. Either branch of the test is the paper.

**Positioning against the closest 2025–26 work** (verified by the novelty judge):
- **QCRI, arXiv:2602.02132** ("More to Refusal than a Single Direction"): its
  eleven non-compliance flavors are all harm/compliance-side; opinionation is
  never included and no bias/neutrality benchmark is used. We add the missing
  flavor and run the cross-steering test QCRI never ran.
- **Fafuła, arXiv:2607.17427** ("Abliteration Is Not a Scalpel"): documents
  off-target disposition shifts behaviorally and explicitly requests the
  mechanistic complement. We supply it.
- **Wollschläger, arXiv:2502.17420** (concept cones): harm-refusal geometry
  only; we test where an independently-derived soft-refusal direction sits
  relative to it — with cosines/principal angles, not a cone-fitting
  reimplementation (see scope cuts).
- **Multiverse, arXiv:2512.16602 / Nadeem, arXiv:2508.08846**: steer political
  refusal or ideological lean respectively, but neither measures the safety
  side; neither compares geometries.
- **Hygiene bar**: AxBench (arXiv:2501.17148) system-prompt baselines, Tan et
  al. (arXiv:2407.12404) multi-family + per-example distributions, activation
  capping instead of constant-add — built in from day 1.

"Soft refusal" as a named activation direction is unclaimed as of Aug 2026.
Claim the term.

**Models (4, three families, one modern):** Qwen1.5-7B, gemma-2b-it,
Llama-3-8B-Instruct, Qwen2.5-7B-Instruct. Qwen2.5 answers the model-vintage
objection; Qwen1.5-14B and Yi-6B are dropped (budget, redundancy).

## 2. Prioritized experiments

**P0 — must ship (the paper):**
1. **Two-axis judge rubric** (stance-taking yes/no vs. hedging register),
   validated to Cohen's kappa ≥ 0.7 against ~150 hand-labeled examples from
   the archived logs. Everything downstream depends on this.
2. **Retroactive re-judging** of the archived steered generations in
   `experiments/past_logs/` under the new rubric — Gate 1, near-zero cost, on
   sunk compute.
3. **Unified re-extraction** of both the opinionation and Arditi harm-refusal
   directions with post-instruction-position conventions on all 4 models (the
   committed mean-pooled vectors in `experiments/best_vecs/` become sanity
   cross-checks only).
4. **The 2×2 cross-steering grid**: 4 models × 5 conditions (baseline,
   system-prompt baseline, ±opinionation steer, harm-refusal ablation) × 2
   batteries (opinion: ~100 held-out comparison prompts + ~300 stratified
   IssueBench + Anthropic Paired-Prompts subset; safety: XSTest ~250 +
   JailbreakBench ~100), with per-example distributions and 95% CIs.
   Activation capping (95th-percentile clamp), not constant coefficients.
5. **Geometry figure**: per-layer cosines and principal angles between the two
   direction families, same-pipeline extraction.

**P1 — strongly should ship:**
6. MMLU-subset capability audit (~500 items per condition).
7. Contrast-set resampling (3 re-draws) on 2 models to bound vector variance
   (Tan et al. requirement).
8. Human spot-check of ~200 judgments + judge-agreement stats.

**P2 — only if ahead of schedule:**
9. Geometry-vs-crosstalk scatter across the 4 models (descriptive, not
   statistical — n=4 is anecdotal).
10. Directional-ablation variant on the most interesting model.
11. QCRI-style shared-core projection (timeboxed to 2 days, drop silently if
    it slips).

## 3. Week-by-week plan

**Week 1 — Construct gate + unified extraction (GPU ~20 hrs)**
- **Edward** (measurement lead): build the two-axis rubric; run annotation
  with Jeremiah (~150 gold labels); iterate to kappa ≥ 0.7 (max two
  iterations); re-judge archived logs via GPT-4o-mini (~$20–50 cash).
- **Farhan**: unified re-extraction pipeline — port opinionation extraction to
  Arditi conventions, extract harm-refusal directions from
  Do_Not_Answer/AdvBench contrasts, all 4 models. Cross-check against
  committed vectors and `experiments/past_vecs/calculated_refusal_vecs/`.
- **Edward** (second task, forced onboarding onto the caching code): first
  per-layer cosine figure by Friday.
- **Gate 1 decision Sunday** (see kill criteria).

**Week 2 — Causal sanity + geometry (GPU ~25 hrs)**
- **Farhan**: reproduce bidirectional in-distribution opinionation steering
  with re-extracted vectors (per-example distributions, system-prompt
  baseline) and Arditi refusal bypass/induction on the same models,
  ~100-prompt probes.
- **Edward**: full geometry package — per-layer cosines, principal angles,
  all 4 models.
- **Edward** (agent-assisted): benchmark ingestion harness (IssueBench
  stratified sample, Paired-Prompts, XSTest, JailbreakBench) into the
  existing prompt loader; Jeremiah assists, Farhan reviews.
- **Gate 2 decision Sunday.**

**Week 3 — The main grid (GPU ~110–130 hrs, the crunch week)**
- **Farhan**: the 4×5×2 grid (~30k generations, batched TransformerLens — the
  same loop from the 2025 runs). Pre-committed degradation path if hours
  inflate: drop the −opinionation condition first, then drop to 3 models;
  never shrink the safety battery.
- **Edward**: judging + metric aggregation as results stream in
  (agent-assisted); 2×2 matrix analysis, geometry-behavior comparison; starts
  methods/geometry prose.
- **Jeremiah**: MMLU audit runs, judge-disagreement error analysis.

**Week 4 — Robustness + writing (GPU ~25 hrs + reserve; experiments frozen Aug 26)**
- **Farhan**: contrast-set resampling, any spillover cells, reproducibility
  appendix.
- **Edward**: full 5-page draft; Figure 1 = 2×2 cross-intervention matrix with
  CIs, Figure 2 = per-layer geometry. Limitations section explicitly owns
  judge dependence and remaining 2024-vintage models.
- **Jeremiah**: per-example distribution figures, spot-check report; repro
  pass (re-run one grid cell from README instructions); eval appendix; lead
  editing pass for clarity.
- Internal red-team read Aug 27; **submit Aug 29 AoE**.

**Budget:** $377 Lambda ≈ 290 A100-40GB hrs @ $1.29/hr. Planned ~200 hrs +
~60 hr reserve (25%). Judge costs are **cash, not credits**: GPT-4o-mini only,
~$75–125 total. All models ≤8B fit one A100 in bf16.

## 4. Venue (verified — see docs/2026-08-01_venue_scan.md)

- **Target: Interpretability for Discovery @ NeurIPS 2026** (Atlanta) —
  deadline **Aug 29, 2026 AoE**, 5 pp + refs/appendix, non-archival,
  double-blind. Best topical fit; non-archival keeps the work eligible for
  BlackboxNLP 2027 or ICLR 2027 expansion.
- **Backup (same deadline): AI4GOOD @ NeurIPS 2026** (Paris, 2–8 pp) if the
  bias/neutrality societal framing ends up stronger.
- **Slip path:** ICLR 2027 (abstract Sept 18, full Sept 25) with expanded
  results, or ICLR/AAAI-27 workshops in early 2027.

## 5. Risks, kill criteria, and pivot path

**Kill criteria:**
- **Gate 1 (end of Week 1, GPU spend ~20 hrs):** (a) judge-human kappa ≥ 0.7
  after ≤2 rubric iterations, AND (b) re-judged archived outputs show the 2025
  vectors moved stance-taking by ≥10 pp (not merely hedging register).
  **Fail (a):** stop all steering work — nothing downstream is interpretable.
  **Fail (b):** the soft-refusal construct is dead on these vectors; pivot
  immediately (below).
- **Gate 2 (end of Week 2):** (a) re-extracted opinionation vectors reproduce
  bidirectional in-distribution control (≥20 pp shift, per-example
  distributions) on ≥3 of 4 models, AND (b) Arditi bypass/induction reproduces
  (≥30 pp refusal change) on the same models. **Fail (b):** pipeline bug —
  Arditi is heavily replicated; fix before proceeding, do not run the grid.
  **Fail (a):** the 2025 result was an extraction artifact; pivot.
- **Explicitly NOT kill criteria:** any geometry outcome and any
  cross-steering outcome. Dissociation contradicts the one-knob
  generalization; entanglement explains Fafuła. Both publish — that is the
  design.

**Pivot path (if Gate 1b or Gate 2a fails):** the **construct-validity audit
paper** ("Decisiveness Is Not Bias" — the feasibility judge's top pick). Its
decisive experiment — re-judging the archived corpus under the validated
rubric — is *already complete* as a byproduct of Week 1, and its remaining
needs (fresh runs on 3 models, 50–70 GPU-hrs) fit the leftover budget.
Deliverable becomes a measurement/negative-result note ("the 2025
in-distribution result was a hedging-style artifact"), released rubric +
re-judged corpus as artifacts, same venue (the CFP welcomes negative results).

**Top residual risks:**
1. **Bus factor:** Farhan wrote ~95% of the pipeline and Week 3 funnels
   through him. Mitigation: Edward's Week-1 extraction task is forced
   onboarding; if Farhan is unavailable in Week 3, Edward runs the degraded
   grid (3 models, 4 conditions).
2. **Judge validity remains load-bearing** on the opinion side. Mitigation:
   kappa gate before GPU spend, human spot-checks, agreement stats reported.
3. **Scoop exposure:** the QCRI group could add opinionation as a twelfth
   flavor cheaply. Mitigation is speed — the Aug 29 deadline is the moat; do
   not let scope creep slip it.
4. **Muddy middle:** partial, model-inconsistent entanglement. Fallback
   framing: a reliability finding in the Tan/Da Silva lineage — weaker but
   honest, decided by end of Week 3, not Week 4.
5. **Week-3 GPU window:** spot flakiness in that one week has no parallel
   path. Mitigation: 25% reserve, pre-committed degradation order, start the
   grid the moment Gate 2 passes.

## 6. Explicit scope cuts — what NOT to do

- **No Wollschläger cone-fitting reimplementation** and no "X is a cone"
  framing: the "first cone beyond harm-refusal" claim is factually false
  (truth cones, arXiv:2505.21800), the niche is saturated. Cosines +
  principal angles carry the geometry claim.
- **No ideology/lean direction** (the "which side" second knob): the
  literature says the signal is distributed even at 80B; the 2×2 would likely
  collapse to 1×2. Stance-propensity vs. harm-refusal is the winnable
  dissociation. (Keep the opinionation-vs-ideology *differentiation paragraph*
  in the paper; just don't run the experiment.)
- **No MFA regions, no ACE beyond capping, no QCRI shared-core** (shared-core
  demoted to a 2-day P2 stretch).
- **No 5th model, no Qwen1.5-14B** (80GB dependency, budget risk).
- **No 50k-generation grid** — ~30k generations, 4 models, is the ceiling.
- **No dual-judge infrastructure** — single pinned GPT-4o-mini plus human
  spot-checks; a second judge model is $150+ of unbudgeted cash.
- **No reuse of mean-pooled committed vectors in any headline figure** —
  sanity cross-checks only (extraction-convention confound).
- **No ICLR-full-paper ambitions this cycle.** 5 pages, workshop bar,
  submitted on time.

## 7. Where the judges disagreed and how this resolves it

The novelty and reviewer judges both picked the double-dissociation proposal;
the feasibility judge ranked it 4th and picked the construct-validity audit,
with the feasibility-first MVP second. The disagreement is not about the
science — all three judges treat the dissociation question as the strongest
idea — it is about whether its maximal execution (5 models, cone
reimplementation, shared-core analysis, 50k generations) survives contact with
a $377 budget and a bus-factor-1 team.

This plan resolves it by decomposition rather than compromise:
- **From the dissociation proposal**, keep exactly the elements that earned
  its novelty/reviewer scores: the unified re-extraction (killing the
  extraction confound), the modern-model addition (Qwen2.5-7B), and the
  both-branches-publish framing.
- **From the MVP proposal**, take the execution skeleton the feasibility judge
  praised: real budget arithmetic against Lambda credits, cheap judging,
  forced onboarding as bus-factor mitigation, pre-committed degradation paths.
- **From the audit proposal**, take the zero-GPU Week-1 gate (re-judging sunk
  compute before spending credits) — and hold the full audit paper in reserve
  as the pivot, which converts the feasibility judge's top pick from a
  competing headline into insurance that costs one week of API-only work we
  needed to do anyway.
- **Drop** everything the feasibility judge identified as calendar traps
  (cones, MFA, shared core, model #5), which is also the material the novelty
  judge scored as saturated or decorative.

The result is the dissociation paper at the MVP's cost with the audit as the
safety net — the only configuration all three judges' rationales jointly
support.

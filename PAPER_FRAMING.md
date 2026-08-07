# PAPER_FRAMING.md — canonical framing for the soft-refusal paper

**This file is doctrine.** Every human and every agent writing paper text,
related work, summaries, reviews, or rebuttals for this project follows the
framing below. If you (or your agent) think a different framing is better, PR
this file first — don't fork the narrative in a draft. Last updated: 2026-08-01
(revival kickoff). Owner: Edward. **Status: DRAFT — needs Farhan's sign-off and
the thesis-lock decision at the Tue meeting.**

## The one-sentence pitch

> **The soft-refusal direction is not an artifact of the intervention that
> found it.** An input perturbation designed without reference to any
> direction — a single character substitution forcing byte-fallback
> tokenization — displaces activations *along* the soft-refusal direction; the
> signed displacement predicts which prompts cross the boundary; and removing
> that component abolishes the flip while injecting it reproduces it.

**Working title:** *"Two Routes to One Boundary: Convergent Validation of a
Soft-Refusal Direction via Designed Steering and Undesigned Input
Perturbation."* This is an **interpretability paper about whether a direction
is real**, not a debiasing-method paper, not a jailbreak paper.

**Both the title and the claim table below are PROPOSED, not agreed.** They come
from `docs/2026-08-02_sprint_proposal.md` (a machine-generated design +
adversarial-critique + fusion pass). Until the team signs off, the previously
circulated scope in `docs/2026-08-01_sprint_plan.md` is equally live — in
particular, whether the input-perturbation arm belongs in this sprint at all is
an **open question for the team**, not a settled one. Read the proposal's
preamble before treating anything here as decided.

## The claims, in load-bearing order (PROPOSED)

| # | Claim | Role |
|---|---|---|
| **C0** | Soft refusal is separable from factual hedging, non-engagement, incoherence and meta-commentary by a six-way ordered first-match-wins screen at per-category κ ≥ 0.7 vs 150 double-annotated gold labels. | **Gate, not a paper claim.** Nothing below is interpretable without it. |
| **C1** | Under one unified Arditi-convention extraction, with both interventions **dose-matched to equal on-target effect before any off-target number is read**, d_soft moves the opinion battery with safety movement bounded within ±5pp (TOST), and d_harm the mirror. | Setup. Compressed — this is the most-scooped part (arXiv:2602.02132, arXiv:2512.16602). |
| **C2** | Byte-fallback perturbations displace activations *preferentially* along d_soft: variance share beats a **covariance-matched** null by ≥10×, while the **byte-identical retokenization** control and the ASCII-typo control do not, at matched ‖Δ‖. | Payload 1 — selectivity. |
| **C3** | Signed projection at pre-registered l\* predicts per-prompt flips (AUROC ≥ 0.70, DeLong-significant over ‖Δ‖ and over an OOD-direction competitor). | Payload 2 — prediction. |
| **C4** | **Mediation.** Nulling the along-d_soft component of the induced displacement abolishes ≥50% of flips; injecting the measured component into the clean run reproduces ≥40%. | **Payload 3 — the headline, and the ICLR lead.** Converts convergent correlation into convergent causation. |
| **C5** | In fractional depth f = l/(L−1), the perturbation-alignment profile and the single-layer steering-efficacy profile either coincide or dissociate. Both outcomes are the result. | Descriptive. |

**Models — SUPERSEDED 2026-08-07.** This file previously locked a 2025-era set
as settled. The verified set is in **`docs/MODEL_SET_2026-08-07.md`**: the
Qwen3.5 {2B, 9B, 27B} ladder, Qwen3.6-27B (byte-identical config to
Qwen3.5-27B, so a free controlled test of post-training with architecture held
fixed), gemma-4-31B-it and gemma-4-26B-A4B-it (dense vs MoE, same family).
Qwen 3.8 is a watch item with adoption rules in that file.

## Terminology rules — REQUIRED differentiations

- **Soft refusal** = declining to take sides on controversial-but-not-harmful
  prompts ("I can't pick sides", both-sidesing, fence-sitting). NOT the same
  as: **hard refusal** (declining harmful requests, Arditi et al.), **over-
  refusal** (wrongly refusing benign requests, OR-Bench line), or **abstention**
  (declining unanswerable questions). Define all four on first use; our term
  occupies the cell OpenAI's bias framework calls "political refusal" and the
  consistency-training line calls "fence-sitting".
- **Opinionation ≠ ideology.** Opinionation = taking *a* side (scalar);
  ideology = *which* side (direction). Nadeem et al. (arXiv:2508.08846) and
  the multilingual follow-up steer ideology; we steer opinionation. One
  paragraph must state this differentiation explicitly — but we do NOT run an
  ideology-direction experiment this cycle (scope cut: the ideology signal is
  distributed even at 80B scale; the winnable dissociation is vs. harm
  refusal).
- **Never claim "first cone beyond harm refusal"** — factually false (truth
  cones exist, arXiv:2505.21800). We do not fit cones at all; cosines and
  principal angles carry the geometry claim.
- **Perturbation vocabulary.** Perturbations are **human-legibility-preserving**,
  never "semantics-preserving" — the weaker claim is the true one. Lead the
  motivation with *accidental* typographic drift (apostrophe normalization in
  real prompt pipelines), never with adversarial manipulation. We report
  detection coverage and monitor ROC, never attack success.
- **Only contested-but-benign prompts are ever perturbed.** No harmful battery
  is perturbed in any arm, at any dose. This is a construction constraint, so
  no attack-efficacy number can exist in the paper. See decision doc §9.
- **"A direction", never "the direction."** Steering success does not identify
  the representation (non-identifiability, arXiv:2602.06801); refusal-family
  behaviors are cones/subspaces (arXiv:2502.17420). Claims are about causal
  control, not unique representation — unless we specifically test uniqueness.
- **vs. Refusal Steering (Multiverse, arXiv:2512.16602)** — the closest
  neighbor. They remove *refusal on politically sensitive topics* (a censorship
  framing) at 80B scale. We differ on both axes that matter: our construct is
  *opinionation* (the model CAN answer engagedly without taking a side — their
  intervention doesn't distinguish these), and our contribution is the
  factorization + dissociation, not refusal removal. Never frame our work as
  "uncensoring".

## Must-cite table

| Paper | Why (the reviewer objection it pre-empts) |
|---|---|
| Arditi et al., refusal direction (arXiv:2406.11717) | The method lineage; defines hard refusal we dissociate from |
| QCRI, More to Refusal (arXiv:2602.02132) | "Isn't your direction just the shared refusal knob?" — our dissociation answers exactly this |
| Wollschläger et al., Concept Cones (arXiv:2502.17420) | "Single directions are obsolete" — we report subspace/cone geometry, not just one vector |
| Multiverse, Refusal Steering (arXiv:2512.16602) | "This was done in Dec 2025" — differentiation paragraph above |
| AxBench (arXiv:2501.17148) | "Prompting does this better/cheaper" — we include the system-prompt baseline everywhere |
| Fafuła, Abliteration Is Not a Scalpel (arXiv:2607.17427) | Motivates the dissociation; explicitly requests this experiment |
| Nadeem et al. (arXiv:2508.08846) + CLAS (arXiv:2601.23001) | "Political neutrality steering exists" — they steer which-side, we steer whether |
| IssueBench (arXiv:2502.08395) | "Why BBQ/CrowS-Pairs?" — we evaluate on the 2026 standard |
| Tan et al., reliability (arXiv:2407.12404) | "Steering vectors are unreliable" — per-example distributions + multi-family reporting |
| **Adversarial Robustness of Activation Steering (arXiv:2606.07696)** | **NEAREST NEIGHBOUR — cite in the first paragraph.** They measure whether the *steering effect survives* input perturbation. We measure the *projection of perturbed activations onto the direction*, predict per-prompt flips, and test mediation. State the distinction explicitly; do not let a reviewer find it first. |
| Non-identifiability (arXiv:2602.06801) | The objection the whole paper answers: a vector that responds to your intervention is not evidence you found the representation. |
| BPE-fragmentation / guardrail-evasion line (arXiv:2607.01239, 2510.05025, 2506.07948, 2504.11168) | Establishes publicly that real systems manipulate Unicode variants in prompts — the *only* permitted grounding for the perturbation arm's motivation. |
| Depth-migration (arXiv:2606.29196) | Predicts peak relative depth migrates with scale — the confound the Qwen2.5 ladder defuses. Cite as anchor AND caveat. |

Nice-to-cite (one line each, cut first under length pressure): Persona Vectors
(arXiv:2507.21509); Assistant Axis / activation capping (arXiv:2601.10387);
CAST (arXiv:2409.05907); ACE affine editing (arXiv:2411.09003); OpenAI
political-bias axes (political refusal); Anthropic even-handedness eval;
consistency training / fence-sitting (arXiv:2605.22771); manifold steering
(arXiv:2605.05115) if the geometry track activates.

## Framing rules

1. **The 2025 result is inherited, not claimed fresh.** In-distribution
   bidirectional steering (Batched_Gen.csv) is the starting asset; the CrowS
   transfer failure is stated plainly as motivation. Never present judge-v1
   percentages as current results — they are provisional pending re-judging.
2. **Honest negatives are load-bearing.** If the dissociation shows
   entanglement, that IS the result; do not spin it as a limitation.
3. **Numbers trace to committed artifacts** (`experiments/` CSVs/pickles);
   every judged number carries its judge version.
4. **Deadlines are verified against the venue page** before appearing in any
   doc. Current target verified 2026-08-01: Interp4Discovery @ NeurIPS 2026,
   Aug 29 AoE. Re-verify page limits before formatting.
5. Anthropic's even-handedness ideal is *symmetric engagement*, not refusal —
   frame soft refusal as a behavior to understand and control, not as the
   desirable neutral endpoint.

## Expected experiments, in priority order (see docs/2026-08-01_sprint_plan.md)

- **P0 — Judge v2 rubric** (no GPU): two-axis (stance-taking × hedging
  register), kappa ≥ 0.7 vs ~150 hand labels. Seed:
  `farhan-opinion-spectrum` branch.
- **P0 — Re-judge archived 2025 outputs** under v2 (Gate 1: did the old
  vectors move stance-taking ≥10 pp, or only hedging?).
- **P0 — Unified re-extraction** of opinionation + harm-refusal directions,
  Arditi conventions, all 4 models.
- **P0 — 2×2 cross-steering grid**: 4 models × 5 conditions × 2 batteries
  (~30k generations), activation capping, per-example distributions, 95% CIs.
- **P0 — Geometry figure**: per-layer cosines + principal angles.
- **P1** — MMLU capability audit; contrast-set resampling (Tan et al.);
  human spot-check of ~200 judgments.
- **P2** — geometry-vs-crosstalk scatter (descriptive); directional-ablation
  variant; QCRI shared-core projection (2-day timebox).
- **Kill gates**: Week 1 (kappa + archived stance-shift) and Week 2
  (reproduce steering + Arditi) — full criteria and the construct-audit pivot
  path in the sprint plan.

## Where things live

- Venue target: **Interpretability for Discovery @ NeurIPS 2026 — deadline
  Aug 29, 2026 AoE** (verified on CFP 2026-08-01; 5 pp + refs/appendix,
  non-archival, double-blind). Backup: AI4GOOD @ NeurIPS 2026 (same
  deadline). Slip path: ICLR 2027 (abstract Sept 18). Details:
  `docs/2026-08-01_venue_scan.md`.
- Sprint plan (thesis, gates, week-by-week): `docs/2026-08-01_sprint_plan.md`
- Dashboard: `dashboard/index.html` (build via `scripts/build_dashboard.py`)
- Post-mortem + frontier scan: `docs/2026-08-01_project_analysis.md`
- 2025 outline (rough, superseded by this file): Overleaf 4514258212zmrztmsxptvy
- Meeting cadence: Tue/Thu/Sat 9pm ET.
- **Dates: use `docs/2026-08-02_sprint_proposal.md` §5 — freeze Aug 24,
  red-team Aug 26, submit Aug 28** (one day of slack against the AoE deadline).
  This file previously said freeze Aug 26 / red-team Aug 27; that was a second,
  conflicting schedule and is withdrawn. One schedule, one place.
- **Venue is an open decision, not settled** — Interp4Discovery (Aug 29) and
  Interpretability as a Science (Aug 28) are mutually exclusive. See
  `docs/PRIOR_ART_2026-08-07.md`.

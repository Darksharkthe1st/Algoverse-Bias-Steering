# PAPER_FRAMING.md — canonical framing for the soft-refusal paper

**This file is doctrine.** Every human and every agent writing paper text,
related work, summaries, reviews, or rebuttals for this project follows the
framing below. If you (or your agent) think a different framing is better, PR
this file first — don't fork the narrative in a draft. Last updated: 2026-08-01
(revival kickoff). Owner: Edward. **Status: DRAFT — needs Farhan's sign-off and
the thesis-lock decision at the Tue meeting.**

## The one-sentence pitch (working)

> Soft refusal — whether an LLM takes a side at all on controversial-but-not-
> harmful questions — is a measurable, steerable representation, and we test
> whether it is one knob with hard refusal of harm, or two.

**Working title:** *"One Knob or Two? Dissociating Soft Refusal (Opinion
Suppression) from Harm Refusal in Chat LLMs"*. This is an **interpretability
paper about a behavioral construct**, not a debiasing-method paper and not a
refusal-jailbreak paper. Full plan: `docs/2026-08-01_sprint_plan.md`.

## The claims, in load-bearing order (fused recommendation — sign off Tue)

1. **Construct (the fix that unlocks everything):** "soft refusal" can be
   measured separately from factual decisiveness. The 2025 judge conflated
   them; our two-axis judge (v2, validated to Cohen's kappa ≥ 0.7 against
   ~150 hand labels) separates them, and re-judging the archived 2025 outputs
   shows what the original "neutrality direction" actually encoded.
2. **Dissociation (the headline bet):** the soft-refusal direction is / is not
   causally separable from the Arditi hard-refusal direction — a 2×2
   cross-steering grid (steer each, measure both benchmark families: opinion
   side IssueBench + Paired Prompts; safety side XSTest + JailbreakBench).
   **Either outcome is the paper**: separability contradicts the
   shared-refusal-knob finding (arXiv:2602.02132); entanglement
   mechanistically explains refusal-surgery side effects (arXiv:2607.17427).
3. **Geometry (the mechanistic account):** per-layer cosines and principal
   angles between the two direction families, extracted under ONE unified
   Arditi-convention pipeline (post-instruction token positions) so the
   comparison is confound-free. The 2025 mean-pooled vectors are sanity
   checks only — never in a headline figure.

**Models (4, three families, one modern):** Qwen1.5-7B, gemma-2b-it,
Llama-3-8B-Instruct, Qwen2.5-7B-Instruct.

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
- Meeting cadence: Tue/Thu/Sat 9pm ET; experiments freeze Aug 26; internal
  red-team read Aug 27

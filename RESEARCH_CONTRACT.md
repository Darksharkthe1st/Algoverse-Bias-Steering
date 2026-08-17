# RESEARCH_CONTRACT.md — scientific commitments

**Status: FROZEN pending final adversarial review (WP-13).**
Owner: Edward. Venue: **Interpretability as a Science @ NeurIPS 2026**, Sydney,
long track ≤9pp, non-archival. **Deadline 2026-08-28 AoE. Numbers freeze 2026-08-24.**

This file owns the science. `PROJECT_STATE.md` owns current state;
`WORK_LEDGER.md` owns execution. Neither may redefine anything here. Personal
runbooks are scratchpads and may not redefine the paper, an experiment, a metric,
a deadline, a model set, a rubric, a claim, or a definition of done.

**After freeze, material change requires a dated amendment in §12 with a reason.**

---

## 1. The question

> CoCoNot (Brahman et al., NeurIPS 2024 D&B) established that models *should*
> abstain on indeterminate/subjective requests, and built contrastive counterparts
> for its other categories but **explicitly not for this one**. We build that
> missing contrast set — benign comparative questions that *do* have a privileged
> answer — and ask: **when a model engages such a question substantively but names
> neither alternative, is that hedging controlled by the same mechanism as harm
> refusal, or by a distinct one?**

## 2. Construct — grounded, not coined

**"Soft refusal" is retired.** It collides with the safety literature's use for
partial compliance, and the behaviour is already named.

- **Behaviour: hedging** — *"a lack of commitment… mentioning opposing
  perspectives to a question"* (arXiv:2502.19463).
- **Failure mode: over-abstention on answerable items** (AbstentionBench,
  arXiv:2506.09038, for "abstention").

**Operational definition.** *Hedging: a response that engages a forced-binary
comparative question substantively but names neither alternative as the answer.
On items with a privileged answer this is over-abstention; on peer-subjective
items it is appropriate non-commitment.*

### 2.1 The battery is stratified. This is freeze-blocking.

Verified: `datasets/GPT_Prompts/comparison_questions_200.csv`, 296 rows, 296/296
parse as `Which <PROP>: <A> or <B>?`, **293 unique** (3 exact duplicates —
*iron/cardboard*, *flowers/garbage*, *metal/bread* — must be deduplicated or
item-level pairing breaks).

| Stratum | n | Existing construct | Declining is |
|---|---|---|---|
| **S1** first-person preference (*"Which do you prefer…"*) | 28 | CoCoNot **Humanizing** = Joad `Humanizing-CCN` | correct |
| **S2** privileged answer (*bed vs rock*) | ~193 | **none in prior work** | **a failure** |
| **S3** peer-subjective (*zoo vs aquarium*) | ~75 | CoCoNot **Indeterminate–Subjective** = Joad `Indeterminate-CCN` | correct |

**~35% of the battery elicits behaviour Joad et al. already have directions for,
across two different CoCoNot categories.** A single direction fit over the
unstratified 296 is a *mixture*, and if that mixture looks distinct from harm
refusal, the mixture is a sufficient explanation and the claim dies in review.

**Commitments.** S2 is the **primary battery**. S3 is the **appropriate-hedging
control**. **S1 is excluded** — it is `Humanizing-CCN` and does not belong in a
stance battery. Stratum labels ship as a column on all 296 items before any
direction is fit (WP-25). The S2/S3 boundary is currently heuristic (±5 items) and
must be human-adjudicated.

## 3. Hypotheses

- **H_shared** — hedging on S2 and harm refusal are read out from one latent
  control at different operating points (Joad et al.'s "shared one-dimensional
  control knob", extended to a behaviour outside their taxonomy).
- **H_distinct** — they are separately controlled.
- **H_nested / oblique** — partially shared.

## 4. Intervention

Partial directional ablation, applied identically to every arm:

```
x_λ  =  x − λ (x · r̂) r̂        λ ∈ {0, 0.5, 1}
```

**λ ∈ {0, 0.5, 1}, not a dense grid.** Simulation (§5.1) shows denser λ *costs*
power at matched generation budget: at ~6000 generations against a true 16° effect,
`{0,1}` at n=1000 gives power **0.942**; `{0,.5,1}` **0.733**; `{0,.25,.5,.75,1}`
**0.525**; dense-near-1 is no better. Three points is the minimum that supports the
monotonicity gate and gives the secondary LRT df>1. Surplus budget goes to n and k,
never to λ resolution.

## 5. Primary statistic, and why not the obvious one

### 5.1 The single-point selectivity ratio is not identifying — quantified

Holding the world **fixed at H_shared** and varying only direction-estimate
quality (which is unmeasurable), the prior design's `SEL ≥ 2` rule fires with
probability **1.8% → 64%**, and at ρ_stance=0.85/ρ_harm=0.55 it silently *inverts*.
The critique was correct. Confirmed by simulation, not argued.

### 5.2 Logit space, not probability space — this is the identification

Under a shared knob, `Δlogit P_harm / Δlogit P_stance` is **constant across λ and
across directions**, and is invariant both to direction efficacy and to the
differing thresholds and baseline rates that break the ratio rule. Verified
analytically in simulation (exactly 2.250000 at every λ, both directions).
**Probability space destroys this invariant**: null-hypothesis angle bias is 11.6°
in probability space vs 3.6° in logit space, and 24.7° vs 8.2° under a 0.99 harm
ceiling — probability space spills over any usable threshold.

### 5.3 The statistic

For `r ∈ {r̂_stance, r̂_harm, r̂_random}`, at each λ measure `P_stance` on S2 and
`P_harm` on the harm battery. Form `Δ_b(r,λ) = logit P_b(r,λ) − logit P_b(r,0)`,
scale each axis by its item-bootstrap SD, and let `v(r)` be the origin-anchored
first principal component over λ.

> **θ = angle( v(r̂_stance), v(r̂_harm) )**, with a 90% CI from an item-level
> bootstrap (resample items within battery; reuse indices across all λ and
> directions).

**Secondary:** a rank-1 "one-knob" LRT — one scalar per cell reproducing both
behaviours at every λ. Lower power, kept because it tests Joad et al.'s literal
claim and is the only statistic that *needs* the sweep.

### 5.4 Competing result signatures

| World | population θ (logit) | AUC vs shared |
|---|---|---|
| **Shared control** | 0.1° – 6.1° across the whole nuisance sweep | — |
| **Oblique / partial** | 51° | 1.000 |
| **Nested** | 66° | 1.000 |
| **Distinct controls** | 90° | 1.000 |
| **Generic rank-1 damage** | ≈0° — **mimics shared** | 0.194 |
| **Positive-control failure** | ≈0° — **mimics shared** | 0.632 |

**The angle is blind to generic damage and to positive-control failure.** Those
are caught by gates, never by the statistic. This is why the gates are not
optional decoration.

## 6. Gates — all four must pass or no claim is made

| | Gate | Threshold |
|---|---|---|
| **G1** | **Positive control** (Arditi lineage) | ΔP_harm under r̂_harm at λ=1 ≤ −0.15 |
| **G2** | **Precision** — each direction moves its *own* target DV | \|Δlogit\|/SE ≥ 4 |
| **G3** | **Specificity** — covariance-matched random direction | moves both DVs by < 4 SE |
| **G4** | **Coherence / monotonicity** | no reversal >2 SE against trend where \|ΔP\|>0.05; no sign disagreement between DVs when both \|ΔP\|>0.05 |

G2 exists because a weak direction is the dominant failure mode: as the summed
\|ΔP\| falls 0.481 → 0.157 → 0.064, median θ under a true shared world rises
3.3° → 9.9° → **24.3°** (95th pct 124°). G3 exists because incoherence riding on
a real knob biases toward false "distinct" (θ 3.6° → 16.9° as damage goes 0 → 0.20).

## 7. Decision rule and falsifiers — preregistered

**θ_eq = 25°.**

- `CI_hi < 25°` → **one shared control.**
- `CI_lo > 25°` → **not one control.**
- otherwise → **inconclusive**, reported as such.

**Falsifiers.** The shared-knob claim is falsified iff all four gates pass **and**
the bootstrap 90% CI *lower* bound on θ exceeds 25°. The distinct-mechanism claim
is falsified iff all gates pass and `CI_hi < 25°`.

**Why 25°:** shared-world population θ is 0.1–6.1° across the nuisance sweep and
≤10° under mild contamination; oblique 51°, nested 66°, distinct 90°. θ_eq=15°
costs equivalence power (0.75); θ_eq=40° admits false "shared" for nested.

**Power.** Critical values (95th pct under shared): n=150,k=1 → 24.3°;
n=296,k=1 → 15.9°; **n=296,k=5 → 9.2°**. Minimum detectable true angle at 80%
power: **27° at n=296,k=1; 14° at k=5.** Detecting distinct (90°) is ≥0.985
powered at every configuration tested including n=100,k=1. **Run k=5** — the
primary battery is S2 (~193 items), so k=5 is what keeps the oblique case (51°)
comfortably detectable.

## 8. Outcome measurement

**Primary DV: `named_a_side`** — did the response commit to one of the two
explicitly named alternatives? Deterministic extractor, audited on 120
hand-adjudicated responses drawn from 18,698 real archived (prompt, response)
pairs on these exact items:

| | value | 95% Wilson CI |
|---|---|---|
| precision | **0.967** | [0.886, 0.991] |
| NPV | **0.917** | [0.819, 0.964] |
| signed bias | **−0.041** | |

**The extractor under-counts commitment by ~4pp. That is tolerable if constant
across λ and fatal if not** — so bias is estimated *per λ level*, not once.

Known systematic failures, all measured: head-noun collisions (40/293),
scope-sharing modifiers (48/293), **unnameable alternatives** (18/293 — e.g.
*"a seatbelt or no seatbelt"*, where commitment to side B can never be extracted;
**these items are excluded from the primary DV**), morphology, negation scope,
truncation at the token cap.

**Validation layer: one ternary judgement**, blind to λ, to direction, and to the
extractor's call, with conditions interleaved in one shuffled sheet. Two
annotators per response, third adjudicates.

> **1 — COMMITTED** · **2 — ENGAGED, DID NOT COMMIT** · **3 — UNUSABLE**
> (verbatim instrument in `WORK_LEDGER.md` WP-07)

Label 3 collapses the entire confound set the DV cannot separate on its own —
incoherence, hard refusal, non-engagement, meta-commentary — into one category,
because for this DV they are interchangeable. **The eight-way cascade is dead and
may not return as a dependency.**

## 9. Controls — operator-matched, and one that is not

**Causal controls use the same operator as the primary (directional ablation):**
covariance-matched random direction at the same λ grid (G3); wrong-layer ablation.

**Forensic demonstration, kept separate:** the 2025 scalar-broadcast failure may be
deliberately reconstructed because it is scientifically informative. **It is an
additive scalar offset, not an ablation, and must never be reported as an
operator-matched control.** (WP-30.)

## 10. Model set

| Role | Model |
|---|---|
| Primary | `Qwen3-8B` — dense `Qwen3ForCausalLM`, TL adapter exists |
| Fallback | `Qwen1.5-7B` — 12 recovered runs already validate this path |
| Generalisation | one second model **only if the primary result is already healthy** |

Qwen3.5/3.6/3.8-27B are **cut**: architecture is identical but not byte-identical,
no `-Base` checkpoints exist at 27B, releases are 6 months apart, so the
post-training comparison has no control. They are also hybrid Gated-DeltaNet +
vision tower with prefix `model.language_model.layers.N` and `head_dim` 256 ≠
5120/24 — head-level analysis would silently break.

## 11. Scope

**Required:** stratification (WP-25); ablation operator; extraction protocol;
positive control; λ-sweep on S2 + harm; random-direction and wrong-layer controls;
DV extractor + ternary validation; the decision rule.

**Optional, only after the paper is complete:** S3 appropriate-hedging control as
a reported arm; second model; wrong-layer beyond one layer.

**Future work:** post-training trajectory; SAE cross-checks; ACE/cone/gradient
methods; conditional steering; bias taxonomy.

**Abandoned:** the eight-way cascade; the selectivity-ratio rule; the byte-fallback
perturbation arm; "soft refusal" as a coinage; Fafuła (2607.17427) as motivation —
its text contains no "soft refusal", no "neutrality", no future-work request, and
it states its task elicits no refusals.

## 12. Stop rule

**May reopen this contract after freeze:**
1. A verified paper that directly preempts the claimed contribution.
2. Failure of the positive control (G1).
3. Invalid measurement — the ternary instrument fails its agreement gate.
4. An implementation error affecting the primary result.
5. Analysis showing the design cannot distinguish the hypotheses at achievable n.

**May not reopen scope:** another interesting steering technique; an extra model;
another benchmark; a new visualisation; more compute becoming available; an agent
proposing a more ambitious paper.

Once the contract survives WP-13, the default is **execute, validate, write.**

### Amendments
*(none — dated entries with reasons go here)*

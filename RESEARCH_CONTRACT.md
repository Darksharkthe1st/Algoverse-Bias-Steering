# RESEARCH_CONTRACT.md — scientific commitments

**Status: FROZEN 2026-08-17 after adversarial review. Scope reduced by that
review — see §0.**
Owner: Edward. Venue: **Interpretability as a Science @ NeurIPS 2026**, Sydney,
long track ≤9pp, non-archival. **Deadline 2026-08-28 AoE. Numbers freeze 2026-08-24.**

This file owns the science. `PROJECT_STATE.md` owns current state;
`WORK_LEDGER.md` owns execution. Neither may redefine anything here. Personal
runbooks are scratchpads and may not redefine the paper, an experiment, a metric,
a deadline, a model set, a rubric, a claim, or a definition of done.

**After freeze, material change requires a dated amendment in §12 with a reason.**

---

## 0. What the freeze review changed — the identification claim is CUT

The adversarial review triggered stop-rule §12.5 (*"analysis showing the design
cannot distinguish the hypotheses at achievable n"*). Three independent findings,
each verified in code:

**0.1 Greedy decoding makes k>1 fictional — so k=1 is forced.** The protocol
mandates greedy decoding for Arditi parity. Under greedy, five generations of one
item are five byte-identical strings. Measured: z_stance median 3.11→4.54 for
sampled k=1→5, but **3.01→3.16 for greedy** — 5× GPU and 5× labelling for zero
variance return. And at k=1 with the 296-item asset, correct-verdict rates are
**0.11 / 0.14 / 0.72 / 0.02** — unusable.

**0.2 Reaching the claim needs 4.7× the asset.** At 80% power, greedy k=1, 2:1
split: distinguishing *nested* (the hardest alternative) needs **n_ben=1400,
n_harm=700 ≈ 10,500 generations**, plus **≥240 audited responses per cell**
(~2,100 human judgements). We have 296 items, ~193 of them S2, and 11 days. The
binding constraint is `n_harm`, not `n_ben` — the angle's precision is set by each
trajectory's *off-target* SE.

**0.3 Every nuisance biases θ upward, toward "distinct".** No nuisance found
biases toward "shared". So the design can *confirm* Joad et al.'s shared-knob
result but cannot credibly *refute* it, and a "distinct control" headline would
rest on a statistic whose only failure direction is the one that produces it.

Two implementation defects were also found and must be fixed regardless, because
they are one afternoon of work: the axis whitening ties θ to the benign/harm
budget split (θ_eq=25° calibrated at 1:1 becomes 33° at 14:1 — inside the bound
meant to exclude nested); and the bootstrap whitens by a different map than the
statistic, giving **0.22 coverage under nested, erring 78% toward falsely
declaring "shared"**. Delete the whitening — the logit invariant makes θ=0 exact
under any linear rescaling, so the null is untouched.

**Consequence.** The λ-ablation machinery, the gates, and the statistic are
**retained and reported**, but the paper's headline is now the measurement
contribution plus a *preregistered bound*, not a mechanism verdict. §5–§7 below
define the machinery; §7.1 defines what we actually claim.

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

### 7.1 What we actually claim — the bounded result

Because of §0, the decision rule above is run and reported, but **only the
equivalence direction is claimable**, and a null is reported as a bound rather
than a finding:

- If `CI_hi < 25°` → **"consistent with the shared non-compliance control Joad et
  al. identify"**, reported with its power.
- Otherwise → **"θ̂ = X° (90% CI [a,b]); at this n we cannot exclude a shared
  control, and we cannot exclude battery-specific leakage that reproduces the same
  signature under a single knob."** State the n required to resolve it: **1400
  benign / 700 harmful, ~10,500 generations, ~2,100 audit judgements.**

**k = 1, forced** (§0.1). **Split 2:1 benign:harm**, not 5:1 — `n_harm` binds.
**Delete the axis whitening** and **match the bootstrap map to the statistic**
before any θ is computed; without both fixes the interval does not cover.

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

**2026-08-17 — A1. Identification claim cut; headline reduced to measurement +
bound.** Reason: stop-rule §12.5. Greedy decoding forces k=1; at k=1 the 296-item
asset yields correct-verdict rates 0.11/0.14/0.72/0.02; reaching the claim needs
n_ben=1400/n_harm=700 plus ~2,100 audit judgements, i.e. 4.7× the asset in 11
days. Every identified nuisance biases θ toward "distinct", so only the
equivalence direction is credible. Machinery retained; headline changed. See §0.

**2026-08-17 — A2. Two statistical defects fixed before any θ is computed.**
Reason: measured coverage 0.22 under nested. Delete axis whitening (ties the
estimand to the budget split); match the bootstrap whitening map to the statistic.

**2026-08-17 — A3. G1 reclassified from "never run" to PASSING.**
**⛔ WITHDRAWN — superseded by A5. Do not cite. Retained for provenance.**
Reason given at the time: audit found
`runs/20260816-011914_refusal-repro_qwen-1.8b` — harmful baseline 38/100 →
ablation **0/100**, ΔP_harm = −0.38 against a −0.15 gate. The positive control was
already reproduced on hardware; earlier plans (mine included) wrongly listed it as
the blocking unknown.

**2026-08-17 — A4. Model revisions pinned to immutable SHAs (post-freeze,
non-scientific).** Reason: the contract requires an immutable revision for a run
to count as evidence, and `docs/PREREG.md` named repos only — so no run could
have satisfied its own provenance rule. `scripts/overnight_validate.py` resolved
them against the HuggingFace API and PREREG §3b now records: primary
`Qwen/Qwen3-8B` @ `b968826d9c46`; watch-only and **cut from this paper**,
`Qwen/Qwen3.5-9B` @ `c20223623576` and `Qwen/Qwen3.8-27B` @ `1d4bf0f2ff60`.

*This amendment changes no hypothesis, statistic, threshold, gate or model set.*
The two watch rows exist so that "a newer checkpoint appeared" is a **recorded
fact rather than a reason to reopen scope**; Qwen3.8-27B is real and was updated
2026-08-14, and it stays cut because there is no `-Base` checkpoint at 27B and
therefore no control (§10). Adding it would need its own amendment.
Tagged `freeze-2026-08-17-a1`. The base freeze `freeze-2026-08-17` → `aed0141`
is unchanged.

**2026-08-17 — A5. A3 is withdrawn: G1 is NOT satisfied.** Reason: A3 tested only
the effect size and read a *mechanism* demonstration as a *replication*. The run
it cites records **"Not reproduced"** in its own findings doc — baseline 0.380
against the paper's 0.700 (Δ −0.32) and extraction cosine **0.90** against a 0.999
target — and it ran on Qwen1.5-1.8B, which is not a submission model. See
`DECISION_LOG.md` D-015, which has governed since the freeze; A3 was left standing
in this file by oversight and is the last text in the repo that disagreed with it.

**G1 stands as written in §6:** on the **submission** model, ΔP_harm ≤ −0.15
**and** extraction cosine ≥ 0.95 **and** un-intervened harmful baseline within
±0.05 of reference. It has never been run. It is the sole current gate.

# PREREG.md — frozen choices

**Frozen 2026-08-17.** Every choice below was fixed **before** any primary
outcome was inspected. Nothing here changes without a dated amendment in
`RESEARCH_CONTRACT.md` §12 and an entry in `DECISION_LOG.md`.

Freeze SHA: recorded in `PROJECT_STATE.md`.

---

## 1. Question

On answerable benign comparative questions, is hedging **consistent with** the
shared non-compliance/refusal control identified in prior work?

## 2. Construct — frozen terminology

| Term | Source | Use |
|---|---|---|
| **hedging** | arXiv:2502.19463 | the behaviour |
| **abstention** | AbstentionBench, arXiv:2506.09038 (Kirichenko, Ibrahim, Chaudhuri, Bell) — *"refuse to answer definitively"* on underspecified / ill-posed / **unanswerable** questions | the failure mode's family |
| **noncompliance taxonomy** | CoCoNot, arXiv:2407.12043 (Brahman et al., NeurIPS 2024 D&B) | the category system |

**We do not coin "soft refusal."**

**Frozen positioning, verified verbatim from the CoCoNot full text:**

> *"Not all the categories in our taxonomy have the potential of having
> contrastive counterparts. We thus create such queries only for 'incomplete
> requests' (specifically, false presuppositions and underspecified queries),
> 'unsupported requests' (specifically, modality limitations), and 'requests with
> safety concerns.'"*

CoCoNot Table 1 contrast-set test counts: Incomplete 148 · Unsupported 82 ·
Safety 149 · **Indeterminate — none · Humanizing — none.**

AbstentionBench targets **unanswerable** questions. Our S2 items are
**answerable**. That is the contrast, and it is the gap.

## 3. Item classification rule — FROZEN BEFORE PRIMARY OUTCOMES

Source: `datasets/GPT_Prompts/comparison_questions_200.csv`, 296 rows, 296/296
parse as `Which <PROP>: <A> or <B>?`.

**Deduplicate first** — 3 exact duplicates removed (*iron/cardboard*,
*flowers/garbage*, *metal/bread*) → **293 unique items**.

| Stratum | Rule | Declining is |
|---|---|---|
| **S1** | Stem contains a first-person preference verb (`do you prefer`, `do you like more`, `do you enjoy more`) — deterministic regex | correct: CoCoNot **Humanizing** |
| **S2** | A privileged answer exists — a competent adult would agree one alternative is correct on the stated property | **a failure**: the contrast set CoCoNot did not build |
| **S3** | No privileged answer — the alternatives are defensible peers on the stated property | correct: CoCoNot **Indeterminate–Subjective** |

**S1 is excluded from all analysis.** S2 is the primary battery. S3 is the
appropriate-hedging control.

**Adjudication, frozen:** three annotators independently label every non-S1 item
S2 vs S3, blind to any model output. Majority wins; **ties → S3** (conservative,
since S3 is where hedging is correct). Labels are committed **before** any λ>0
generation is inspected. Disagreement rate is reported.

**Excluded from the primary DV:** 18 items whose second alternative has no
nameable surface form (*"a seatbelt or no seatbelt"*, *"sunscreen or none"*) —
commitment to side B is not extractable.

## 3b. Model revisions — pinned

Provenance requires an **immutable revision**, not a tag. Resolved from the
HuggingFace API 2026-08-17 by `scripts/overnight_validate.py`:

| Role | Repo | Revision | Upstream lastModified |
|---|---|---|---|
| **Primary** | `Qwen/Qwen3-8B` | `b968826d9c46` | 2025-07-26 |
| Fallback | `Qwen/Qwen1.5-7B-Chat` | resolve at run time | — |
| *Watch only — cut from this paper* | `Qwen/Qwen3.5-9B` | `c20223623576` | 2026-03-02 |
| *Watch only — cut from this paper* | `Qwen/Qwen3.8-27B` | `1d4bf0f2ff60` | 2026-08-14 |

The two watch rows exist so that "a newer checkpoint appeared" is a recorded fact
rather than a reason to reopen scope. Qwen3.8-27B is real and recent; it remains
cut because there is no `-Base` checkpoint at 27B and therefore no control (§10 of
`RESEARCH_CONTRACT.md`). Adding it needs a §12 amendment.

**Every run must record the revision it actually loaded.** A run whose manifest
carries a bare model name is not reproducible and does not count as evidence.

## 4. Intervention

```
x_λ = x − λ (x · r̂) r̂        λ ∈ {0, 0.5, 1}
```

- Directions: `r̂_stance`, `r̂_harm`, `r̂_random` (covariance-matched).
- Projection in **fp32** regardless of model dtype.
- Arditi hook set: `blocks.{l}.hook_resid_pre`, `hook_attn_out`, `hook_mlp_out`,
  all layers.
- **Decoding: greedy ⇒ k = 1.** Sampling would forfeit Arditi parity; under
  greedy, repeated generations are byte-identical, so k>1 buys nothing.

## 5. Primary statistic

For each direction, at each λ, measure `P_stance` (S2) and `P_harm` (harm
battery). Form `Δ_b(r,λ) = logit P_b(r,λ) − logit P_b(r,0)`. Let `v(r)` be the
origin-anchored first principal component over λ.

> **θ = angle( v(r̂_stance), v(r̂_harm) )** · 90% CI by item-level bootstrap,
> resampling items within battery, reusing indices across all λ and directions.

**No axis whitening** — it ties the estimand to the benign/harm split (θ_eq
calibrated at 1:1 becomes 33° at 14:1). The logit invariant makes θ=0 exact under
any linear rescaling, so removal does not touch the null.

**The bootstrap uses the same map as the statistic.** Mismatched maps give
measured coverage 0.22 under *nested*, erring 78% toward falsely declaring
"shared".

**Split: 2:1 benign:harm** — `n_harm` binds, not `n_ben`.

## 6. Threshold — FROZEN BEFORE PILOT OUTCOMES

**θ_eq = 25°.**

From simulation run *before any pilot*: shared-world population θ is 0.1–6.1°
across the full nuisance sweep, ≤10° under mild contamination; oblique 51°;
nested 66°; distinct 90°. θ_eq=15° costs equivalence power (0.75); θ_eq=40°
admits false "shared" for nested. Valid window [20°, 30°].

## 7. Gates — all must pass or no claim is made

| | Gate | Threshold |
|---|---|---|
| **G1** | Positive control **on the submission model** — model-internal, three legs, all required (§7a) | **G1a** `S_split` beats its permutation null *and* ≥ 0.68 · **G1b** ΔP_refuse ≤ −0.15 on held-out `harmful_test` · **G1c** permuted and random controls each \|ΔP\| < 0.05 and ≥ 4 SE below `r̂_harm` |
| **G2** | Precision — each direction moves its **own** target DV | \|Δlogit\|/SE ≥ 4 |
| **G3** | Specificity — covariance-matched random direction | moves both DVs by < 4 SE |
| **G4** | Coherence | no reversal >2 SE against trend where \|ΔP\|>0.05; no sign disagreement when both \|ΔP\|>0.05 |

**G1 is NOT yet satisfied.** `runs/20260816-011914_refusal-repro_qwen-1.8b` shows
the *mechanism* — harmful 38/100 → 0/100 under ablation — but its own findings
doc records **"Not reproduced"**: baseline 0.380 vs the paper's 0.700 (Δ −0.32),
extraction cosine 0.90 vs a 0.999 target. **That run demonstrates the operator
works. It does not satisfy G1, and it is not on a submission model.** It is
retained as historical mechanism evidence (contract §12 A6).

## 7a. G1 — the model-internal positive control · FROZEN BEFORE ANY QWEN3-8B OUTCOME

Fixed by contract §12 **A6**, before any Qwen3-8B activation was inspected.

**Why not the reference cosine.** Arditi ships per-model artifacts for five
models; `Qwen/Qwen3-8B` is not one. Of his 71 files exactly **6 are
model-independent** (the harmful/harmless prompt splits) and those 6 are all G1
needs. "Cosine ≥ 0.95 against reference" is therefore *undefined* on the frozen
primary, not merely hard. We changed the gate rather than the model.

### G1a — the direction is reproducibly estimable

Split the harmful and harmless contrast pools into two disjoint halves each,
preserving the 1:1 balance. Extract a direction from each half by the frozen
recipe (difference in means at the selected cell, fp32, normalised), and take

> **`S_split` = cos( r̂(half A), r̂(half B) )**

**Null — label permutation, B = 500.** Pool the two label groups, re-draw the
harmful/harmless assignment preserving group sizes, recompute `S_split` by the
identical code path. This holds prompt set, layer, token position, sample sizes
and the residual stream's anisotropy fixed and removes only the label signal. A
Gaussian random-direction null would sit near 0 and mistake shared prompt
geometry for signal. **Cost is one extraction pass, not 500** — a direction is a
difference of means over already-cached residuals.

**Pass requires both:**

1. `S_split` > 99th percentile of the null (one-sided permutation test,
   α = 0.01; p reported with the Phipson–Smyth `+1` so it is never 0), **and**
2. `S_split ≥ 0.68`.

**Where 0.68 comes from — derived, not inherited from Arditi's 0.999/0.95.**
Model each half-direction as signal plus independent isotropic noise. Then
E[cos(A,B)] is the signal fraction ρ, each half aligns with the population
direction at ≈ √ρ, and the full pool — twice the data, half the noise variance —
aligns at

> **alignment_full ≈ sqrt( 2·S_split / (1 + S_split) )**

`S_split = 0.68` is where that reaches **0.90**. Isotropy is optimistic (real
residual noise correlates with the signal), so this is an **upper bound** on
direction quality — the conservative way to use one, since it sets a bar to
clear rather than a result to quote.

**Both legs, because either alone is gameable.** A tight null can be cleared by
a direction far too noisy to intervene with; a high cosine can be produced by
shared prompt geometry carrying no label signal at all.

### G1b — full ablation materially reduces refusal, held out

λ=1 directional ablation of `r̂_harm` over the frozen Arditi hook set, scored by
the deterministic `refusal_substring` judge on **`harmful_test`** — a split
touched by neither extraction (`*_train`) nor cell selection (`*_val`).

> **ΔP_refuse ≤ −0.15**

**Regime check, replacing "baseline within ±0.05 of reference":** baseline
refusal **≥ 0.60** on `harmful_test` (the model is refusal-trained and there is
headroom for a −0.15 move) and **≤ 0.10** on `harmless_test` (not pathologically
over-refusing, which would make ΔP_harm easy for the wrong reason). Both are
properties of this model, needing no external number.

### G1c — matched controls do not reproduce the effect

Two controls, each extracted and applied by the identical pipeline:

- **`r̂_perm`** — a direction from one label permutation. Matches data, norm,
  layer, position and geometry; differs only in carrying no label signal. This
  is the stronger control, and it is the one the old gate lacked.
- **`r̂_random`** — covariance-matched random direction (already in §4).

> Each must move refusal by **\|ΔP\| < 0.05**, and `r̂_harm`'s effect must exceed
> each control's by **≥ 4 SE**.

### What passes, what fails

**Pass** = G1a ∧ G1b ∧ G1c. Any single leg failing fails G1, and failing G1
fires stop-rule §12.2 — it does not get retried with a different model until it
passes. A leg-specific failure is diagnostic, and is reported as such: G1a alone
implicates extraction; G1b alone implicates the operator or the regime; G1c
alone means the effect is generic rank-1 damage, which is one of the two worlds
§3 lists as mimicking "shared".

## 8. Inference — deliberately asymmetric

**A small θ, conditional on G1–G4 passing, licenses exactly:**

> *"Consistent with a shared-control model, within the resolution of this
> experiment."*

It does **not** establish a unique shared mechanism.

**A large θ licenses nothing on its own.** Every identified nuisance — weak
direction, incoherence riding on a real knob, ceiling effects — biases θ *upward*,
toward "distinct". None biases toward "shared".

**If the equivalence direction is not reached:**

> *θ̂ = X° (90% CI [a, b]). At this n we can neither exclude a shared control nor
> exclude battery-specific leakage reproducing the same signature under a single
> knob. Resolution requires n_ben = 1400, n_harm = 700 (~10,500 generations) and
> ~2,100 audit judgements.*

## 9. Primary DV and its audit

`named_a_side` — deterministic extractor. Audited on 120 hand-adjudicated
responses drawn from 18,698 archived (prompt, response) pairs on these exact
items: precision **0.967** [0.886, 0.991]; NPV **0.917** [0.819, 0.964]; signed
bias **−0.041**.

**Bias is re-estimated at every λ, not once.** A constant −4pp is tolerable; a
λ-dependent one is fatal, and that is precisely the differential-error question.

**Validity instrument:** one ternary judgement — COMMITTED /
ENGAGED-DID-NOT-COMMIT / UNUSABLE — blind to λ, to direction, and to the
extractor's call, interleaved in one shuffled sheet, two annotators per response
with a third adjudicating. n = 120 minimum.

## 10. Freeze-reopen conditions

Only these reopen the contract:

1. A verified paper that directly preempts the contribution.
2. Failure of G1 on the submission model.
3. The validity instrument fails its agreement gate.
4. An implementation error affecting the primary result.
5. Analysis showing the design cannot distinguish the hypotheses at achievable n.

**Not grounds:** another steering technique · an extra model · another benchmark ·
a new visualisation · more compute · an agent proposing a more ambitious paper.

## Amendments

Amendments are numbered in `RESEARCH_CONTRACT.md` §12. Each entry below is a
pointer; the reason lives there.

**2026-08-17 — A4. §3b added: model revisions pinned to immutable SHAs.** The
only post-freeze change to this document. It fixes a self-inconsistency — this
file demanded an immutable revision as a condition of evidence while naming only
repos — and it changes no hypothesis, statistic, threshold, gate or model set.
State after this amendment is tagged **`freeze-2026-08-17-a1`**; the base freeze
**`freeze-2026-08-17` → `aed0141`** is unchanged and remains the reference for
every pre-outcome choice in §1–§10.

*No other section of this file has changed since the freeze.*

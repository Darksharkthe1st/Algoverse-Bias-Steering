# Pre-registration — Experiment 1, run 2

> ### ⚠ SUPERSEDED IN PART — read `17-reference-paper-and-contrast.md` first
>
> After this file was written, Jeremiah supplied the paper this project is
> modelled on (Joad et al., *There Is More to Refusal in LLMs than a Single
> Direction*). Its method differs from this pre-registration on **the contrast
> itself**, which is the one thing that cannot be treated as a parameter.
>
> **Their contrast is labelled by dataset annotation. Ours was labelled by the
> model's own behaviour.** They obtain within-category floors of **0.95–0.99 from
> 32 items per class**; run 1 obtained −0.204 to 0.880 from 240–600. Sample size
> cannot explain that gap; the contrast can.
>
> Sections **2.1, 2.2, 5, 6, 8, 12.1, 12.2, 12.3** below are amended or void.
> Section 15 at the foot of this file lists every change. Sections 3, 4, 7, 9, 10,
> 11 stand unchanged and remain better than the reference paper's equivalents.

Written 2026-08-23, **before any run-2 data exists.** Planning only; no GPU used.

Every free parameter below is fixed here with a rule and a justification.
Anything not fixed here is a degree of freedom and is declared as such in §12.

**Freeze procedure:** commit this file, record the SHA in every run manifest.
After the freeze, changing anything here means stopping and re-planning, not
patching. See `notes/11-EXPERIMENT-PROTOCOL.md` Gate P3.

---

## 1. Question and primary hypothesis

**Question.** Do different BBQ bias categories yield distinct, reproducible
directions in the residual stream — or is bias one mechanism?

**Primary hypothesis (H1) — AMENDED 2026-08-23.** For at least one BBQ category, a
direction extracted from the **annotation-labelled contrast** (ambiguous vs
disambiguated items, §2.1) reproduces against itself **above that category's own
negative control**, in **at least two model families** — and the cross-category
cosines among reproducing categories sit **below** those within-category floors,
indicating geometrically distinct directions.

The second clause mirrors the reference paper's logic directly: within-category
0.95–0.99 versus cross-category 0.4–0.6 is what licenses their "distinct
directions" claim, and it is the shape of result we are trying to obtain for bias.

> *Original H1, now void: "a direction extracted from the stereotype-margin
> extremes reproduces against itself above that category's own negative control,
> in at least two model families." That contrast is now the third analysis.*

**Falsification condition.** H1 is abandoned if no category clears its negative
control in two or more families. That is a publishable negative given the
positive controls in §7.

---

## 2. Estimators

### 2.1 PRIMARY — AMENDED 2026-08-23 · annotation-labelled difference of means

> **VOID:** the original §2.1 made `extremes` — difference of means over margin
> quintiles — the primary estimator. That is a **behaviour-derived** contrast: it
> ranks items by the model's own stereotype margin. It is the source of defect M1
> and the reason run 1's floors collapsed. Do not use it as the primary.

**The primary contrast is now annotation-derived, exactly as the reference paper
does it:**

```
direction_C = mean(residuals | category C, ambiguous items)
            − mean(residuals | category C, disambiguated items)
```

matched on `question_index`, so both arms share the same question text and the
same three answer options and differ only in whether the context resolves the
referent.

**Why.** Joad et al. contrast prompts where the target behaviour *should* occur
against prompts where it *should not*, using dataset labels. In BBQ, an ambiguous
item has no correct named answer, so any named choice is stereotyping; a
disambiguated item has one, so a named answer is simply correct. `context_condition`
is shipped with the dataset and is completely independent of the model.

**This still has no free hyperparameter**, which was the original reason for
choosing `extremes`, so that argument carries over intact.

**Mandatory control — the specificity check.** The disambiguated arm carries an
extra resolving clause, so the arms differ slightly in context length. Build the
same contrast for every category and compare across categories: if the direction
merely encodes context specificity, all categories collapse onto one direction
(cosine → 1.0); if it is bias-specific, cross-category cosines should land near
the reference paper's 0.4–0.6 band, well below the within-category floor. **Run
this control before anything else.**

**Secondary contrast:** `question_polarity` (neg vs nonneg) at fixed
`question_index` — even more tightly matched, but it measures valence rather than
bias, so it is a comparison and not the headline.

**Third analysis, deliberately retained:** the original margin-quintile contrast,
at matched n, kept in order to demonstrate that a behaviour-derived contrast fails
where an annotation-derived one succeeds. See `17-reference-paper-and-contrast.md`
§5.4.

### 2.2 DEMOTED 2026-08-23 — the ridge probe

> **The probe applies only to the third analysis (§2.1, margin-quintile arm).**
> It exists to squeeze signal out of a *graded* behavioural margin. The primary
> contrast is a binary annotation label with no gradation, so a regression target
> does not exist and the probe is not applicable to it. Everything below stands
> for the third analysis only, and every defect it carries (S3, and the residual
> risk in §12.2) is confined there.

### 2.2 SECONDARY: `probe` — per-layer ridge, alpha by formula

Run 1 chose alpha by sweeping and reading extraction floors, which is the exact
quantity the threshold then gates (S3). **Replaced by a scale-normalising rule
that is fixed before data:**

```
alpha(X) = C · trace(XXᵀ) / n
C = 1.0, fixed for every model, category and split
```

`trace(XXᵀ)/n` is the mean squared norm of the centred residual rows. Dividing by
it makes the penalty dimensionless, so the same `C` means the same shrinkage at
`d_model` 2048 and 5120. **This is data-dependent but not outcome-dependent** —
it reads the scale of X, never the floor, the cosine, or any decision quantity.

`C = 1.0` is the declared default.

**Calibration of `C`, pre-declared here so it is not a post-hoc choice.** Before
any category floor is computed, `C` is selected from `{0.1, 1.0, 10.0}` by running
the **extraction control** — the topic-identity contrast, whose answer is already
known — through the probe on cached residuals, and adopting whichever `C` makes
the *control* reproduce best against its own negative control.

This selects on a control, never on the categories under test, so it is not
selection-on-outcome. It runs on cached residuals in ~20 minutes of CPU at zero
marginal cost, and it is the reason residual caching (§14 of `notes/14`) is a
hard requirement rather than a convenience.

If no `C` in that set makes the control reproduce, the probe is dropped from the
analysis entirely and only `extremes` is reported. That decision rule is fixed
now.

**S3 residual risk, declared:** a direction at one `C` and another `C` may still
disagree (run 1: 0.10–0.21 between alpha 1 and 1e6). The formula fixes *which*
direction we report; it does not make the probe's output alpha-invariant. §12.

---

## 3. The usability criterion — replaces the arbitrary 0.50

**Run 1's 0.50 bar is retired.** It was a constant with a post-hoc justification,
calibrated through one estimator and applied to another (S4).

### 3.1 A category reproduces iff it beats its OWN negative control

For each (model, category, estimator):

1. **Observed floor** — split the category's items in half `B` times, extract a
   direction from each half, record the cosine. Statistic = **mean** of those
   cosines, with a **bootstrap 95% CI** over the splits.
2. **Negative control floor** — repeat identically, but with the **arm labels
   shuffled within that category**, so items are assigned to arms at random while
   topic, vocabulary, prompt format and n are held exactly fixed.
   *For the primary contrast the shuffled label is `context_condition`; for the
   third analysis it is the margin value, as originally written.*
3. **Decision:** the category reproduces iff
   `CI_lower(observed) > CI_upper(negative control)`.

**This single change closes four defects at once:**

- **S4** — the criterion is computed per estimator by construction; nothing is
  borrowed across estimators.
- **N4** — the comparison is against a topic-matched, n-matched alternative
  rather than against noise.
- **S1** — the decision is made on an interval, not a point estimate.
- **N1** — the layer-summary rule cancels to first order, since it is applied
  identically to the observed and control arms.

There is no longer any threshold constant in the pipeline.

### 3.2 Pre-declared rule for a straddling category

If the two CIs **overlap**, the category is **INDETERMINATE**. It is:
- excluded from the primary clustering analysis;
- reported by name in the results table with both CIs;
- included in a pre-declared sensitivity analysis that reruns clustering with all
  INDETERMINATE categories added.

This rule is fixed now, before it is known which categories straddle. Run 1's
Nationality (floor 0.5103, gate 0.50, 26.6% bootstrap failure) is exactly the
case it exists for.

---

## 4. Number of splits — calculated, not defaulted

Run 1 used `n_splits = 10` and reported a `quantile(0.05)`, which with 10 points
interpolates to a blend of the two worst draws (S1).

**Measured from run 1's own artifacts**, across 48 (run, category) pairs, the SD
of the ten split-half cosines is: median **0.1377**, mean 0.1335, 90th percentile
**0.2023**, max 0.2417.

Using the 90th percentile as the design value:

| target 95% CI half-width on the mean | required n_splits |
|---|---|
| 0.020 | **393** |
| 0.010 | 1,572 |
| 0.005 | 6,287 |

**Decision: `n_splits = 400`**, giving a 95% CI half-width of **±0.020** on the
mean split-half cosine at the 90th-percentile variance, and better than that for
a typical category.

**Run 1 used 10 where ~400 is needed — a 40× shortfall.** This is affordable only
because residuals are cached (§8): every split is CPU arithmetic on a stored
array, not a forward pass.

**Statistic changed from q05 to the mean**, because a quantile over B draws has
materially larger error than the mean and run 1 combined the worst of both.

---

## 5. Sampling, matched across categories

Run 1's n varied 172–320 within a model and correlated with the floor at +0.505
on qwen-1.8b (M3).

> **AMENDED 2026-08-23.** The n=600 figure was sized for ranking items by margin
> and taking quintiles. The primary contrast no longer ranks anything — it uses
> **every matched item in each arm**. Revised sampling:
>
> - **PRIMARY: all matched ambiguous/disambiguated pairs per category**, paired on
>   `question_index`. Declared **minimum 32 per arm**, following Arditi et al. and
>   the reference paper, which achieves floors of 0.95–0.99 at exactly that n.
>   Report the achieved n per category rather than truncating to a common figure.
> - **Balance is enforced per arm**, not across categories: an unequal number of
>   ambiguous and disambiguated items in a category is subsampled to the smaller.
> - The matched-n concern behind M3 is handled by the pairing itself, since every
>   item in one arm has a partner in the other.
>
> The figures below apply to the **third analysis** (margin-quintile arm) only,
> where ranking still happens and matched n still has to be imposed by hand.

- **PRIMARY: n = 600 per category, 9 categories** (all but Sexual_orientation,
  which has only 432 scoreable ambiguous items).
- **SECONDARY: n = 430 per category, all 10 categories.**

Declaring both in advance, with the primary named, prevents choosing after seeing
which is friendlier. The secondary tests whether excluding one category changes
the conclusion.

Sampling is `SampleSpec(filter={"context_condition": ["ambig"]}, limit=n, seed=S)`.

**Quintile = 0.20**, unchanged from run 1 and fixed here. At n=600 that is 120
items per pole. Not tuned.

---

## 6. Abstention — a pre-declared rule (closes N3)

> **SCOPE NARROWED 2026-08-23.** This entire section is a rule about *pole
> assignment by margin*. The primary contrast assigns arms by
> `context_condition`, not by margin, so no item is ever placed in a pole by a
> quantity the model produced — and the rule does not apply to it.
>
> It remains **mandatory for the third analysis** (§2.1, margin-quintile arm),
> where the defect it closes is real and measured: 23.5% of qwen-14b items had
> `logP(unknown)` above both named options.
>
> The abstention fraction is still **reported descriptively for every category in
> every analysis**, because it characterises the model's behaviour and costs
> nothing once the log-probs are stored.

Measured on run 1's cached margins: across 5,830 qwen-14b items, **1,369 (23.5%)**
have `logP(unknown)` exceeding **both** named options. Run 1 recorded this and
never used it — those items were ranked by a stereotype margin between two
options the model did not prefer, and could enter either pole.

**Rule.** An item is **eligible for pole assignment** only if
`max(logP(biased), logP(other)) > logP(unknown)` — that is, only if the model
actually prefers a named person over abstaining.

- Ineligible items are **scored and reported** (count and fraction per category)
  but excluded from pole assignment.
- The `n` in §5 is counted **after** this filter, so matching still holds.
- **Sensitivity, pre-declared:** the whole primary analysis is repeated including
  ineligible items, and both results are reported.

If the filter leaves a category with fewer than the matched `n`, that category is
excluded and named, under the §3.2 exclusion machinery.

---

## 7. Controls — all four, all pre-registered

| control | question | pass condition | closes |
|---|---|---|---|
| **Task control** | can the model pick the right person when the context says? | accuracy ≥ 0.50 **and** z ≥ 3.0 on disambiguated items | — |
| **Extraction control** | can this estimator recover a direction we know exists? | topic-identity contrast (race prompts vs gender prompts) beats its own negative control, **run separately for each estimator** | S4 |
| **Negative control** | does a category beat a topic-matched, n-matched shuffle? | §3.1 | N4, S1 |
| **Matched-n estimator control** | does the probe beat extremes at equal data? | probe fit on **only the 2×120 pole items**, not all 600 | **S2** |

The matched-n control is the one run 1 lacked entirely. Until it exists, no claim
of the form "the probe is the better estimator" may be made — run 1's probe saw
2.50× the data (verified: 240 vs 600 in every category).

---

## 8. Capture site and layer handling

- **Site:** residual stream at the **final prompt token**, before any answer token
  exists. Prevents the direction encoding the output it was labelled from.
- **AMENDED 2026-08-23 — align with the reference paper.** Capture at the
  **chat-template token immediately preceding the assistant's response
  (index −2)**, hook `resid_pre`, which Joad et al. call "the model's decision
  state." Confirm this is the same position our loader already used; if it is not,
  the reference paper's position wins, because comparability with it is worth more
  than continuity with run 1.
- **Primary layer read:** a **fixed mid-layer**, as they do (layer 20 for
  gemma-2-9b-it, 15–16 for Llama-3.1-8B). We still capture and store **all**
  layers — that is a superset and costs nothing — but the headline number is
  reported at the mid-layer for direct comparability, with the all-layer summary
  reported alongside.
- **Layers:** **all** layers captured and stored. **No layer selection, ever** —
  not before, not after. Run 1 did this correctly; it is fixed here so it stays.
- **Layer summary rule, fixed in advance:** **norm-weighted mean** of per-layer
  cosines, weights = the L2 norm of the reference direction at each layer.
  Justified because per-layer norms span orders of magnitude, so an unweighted
  median treats a near-zero-norm layer as equal to the highest-signal one.
  Run 1's unweighted median is reported alongside as a **sensitivity** — measured
  difference in run 1 was ≤0.033, so this is expected to be immaterial and is
  declared so that it cannot become a choice later (N1).
- Layer 0 is exactly zero for every direction (the capture token is the chat
  template's final token, identical across items) and propagates as NaN. It is
  excluded by the finite filter. Documented, not a defect.

---

## 9. Split construction (closes N5)

Splits are **stratified by pole**: each half receives exactly half of the top-pole
items and half of the bottom-pole items. Run 1 shuffled the pooled list and cut
blind, admitting ±11 pole imbalance at 120+120, which added variance to the floor
unrelated to reproducibility.

---

## 10. The permutation null

- **What is shuffled:** margin values **within each category**, holding category
  membership, topic, n and prompt format fixed. Run 1 shuffled items *across*
  categories, so fake groups were topic-heterogeneous while real ones were
  topic-homogeneous (N4).
- **Count:** 1,000 permutations (run 1 used 200).
- **Statistic:** `cluster_strength` = max gap between consecutive merge heights,
  unchanged — **but now reported with a bootstrap CI over splits** (N2). Run 1
  reported it as a point value; on 5 leaves it is the max of 3 differences and its
  error was never estimated.
- **p:** `(r + 1) / (n + 1)`, unchanged.

---

## 11. Models, seeds, and analysis structure

**Models.** qwen-1.8b, qwen-7b, qwen-14b, yi-6b, gemma-2b — the five that ran in
run 1, all ungated, all verified loadable under TransformerLens on an A100-40GB.
**llama3-8b is excluded** unless HF access is confirmed *before* rental; run 1
discovered the 403 only after paying for GPU time.

**Seeds.** Sampling seed `S = 0`. Split seeds `0..399`. Permutation seeds
`0..999`. Fixed here, recorded in every manifest.

**Exactly one primary analysis:** §3.1 applied with the `extremes` estimator at
n=600 over 9 categories. Everything else — the probe, the n=430 arm, the
abstention sensitivity, the layer-summary sensitivity, clustering — is
**secondary** and labelled as such in every table.

---

## 12. Declared degrees of freedom — not closable by this design

Honesty requires naming what remains open.

> ### ⚠ §12.1 (M1) IS CLOSED AS OF 2026-08-23 — do not carry it into the paper
>
> M1 said the floor is confounded with behavioural tilt, and that this is *not
> closable by this design*. That was true of the run-1 design and **false of the
> design we have now adopted.**
>
> M1 exists only because directions were built by contrasting the extremes of a
> **behaviour-derived** margin, which makes "no tilt" and "no representation"
> indistinguishable. The primary contrast is now labelled by
> `context_condition` — a dataset annotation. A category where the model shows
> zero behavioural tilt can still yield a direction, because tilt was never
> consulted.
>
> **The drafted limitation sentence below must be removed from the paper.**
> It survives only as a limitation of the third analysis (margin-quintile arm),
> where it is now a deliberately demonstrated contrast rather than an admission.
>
> The same reclassification applies to **§12.2 (probe alpha)** and **§12.3 (heavy
> tails)**: both are properties of estimators that only the third analysis uses.
> Kurtosis mattered because the quintile contrast selected distribution tails; the
> annotation contrast uses every item in each arm.
>
> The original text is retained below unaltered, for the record.

1. **M1 — floor is confounded with behavioural tilt.** Directions are built by
   contrasting margin extremes, so a category with no tilt is split on noise and
   cannot reproduce whether or not a representation exists. Confirmed at +0.660
   to +0.769 across all five models. The negative control in §3.1 narrows this —
   it holds tilt fixed and shuffles only the labels — but it cannot separate
   "no representation" from "no contrast with which to find one."
   **Limitation sentence, drafted now:** *"Our procedure detects a bias direction
   only where the model exhibits a systematic behavioural tilt on that category;
   categories without such a tilt are untestable by this method rather than shown
   to lack a representation."*
2. **S3 residual — the probe's output still depends on `C`.** The formula fixes
   which direction is reported; it does not make the probe alpha-invariant.
   Reported as a limitation of every secondary probe result.
3. **M2 — heavy tails.** Physical_appearance and Age have excess kurtosis +3.9
   with the top 5% of items carrying ~half the variance, and the extremes contrast
   selects exactly those tails. **Mitigation:** report each with and without
   winsorising the margin at the 2.5/97.5 percentiles, both pre-declared.
   Disability_status (kurtosis −0.9) is the cleanest positive and should carry the
   headline.
4. **Single dataset.** All conclusions are BBQ conclusions. No claim generalises
   beyond it.

---

## 13. What is recorded for every run

Non-negotiable, and directly from the user's requirement for run 2:

- **Every prompt, verbatim.** The exact string scored, per item.
- **Every generated response, verbatim, character for character**, from a
  dedicated generation pass — even though the primary statistic is a likelihood.
  Run 1 could not answer basic questions about model behaviour because no
  completion was ever persisted.
- Every per-option log-probability, not just the derived margin.
- Every residual tensor used in any extraction (§8, and see `notes/14`).
- Estimator, `C`, seeds, n, quintile, layer rule, code SHA, per-category
  eligibility and exclusion reasons.

---

## 14. Summary of every fixed parameter

> **The table below is the ORIGINAL 2026-08-23 version. Rows marked ✗ are void.**
> The authoritative table is §15, immediately after it.

| parameter | value | fixed by |
|---|---|---|
| primary estimator ✗ | `extremes` | no free parameter |
| secondary estimator | ridge probe | `alpha = C · trace(XXᵀ)/n`, `C` calibrated on the extraction control from `{0.1, 1, 10}` (§2.2) |
| usability criterion | beats own negative control, CI-disjoint | §3.1 — no constant |
| n_splits | **400** | calculation, ±0.020 CI (§4) |
| floor statistic | **mean** + bootstrap 95% CI | §4 |
| straddle rule | INDETERMINATE, excluded, sensitivity both ways | §3.2 |
| n per category | **600 primary (9 cats)** / 430 secondary (10 cats) | §5 |
| quintile | 0.20 | unchanged from run 1, declared |
| abstention | excluded from poles; sensitivity includes | §6 |
| capture site | final prompt token, all layers | §8 |
| layer summary | norm-weighted mean; median as sensitivity | §8 |
| split construction | stratified by pole | §9 |
| null | within-category margin shuffle, 1,000 perms | §10 |
| seeds | S=0; splits 0–399; perms 0–999 | §11 |
| models | 5 ungated; llama3 only if access verified pre-rental | §11 |
| primary analysis | exactly one (§11) | §11 |

---

## 15. AUTHORITATIVE PARAMETER TABLE — amended 2026-08-23

Supersedes §14 wherever the two disagree. Rationale in
`17-reference-paper-and-contrast.md`.

| parameter | value | changed? |
|---|---|---|
| **primary contrast** | `mean(resid \| ambig) − mean(resid \| disambig)`, matched on `question_index` | **CHANGED** — was margin quintiles |
| **label source** | `context_condition`, shipped with BBQ | **CHANGED** — was the model's own margin |
| secondary contrast | `question_polarity` neg vs nonneg, matched on `question_index` | new |
| third analysis | margin quintiles, retained to demonstrate it fails | new framing |
| mandatory first control | cross-category cosine on the primary contrast — catches "context specificity" | new |
| primary estimator | difference of means | unchanged in kind; no free parameter |
| ridge probe | third analysis only | **DEMOTED** |
| usability criterion | beats own negative control, CIs disjoint | unchanged |
| n_splits | 400, bootstrap 95% CI | unchanged |
| floor statistic | mean + bootstrap CI | unchanged |
| straddle rule | INDETERMINATE, excluded, sensitivity both ways | unchanged |
| **n per category** | all matched pairs; **minimum 32 per arm** | **CHANGED** — was 600 by margin rank |
| quintile | 0.20 | third analysis only |
| abstention rule | third analysis only; reported descriptively everywhere | **NARROWED** |
| **capture position** | chat token **index −2**, `resid_pre` | **CHANGED** — align with reference paper |
| **primary layer** | fixed mid-layer (gemma-2-9b-it: 20; Llama-3.1-8B: 15–16); all layers still stored | **CHANGED** |
| layer summary | norm-weighted mean; median as sensitivity | unchanged |
| split construction | stratified **by arm** | amended from "by pole" |
| null | shuffle **arm labels** within category, 1,000 perms | amended from "margin labels" |
| seeds | S=0; splits 0–399; perms 0–999 | unchanged |
| models | 5 ungated, **plus `google/gemma-2-9b-it`** if the SAE stage is in scope | **AMENDED** |
| steering | `x' = x + α·r` (r unit-normalised); ablation `x' = x − (x·r)r`; α grid `{5,10,20,30,60}` | **NEW STAGE** |
| controlled test set | balanced across item type × unsteered response, so baseline = 50% by construction | **NEW** |
| SAE stage | GemmaScope layers 9/20/31; firing-rate separation; controls = random latent subsets + random unit vectors | **NEW, stretch scope** |
| primary analysis | exactly one — the annotation contrast at the mid-layer | unchanged in principle |

### Defect status after the amendment

| id | was | now |
|---|---|---|
| M1 tilt confound | not closable | **CLOSED** — labels are annotations |
| S3 probe alpha | mitigated | **moot for primary** — no alpha in the primary |
| M2 heavy tails | mitigated | **moot for primary** — no tail selection |
| N3 abstention | closed | **narrowed** — third analysis only |
| M3 varying n | closed | **closed** — by pairing |
| S1, S2, S4, S5, N1, N2, N4, N5 | closed | closed, unchanged |

**Every defect from run 1 is now either closed or confined to the third analysis,
which exists precisely to exhibit them.**

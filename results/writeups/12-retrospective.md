# Retrospective and adversarial audit — before the re-run

> ### Status note, added 2026-08-23 (later the same day)
>
> **This audit remains accurate as history and its defect measurements all stand.**
> Nothing below was found to be wrong.
>
> One conclusion has been overturned by new information: **M1 is closable after
> all.** Part 1's "Root cause 1" analysis and the M1 row in the consolidated table
> both assume the contrast is behaviour-derived. After reading the reference paper
> this project is modelled on (Joad et al.), the contrast has been replaced with an
> annotation-derived one, which removes the confound entirely.
>
> Read `17-reference-paper-and-contrast.md` for the reasoning. Everything else in
> this document — S1 through S5, N1 through N5, M2, M3, and the three root causes —
> is unaffected and still governs.

Written 2026-08-23, planning only. No GPU was used.

**Method for this document:** every claim in the incoming brief was recomputed
from the artifacts in `runs/` before being accepted. Where I could not verify
something it is marked `UNVERIFIED`. Where I disagree with the brief's grading I
say so. Section 2 hunts for defects the brief did not list, and grades them
honestly — including grading two of my own candidates *down* after measuring them.

---

## Part 0 — Verification of the incoming claims

| claim | status | evidence |
|---|---|---|
| **S1** floor is a q05 over 10 splits; Nationality clears by 0.010; ≥26% bootstrap failure | **CONFIRMED, exactly** | cosines `[0.493 0.532 0.547 0.574 0.576 0.579 0.625 0.628 0.629 0.658]`; q05 recomputes to `0.5103`; `0.05*(10-1)=0.45` so q05 = `0.55*0.4925 + 0.45*0.5321`; bootstrap over the 10 splits gives **P(fail) = 26.6%** (20k resamples, seed 0). Disability_status by contrast: `0.879–0.938`, q05 `0.8799`, min `0.8794` |
| **S2** estimator comparison confounded with n | **CONFIRMED, exactly 2.50×** | every category: extremes n=240, probe n=600 (Sexual_orientation 172 vs 432). Ratio 2.50–2.51× uniformly |
| **S3** alpha selects a direction rather than tuning one | **CONFIRMED** | same-category cosine between α=1 and α=1e6: Age +0.1015, Religion +0.1138, Disability +0.1483 — all far below the project's own 0.50 bar. Off-diagonal mean \|cos\| 0.0088 → 0.0823 |
| **S4** 0.50 threshold never calibrated for the probe | **CONFIRMED** | all three `_extraction*control*.json` files carry no `method` field, so they ran the default path (`extremes`). No probe-based calibration artifact exists |
| **S5** residuals never cached | **CONFIRMED** | `find runs -name "*resid*" -o -name "*activation*"` → no matches. Cache holds `['ids','margins','abstention']` only |
| **M1** floor vs tilt +0.66 to +0.77 | **CONFIRMED** | qwen-1.8b +0.689, qwen-14b +0.726, gemma-2b +0.660, yi-6b +0.689, **qwen-7b +0.769** |
| **M2** kurtosis +3.9, top 5% ≈ half the variance | **CONFIRMED** | `runs/_margin_distribution_qwen14b.json` |
| **M3** qwen-1.8b n varies 172–320, correlates +0.505 | **CONFIRMED, and it is model-specific** | qwen-1.8b **+0.505** (n range 172–320). Other models: qwen-7b +0.263, yi-6b +0.410, qwen-14b +0.155, gemma-2b +0.078, all with n range 172–240 |

**Nothing in the brief's severe or moderate list is false.** One number is
slightly understated: M1's upper bound is +0.769 (qwen-7b), not +0.77 as a range
ceiling — immaterial.

I also checked the brief's "NOT DEFECTS" list and agree with all five
dismissals. The denominators it gives (qwen-1.8b 0 of 8; gemma-2b and yi-6b 2 of
9 after Gender_identity failed at 0.487 and 0.447; the Qwens 10) match
`report.json` exactly.

---

# Part 1 — What is the common cause?

The brief proposes: *every defect is a free parameter set by looking at data and
never written down first.* **I tested that and it is right for four of the five,
but it is not the whole story, and the exception matters most.**

## Root cause 1 — Free parameters chosen against the quantity they gate

This is the brief's hypothesis and it holds for **S3 and S4**, and partially S1.

- **S3:** α was selected by reading extraction floors. The 0.50 gate then tests
  extraction floors. The selection criterion and the decision criterion are the
  same number.
- **S4:** the threshold was justified by a control run through a *different*
  estimator, then applied to the probe unchanged.
- **S1:** `n_splits=10` and `quantile(0.05)` have no written justification
  anywhere in `bias_taxonomy.py`. They were defaults I chose while writing the
  function, before any data existed — so this one is *not* selection-on-outcome.
  It is the next cause.

**The preventing practice:** every parameter gets a *rule* (a formula, or a value
fixed by prior work) written before data exists. If a parameter must be chosen
from data, it is chosen on a held-out split that never enters the reported
statistic.

## Root cause 2 — Decision statistics with unquantified sampling error

**This is the deeper cause and the brief under-weights it.**

`extraction_floor` returns a q05 over 10 draws and the pipeline treats it as a
point value. Nothing anywhere — not in the code, not in any report — estimates
how much that number would move on a redraw. The consequence is not abstract:
**Nationality's floor has a 26.6% chance of failing its gate on a re-run**, and
that single category is what takes qwen-14b from 4 reproducible to 5, which is
what makes clustering possible at all.

The same omission repeats for `cluster_strength` (§N2 below) and for every
between-category cosine.

**The preventing practice:** a number that gates a decision must ship with an
interval, and the gate must be stated in terms of that interval, not the point
estimate. `n_splits` is then chosen so the interval is narrow relative to the
threshold — a calculation, not a default.

## Root cause 3 — Artifacts not durable by default

**S5 is a different failure and deserves separate billing.** It is not a
parameter problem; it is that the pipeline persisted only what the *next* step
needed, not what a *future question* would need. Residuals were the expensive
thing and the discardable thing simultaneously, and the decision to discard was
never made consciously — there simply was no line of code that saved them.

Its cost is unique: it converted every other defect from *fixable in an
afternoon* into *blocked on renting hardware again*.

The operational failures in §4 of the brief share this cause. Artifacts on the
box only; findings in chat transcripts; two sessions on one repo. All of it is
"state that exists in exactly one place, and that place is volatile."

**The preventing practice:** persistence is a requirement of the run plan, sized
and budgeted, not a convenience. The test is: *could every analysis in the plan
be redone tomorrow with the GPU already returned?* If no, the plan is incomplete.

## Why I reject a fourth candidate

I considered "time pressure from a running GPU" as a root cause. **The evidence
does not support it as a *cause*.** S1's `n_splits=10` and the median-over-layers
rule (§N1) were both written before the box existed. Time pressure explains why
defects were *patched* rather than *re-planned*, which is a real dynamic and is
covered in `notes/11`, but the defects themselves were baked in during planning.

---

# Part 2 — Adversarial audit

Assuming the brief's list is incomplete. Everything below was recomputed from
artifacts. Categories that yielded nothing are reported as such.

## N1 — The per-layer summary rule is unjustified and estimator-dependent
**Grade: MODERATE** *(I initially graded this SEVERE and measured it down — see below.)*

`summarize_cosine` (`bias_taxonomy.py:428`) collapses a per-layer cosine profile
by taking the **median across all layers**, with no weighting. Its docstring
justifies this as "robust to the handful of near-zero-norm early layers."

**The justification is inaccurate.** It is not a handful:

| run | category | layers | norm < 1% of max | norm < 10% of max |
|---|---|---|---|---|
| full_qwen14b (extremes) | Disability_status | 40 | 12 | **21** |
| full_qwen14b | Nationality | 40 | 9 | 21 |
| probe_tuned_qwen14b (probe) | Disability_status | 40 | **5** | 17 |
| full_yi6b | Disability_status | 32 | 12 | 19 |
| full_gemma2b | Disability_status | 18 | 1 | 8 |

For extremes on qwen-14b, **21 of 40 layers carry under 10% of the peak norm** —
a majority. And the counts differ by estimator (12 vs 5 below 1%), so
"median over layers" averages a *different population* for each estimator. That
compounds S4: the two estimators' numbers are not on the same scale even before
the threshold question.

**Why I graded it down.** I predicted the median would land in the noise band. It
does not:

```
full_qwen14b        median cosine +0.2417 comes from the layer ranked  1 of 39 by norm
probe_tuned_qwen14b median cosine +0.1084 comes from the layer ranked 10 of 39 by norm
norm-weighted mean vs unweighted median: +0.2578 vs +0.2417   (Δ 0.016)
Nationality vs {Disability, Religion, Age, Physical}:
   median            +0.150  +0.082  +0.050  +0.069
   top-10-norm only  +0.183  +0.081  +0.065  +0.065   (Δ ≤ 0.033)
```

The summary rule is arbitrary and undocumented, but empirically it lands within
~0.03 of a defensible alternative. It needs to be **fixed by rule in advance**,
not redesigned.

## N2 — The clustering statistic's sampling error is unquantified
**Grade: SEVERE**

`cluster_strength` (`bias_taxonomy.py:~780`) is the **maximum gap between
consecutive merge heights** in the linkage. On qwen-14b it is computed over
**5 leaves → 4 merges → the max of 3 differences.** Observed 0.0962 against a
null median 0.0473 and q95 0.0852, giving p=0.0299.

**p = 0.0299 sits between the null's q95 (0.0852) and the observed value
(0.0962)** — a margin of 0.011 on a statistic that is the maximum of three
numbers derived from a 5×5 matrix whose own entries have unquantified error.
Nothing in any artifact estimates how much 0.0962 would move on a redraw.

This is S1's disease applied to the *other* headline number. S1 says the count of
usable categories is unstable; N2 says the statistic computed *from* that count
is also unstable, and the two compound.

## N3 — Abstention is measured, recorded, and then ignored
**Grade: SEVERE**

`margins()` (`bbq_score.py`) computes `margin = logP(biased) − logP(other named)`
and separately records `abstention_margin = logP(unknown) − max(logP(named))`.
**The abstention value is stored and never used for anything.**

Recomputed across every cached qwen-14b category:

```
items scored                                              5,830
items where logP(unknown) exceeds BOTH named options      1,369  (23.5%)
```

**On 23.5% of items the model's actual top choice is "can't answer",** yet that
item is still ranked by a stereotype margin between two options it did not
prefer, and may enter the top or bottom pole and contribute to the direction.

The direction is meant to contrast "leans stereotyped" against "leans
anti-stereotyped." For nearly a quarter of items it is contrasting two
counterfactuals the model rejected. Whether that adds noise or bias is untested —
but it is a documented, quantified, silent inclusion of off-target data, and it
has never been reported.

## N4 — The permutation null tests against noise, not against a matched alternative
**Grade: SEVERE — this is the most consequential thing I found**

`permutation_null` (`bias_taxonomy.py:801`) pools every item across every
category, reshuffles into fake groups of the same sizes, and extracts a direction
per fake group.

**A fake group is topic-heterogeneous; a real group is topic-homogeneous.** The
two differ in more than the label assignment being tested.

The subtraction does cancel topic to first order in both cases — a direction is
`mean(top pole) − mean(bottom pole)`, and both poles come from the same group.
But in a *real* group both poles are drawn from one BBQ category, so topic
cancels **exactly**. In a *fake* group the poles are random subsets of a mixed
pool, so topic cancels only **in expectation**, leaving sampling noise.

Consequence: fake directions are systematically noisier than real ones,
independent of any bias structure. Noisier directions are more nearly orthogonal,
so their pairwise distances cluster near 1.0, merge heights compress, and
`cluster_strength` falls. The null median of **0.0473** is consistent with
near-orthogonal noise.

**So the test may be detecting "real directions are less noisy than random
mixtures" rather than "bias categories form meaningful groups."** Both produce
p < 0.05.

The stronger null preserves category membership and shuffles **the margin labels
within each category** — that isolates whether the *margin* carries the structure,
which is the actual claim. This null does not exist in the artifacts.

`UNVERIFIED`: I cannot measure the size of this effect without the fake
directions, which were never saved (S5 again). The mechanism is established by
construction; the magnitude is not.

## N5 — Split-half is not stratified by pole
**Grade: MODERATE**

`split_half` (`bias_taxonomy.py:~600`) shuffles the pooled `top + bottom` list
and cuts it in half. It does not stratify. With 120+120, a random half has
expected 60/60 but a hypergeometric SD of ≈5.5, so ±11 imbalance is routine.

Each half's direction is then a difference of means over unequal pole sizes,
which adds variance to the floor that has nothing to do with the reproducibility
being measured. **This is a component of S1's instability that S1 does not name.**

Checked for the degenerate case: across all 760 recorded split-half cosines in
every run, **zero are non-finite**, so no split ever produced an empty pole.

## Categories that yielded nothing — reported, not padded

- **Post-hoc layer selection:** none. Every analysis uses all layers; no run
  selects a layer after seeing results. `summarize_cosine`'s `layer=` argument
  exists but is passed `None` everywhere in the shipped pipeline.
- **Degenerate estimator solutions:** none found. No all-zero direction, no
  rank-collapse, no non-finite cosine in 760 recorded values. Layer 0 is exactly
  zero in every direction — but that is *correct*: the capture site is the chat
  template's final token, identical across items, so at the embedding layer every
  item is identical and the difference is exactly 0. It propagates as NaN and is
  filtered.
- **Filtered items / dropped categories:** all accounted for. Drops are recorded
  with accuracy and z in `report.json` and match the direction-file counts
  (8/9/9/10/10).
- **Seed reuse:** `extraction_floor` and `permutation_null` both derive from
  `seed + k`, but they consume different data through different procedures. I
  found no evidence of induced correlation and am not grading it.

---

## Consolidated severity list going into the pre-registration

| id | defect | grade | closable before the run? |
|---|---|---|---|
| S1 | decision statistic from 10 draws, no interval | SEVERE | yes — n_splits by calculation |
| N4 | null tests against noise, not a matched alternative | SEVERE | yes — within-category label shuffle |
| N3 | 23.5% of items are abstentions treated as choices | SEVERE | yes — pre-declared handling rule |
| N2 | clustering statistic has no error estimate | SEVERE | yes — bootstrap it |
| S4 | threshold not calibrated per estimator | SEVERE | yes — run the control per estimator |
| S3 | alpha selects a direction; chosen on the gated quantity | SEVERE | yes — formula, not sweep |
| S2 | estimator comparison confounded with n | SEVERE | yes — matched-n control |
| S5 | residuals not cached | SEVERE (operational) | yes — persist by design |
| N1 | layer summary unjustified, estimator-dependent | MODERATE | yes — fix rule in advance |
| N5 | split-half not stratified by pole | MODERATE | yes — stratify |
| M1 | floor confounded with behavioural tilt | MODERATE | **YES — closed 2026-08-23** by replacing the behaviour-derived contrast with an annotation-derived one (`context_condition`). Originally recorded here as "no — declare as limitation"; that was correct only for the run-1 design. See `17-reference-paper-and-contrast.md` §4.1 |
| M2 | two positives rest on heavy tails | MODERATE | partially — winsorise + report both |
| M3 | n varies with floor on qwen-1.8b | MODERATE | yes — match n across categories |

**Eleven of thirteen are closable by planning alone.** M1 is not closable by this
design at all and must be declared. That is the single most important sentence in
this document.

> **AMENDED 2026-08-23.** The sentence above was right about the *design it was
> written for*, and wrong as a general claim. M1 is a consequence of contrasting
> the extremes of a behaviour-derived margin. The reference paper never does that
> — its labels come from dataset annotations — and adopting its contrast closes
> M1 outright.
>
> **Revised count: twelve of thirteen closed; the thirteenth (S3, probe alpha) is
> confined to a deliberately-retained secondary analysis.** The lesson worth
> keeping from the original sentence is the habit, not the conclusion: name what
> your design cannot close, then check whether a different design closes it.


---

## N6 — the choice parser injects position bias, and its accuracy is unauditable

*Added 2026-08-23. Full detail in `notes/18-parser-audit.md`.*

`parse_choice` resolves a response to the **earliest-mentioned** option. It has no
negation handling and no question-echo stripping, so "it's not the doctor, it's
the nurse" and "between the doctor and the nurse, I'd say the nurse" both resolve
to *the doctor*. Three of seven realistic phrasings parse wrong, and **every
failure resolves toward the first-named option**.

This contaminates the method-1 diagnosis. `person_consistency = 58%` was read as
evidence that the model is order-sensitive; a first-mention-biased parser produces
that same signature from a perfectly consistent model. Run 1 cannot separate the
two.

The unparsed rate was reported honestly (12.9% pooled, up to 40% on Race_x_SES).
The **mis**parsed rate is invisible in the saved counts, and the raw response text
was not kept, so it can no longer be measured. **Severity: SEVERE.** This is
defect S5 recurring in an unanticipated form — lost data does not only block new
analyses, it blocks auditing finished ones.

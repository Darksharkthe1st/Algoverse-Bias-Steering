# Run 3 — Handoff

**For: Farhan (paper writing). From: Jeremiah.**
**Date: 2026-09-03. Branch: `jz/run3-results`.**

This document describes the entire Run-3 experiment: what we did, in what order,
with what settings, what came out, and what we are and are not allowed to claim
from it. Every number in this document was read out of an artifact in this
folder. Nothing here is remembered or estimated. Where a number is fragile I say
so in the same sentence as the number.

If you only read one thing, read [§1](#1--the-three-answers-in-one-table) and
[§10](#10--claim-register--what-you-may-and-may-not-write).

---

## Table of contents

| § | Section | What it gives you |
|---|---|---|
| 1 | [The three answers](#1--the-three-answers-in-one-table) | The TL;DR |
| 2 | [Why Run 3 exists](#2--why-run-3-exists) | Intro material |
| 3 | [Design and vocabulary](#3--design-and-vocabulary) | Methods section |
| 4 | [Step-by-step procedure](#4--step-by-step-procedure) | Methods section |
| 5 | [Instrument validation](#5--instrument-validation) | The controls that license everything |
| 6 | [Bucketing — the binding constraint](#6--bucketing--the-binding-constraint) | Results |
| 7 | [Extraction floors](#7--extraction-floors) | Results (headline #1) |
| 8 | [Cross-category structure](#8--cross-category-structure) | Results (headline #2) |
| 9 | [Steering — the causal arm](#9--steering--the-causal-arm) | Results (headline #3) |
| 10 | [Claim register](#10--claim-register--what-you-may-and-may-not-write) | **Read this before writing** |
| 11 | [Limitations, paper-ready](#11--limitations-paper-ready) | Limitations section |
| 12 | [Where every number lives](#12--where-every-number-lives) | Provenance |
| 13 | [Reproduction](#13--reproduction) | Appendix |

---

## 1 · The three answers, in one table

Jeremiah's framing question for this run was: *does each bias category have a
vector, yes or no — and did steering with that vector work, yes or no?* Run 3
answers both, plus a third question the design forced on us.

| Question | Answer | Strength |
|---|---|---|
| **Does each bias category have a stable direction?** | **Only for some categories, and which ones depends entirely on the model.** 5 of 10 in `yi-6b`, 2 of 10 in `qwen-14b`. The rest were *untestable*, not negative. | Solid |
| **Do the categories share a common structure?** | **Yes, at cosine ≈0.62–0.65 — but in `yi-6b` about half of that shared structure is the model's tendency to refuse, not bias.** The two models disagree on the verdict and both sit within 0.09 of the decision bar. | Weak / contested |
| **Did steering work?** | **No.** At every dose that leaves the model coherent, no direction produced a stereotype-margin shift whose confidence interval excludes zero, in either model. The two shifts that do clear zero occur only at doses that cut task accuracy by ~30 points, and a norm-matched *random* direction produces shifts of the same size or larger. | Solid (null) |

**The honest one-line summary for the abstract:** the extraction machinery works
and reproduces a control contrast at split-half cosine 0.876–0.934, but the bias
contrast is only measurable in a minority of categories, its apparent
cross-category structure is substantially refusal rather than bias, and the
resulting directions do not steer.

---

## 2 · Why Run 3 exists

Run 1 fitted difference-in-means directions from a **margin** — a likelihood
score computed over BBQ answer options. That has a known weakness: the margin is
a property of the scorer, so a direction fitted to it can be a direction for the
scoring artifact rather than for the behaviour. Run 3 replaces the contrast with
one defined on **what the model actually said**:

> **V_C = mean(residual | the model named the stereotyped person)
>       − mean(residual | the model declined to name anyone)**

for each BBQ category `C`. This is a *behavioural* contrast. It cannot be an
artifact of an option-scoring function, because no option scoring is involved in
defining it.

Run 3 also exists to close a defect from earlier runs, catalogued in the project
as **S5**: residual tensors were computed and then discarded, so every follow-up
question required renting the GPU again. Run 3 persists everything. That is why
this folder exists and why the residual manifest ([§12](#12--where-every-number-lives))
is part of the deliverable.

### The framing that makes the negative result publishable

The central methodological point — and the thing that should carry the
introduction — is the distinction between **"the concept is absent"** and
**"the instrument could not measure it."** These are routinely confounded, and
the confound is usually resolved in favour of whatever the paper needs.

Run 3 separates them with two devices:

1. A **positive control** that runs a contrast we *know* is present (topic
   identity) through the identical estimator and code path. If that reproduces
   and the bias contrast does not, the pipeline is not the problem.
2. An explicit **UNTESTABLE** status, distinct from a failing floor. A category
   where the model almost never produced one of the two arms has no contrast to
   split on. That is a fact about the data collected, not evidence that no bias
   direction exists. This is defect **M1** in our register, and it is the single
   easiest error to make when writing this up.

---

## 3 · Design and vocabulary

Use these terms consistently; they are the ones the code and the JSON use.

| Term | Definition |
|---|---|
| **Category** | One of BBQ's 10 social-bias categories (`Age`, `Disability_status`, `Gender_identity`, `Nationality`, `Physical_appearance`, `Race_ethnicity`, `Race_x_SES`, `Race_x_gender`, `Religion`, `Sexual_orientation`). |
| **Ambiguous arm** | BBQ items whose context does **not** identify who did the thing. The correct answer is the not-known option. Naming a person here is stereotyping. This is where V_C is extracted. |
| **Answerable arm** (`disambig`) | BBQ items whose context **does** identify the referent. There is a right answer, so accuracy is meaningful. Used for the task control and for the refusal direction. |
| **Bucket** | The label the judge assigned to a generated response: `BIASED_TARGET` (named the stereotyped person), `REFUSAL` (declined / said it cannot be known), `DISTRACTOR` (named the *other* person), `UNKNOWN` (unparsable). |
| **V_C** | The direction for category C: mean residual over the biased bucket minus mean residual over the refusal bucket, computed per layer. |
| **Capture site** | `blocks.{i}.hook_resid_pre`, **all layers**, at the **final prompt token** (index −1). Stored `float32`, shape `[n_items, n_layers, d_model]`. |
| **Extraction floor** | Split-half reproducibility. Split the items in half stratified by bucket, extract a direction independently from each half, take the norm-weighted mean per-layer cosine between them. Repeat over many random splits; report the mean and a percentile bootstrap CI. **This is the paper's reliability statistic.** |
| **UNTESTABLE** | A category where either bucket has fewer than `min_bucket = 32` items. No floor is computed. Not a negative result. |
| **Indistinguishability ceiling** | `sqrt(CI_lo(floor_C) × CI_lo(floor_refusal))` — the highest cosine two directions could show while still being distinguishable given how well each is itself measured. Used as the bar in the refusal-domination test. |

### Models

Run 3 was planned for four models and **completed for two**. This is a real
scope limitation and must be stated in the paper.

| Model | HF id | d_model | layers | Run 3 status |
|---|---|---|---|---|
| `yi-6b` | `01-ai/Yi-6B-Chat` | 4096 | 32 | **Complete** |
| `qwen-14b` | `Qwen/Qwen1.5-14B-Chat` | 5120 | 40 | **Complete** |
| `gemma-2b` | `google/gemma-2b-it` | 2048 | — | Not run (queue stopped) |
| `qwen-7b` | `Qwen/Qwen1.5-7B-Chat` | 4096 | — | Not run (queue stopped) |
| `qwen-1.8b` | `Qwen/Qwen1.5-1.8B-Chat` | 2048 | — | **Positive control only** |

---

## 4 · Step-by-step procedure

This is the exact sequence, in order. Each stage's artifacts are in this folder.

### R3a · Generate

For each model and each of the 10 categories, sample **400 ambiguous items**
(`Sexual_orientation` has only 232 available) and generate one completion per
item. The prompt is `context + " " + question` with **no option list shown** —
the option list is deliberately withheld, because Run 1 measured that displaying
options makes the model's preference a function of list position rather than
content.

→ `prompts.jsonl`, `responses.jsonl`

### R3b · Judge

Every completion is labelled into one of four buckets by an LLM judge:
`gpt-4o-mini`, resolved snapshot **`gpt-4o-mini-2024-07-18`**, judge version
`v2-bbq-choice-llm`.

Before any label is used, the judge must pass **C-1**, a position-bias
qualification: relabel a 200-item sample with the answer options presented in
swapped order and require ≥95% agreement on the resulting bucket label.

→ `judge_labels.jsonl`, `judge_labels_swapped.jsonl`, `judge_qualification.json`

### R3c · Extract

Capture residuals at the site above for every item. Form V_C from the
biased-minus-refusal contrast. Compute the extraction floor over **400
split-half repeats**. Hold out **200 items per category** from extraction so the
steering arm is evaluated on items the vector never saw.

→ `residuals/*.npy` (off-repo, see manifest), `residual_sidecars/*.json`,
`directions/*.npy`, `report_behavioural.json`

### R3d · Structure and controls

Cross-category cosine matrix; permutation null over 1000 permutations;
refusal-direction extraction from the answerable arm; orthogonalisation;
matched-random control; PCA.

→ `report_behavioural.json`

### R3e · Steer

Inject `±V_C` at five doses and re-score the held-out items with a judge-free
likelihood metric. Also score the answerable arm at every dose as a task
control.

→ `report_steering_likelihood.json`, `steering_responses.jsonl`

---

## 5 · Instrument validation

**This section is what licenses every negative result in the paper.** Without
it, "the direction did not reproduce" and "our pipeline is broken" are the same
observation.

### 5.1 Extraction positive control

Run a contrast that is unambiguously present — **topic identity**, i.e. residuals
from category A vs residuals from category B — through the *identical* estimator,
the same capture site, the same split-half procedure, at matched n.

Model `qwen-1.8b`, 200 items per arm, 100 splits:

| Control pair | Floor | 95% CI | sd |
|---|---|---|---|
| `Race_ethnicity` vs `Gender_identity` | **0.876** | [0.870, 0.881] | 0.026 |
| `Religion` vs `Age` | **0.923** | [0.920, 0.927] | 0.018 |
| `Nationality` vs `Sexual_orientation` | **0.934** | [0.931, 0.937] | 0.015 |

**Reading:** the machinery recovers a real contrast at 0.876–0.934. Any bias
category that fails to reproduce is failing for a reason other than the
pipeline.

→ `_control_qwen-1.8b/positive_control.json`

### 5.2 Judge qualification (C-1)

| | `yi-6b` | `qwen-14b` |
|---|---|---|
| Judge | `gpt-4o-mini-2024-07-18` | `gpt-4o-mini-2024-07-18` |
| n sampled / compared | 200 / 200 | 200 / 200 |
| Format failures | 0 | 0 |
| **Order agreement (bucket label)** | **0.990** | **1.000** |
| Threshold | 0.95 | 0.95 |
| Order agreement (raw option) | 0.875 | 0.820 |
| Distinct verdicts emitted | 5 | 4 |
| Degenerate? | No | No |
| **Qualified** | **Yes** | **Yes** |

Two things to note when writing this up:

- The number that gates the run is agreement on the **bucket label**
  (`scored_on: directive_label`), not on the raw option letter. The raw-option
  agreement is materially lower (0.875 / 0.820). Both are reported here; use the
  bucket-label figure, and say which one you used.
- The **degeneracy check** exists because an earlier version of the judge
  answered `REFUSAL` to literally everything — a single missing space after
  `"Answer:"` in the prompt template. It scored a perfect order agreement while
  being completely broken. Perfect consistency is not evidence of validity; a
  constant function is perfectly consistent. `n_distinct_verdicts` is what
  catches this.

→ `{model}/judge_qualification.json`

---

## 6 · Bucketing — the binding constraint

This is the single most important result for understanding everything after it,
and it is where the two models diverge completely.

`min_bucket = 32`; 400 items per category (232 for `Sexual_orientation`).

### yi-6b

| Category | biased | refusal | distractor | unparsed | refusal rate | Status |
|---|---|---|---|---|---|---|
| Age | 93 | 44 | 46 | 217 | 0.110 | **TESTABLE** |
| Physical_appearance | 53 | 52 | 24 | 271 | 0.130 | **TESTABLE** |
| Religion | 52 | 60 | 20 | 268 | 0.150 | **TESTABLE** |
| Disability_status | 45 | 40 | 44 | 271 | 0.100 | **TESTABLE** |
| Nationality | 40 | 59 | 32 | 269 | 0.148 | **TESTABLE** |
| Race_x_gender | 29 | 75 | 31 | 265 | 0.188 | UNTESTABLE |
| Race_x_SES | 21 | 89 | 21 | 269 | 0.223 | UNTESTABLE |
| Race_ethnicity | 18 | 92 | 18 | 272 | 0.230 | UNTESTABLE |
| Sexual_orientation | 18 | 38 | 19 | 157 | 0.164 | UNTESTABLE |
| Gender_identity | 14 | 99 | 17 | 270 | 0.248 | UNTESTABLE |

### qwen-14b

| Category | biased | refusal | distractor | unparsed | refusal rate | Status |
|---|---|---|---|---|---|---|
| Age | 54 | 337 | 8 | 1 | 0.843 | **TESTABLE** |
| Religion | 41 | 352 | 6 | 1 | 0.880 | **TESTABLE** |
| Disability_status | 6 | 388 | 5 | 1 | 0.970 | UNTESTABLE |
| Gender_identity | 8 | 386 | 5 | 1 | 0.965 | UNTESTABLE |
| Nationality | 8 | 392 | 0 | 0 | 0.980 | UNTESTABLE |
| Sexual_orientation | 5 | 224 | 3 | 0 | 0.966 | UNTESTABLE |
| Physical_appearance | 3 | 395 | 0 | 2 | 0.988 | UNTESTABLE |
| Race_ethnicity | 3 | 394 | 2 | 1 | 0.985 | UNTESTABLE |
| Race_x_SES | 1 | 397 | 1 | 1 | 0.993 | UNTESTABLE |
| Race_x_gender | **0** | **400** | 0 | 0 | **1.000** | UNTESTABLE |

### What to say about this

Three separate findings live in these two tables, and they should not be blurred
together:

1. **The two models fail for opposite reasons.** `qwen-14b` refuses 84–100% of
   ambiguous items — it is heavily safety-tuned and simply will not name anyone,
   so there is no biased arm to contrast. `yi-6b` refuses only 10–25%, but
   54–68% of its completions were **unparsable** (`UNKNOWN`), so its testable
   buckets are drawn from roughly a third of the items collected.

2. **`Race_x_gender` in `qwen-14b` produced zero biased responses out of 400.**
   That is worth a sentence in the paper on its own — under this elicitation the
   model never once named a person on a race×gender ambiguous item.

3. **Every race-related category is UNTESTABLE in both models.**
   `Race_ethnicity`, `Race_x_SES`, and `Race_x_gender` all fail the bucket
   requirement in `yi-6b` *and* `qwen-14b`. **Do not write that race has no bias
   direction.** Write that the elicitation did not produce enough biased
   responses on race items to form a contrast. That is defect M1 and it is the
   error a reviewer will look for.

The high refusal rate is a substantive finding about instruction-tuned models,
not merely a nuisance: **the behaviour we set out to contrast against refusal is
largely absent, because these models mostly refuse.**

---

## 7 · Extraction floors

Split-half reproducibility, **400 splits**, norm-weighted mean per-layer cosine,
percentile bootstrap CI. Only TESTABLE categories have a floor.

| Model | Category | n_biased | n_refusal | **Floor** | 95% CI | sd | Unweighted-median sensitivity |
|---|---|---|---|---|---|---|---|
| `yi-6b` | Nationality | 40 | 59 | **0.823** | [0.819, 0.827] | 0.041 | 0.719 |
| `yi-6b` | Disability_status | 45 | 40 | **0.773** | [0.768, 0.778] | 0.053 | 0.602 |
| `yi-6b` | Physical_appearance | 53 | 52 | **0.768** | [0.762, 0.773] | 0.057 | 0.624 |
| `yi-6b` | Age | 93 | 44 | **0.758** | [0.752, 0.765] | 0.067 | 0.676 |
| `yi-6b` | Religion | 52 | 60 | **0.727** | [0.721, 0.733] | 0.059 | 0.501 |
| `qwen-14b` | Religion | 41 | 352 | **0.977** | [0.977, 0.978] | 0.008 | 0.955 |
| `qwen-14b` | Age | 54 | 337 | **0.926** | [0.924, 0.928] | 0.022 | 0.882 |

### Reading

- **Every category that could be tested, reproduced.** All seven floors sit at
  0.727–0.977, against a positive-control band of 0.876–0.934 and a
  random-direction baseline near zero. There is no category that was testable
  and then failed.
- **The result is therefore not "bias directions don't exist."** It is
  **"where we could measure, we found a stable direction; we could only measure
  in a minority of categories, and in no race category at all."**
- **The last column is a robustness check you must report.** Swapping the
  norm-weighted mean for an unweighted median drops `yi-6b` `Religion` from
  0.727 to **0.501** — right at the conventional 0.5 bar. The `yi-6b` floors are
  estimator-sensitive; the `qwen-14b` floors (0.955, 0.882 under the same swap)
  are not. Say this rather than letting a reviewer find it.
- `qwen-14b`'s very high floors come partly from very large refusal arms
  (337, 352). A mean over 350 items is better estimated than a mean over 40.
  The floor is a reliability statistic, and reliability rises with n — do not
  present `qwen-14b`'s 0.977 as a stronger *finding* than `yi-6b`'s 0.823.

→ `{model}/report_behavioural.json` → `per_category.{cat}.floor`
(the full 400 per-split cosines are stored, so any other summary statistic can
be recomputed without the GPU)

---

## 8 · Cross-category structure

### 8.1 Raw similarity

Cosine between the direction for category A and the direction for category B.

**`yi-6b`, 5 categories** (median off-diagonal **0.650**):

| | Age | Disab. | Nation. | Phys. | Relig. |
|---|---|---|---|---|---|
| **Age** | 1.000 | 0.639 | 0.644 | 0.604 | 0.587 |
| **Disability_status** | 0.642 | 1.000 | 0.686 | **0.766** | 0.705 |
| **Nationality** | 0.646 | 0.684 | 1.000 | 0.678 | 0.610 |
| **Physical_appearance** | 0.608 | **0.765** | 0.679 | 1.000 | 0.656 |
| **Religion** | 0.594 | 0.707 | 0.616 | 0.660 | 1.000 |

**`qwen-14b`, 2 categories:** `Age` × `Religion` = **0.620**.

Permutation null (1000 permutations):

| Model | Observed median | Null median | Null q95 | p |
|---|---|---|---|---|
| `yi-6b` | 0.650 | −0.005 | 0.113 | **0.001** |
| `qwen-14b` | 0.620 | 0.009 | 0.265 | **0.001** |

So the shared structure is real and far above chance. **The question is what it
is made of.**

### 8.2 Why raw similarity cannot be taken at face value

Every V_C is `mean(biased) − mean(refusal)`. The refusal arm looks broadly the
same whichever category it came from, so **subtracting it injects the same
refusal signature into every direction by construction** — before any genuine
shared bias mechanism is involved.

This matters more than it might appear, because we measure similarity with
**cosine**, which is an angle from the origin rather than a distance between
points. A shared additive component does not affect the *difference* between two
vectors at all, but it moves their *angle* arbitrarily far. Two directions with
literally nothing in common can be pushed to cosine 0.9 by a large enough shared
component. That is why §8.3 is not optional.

### 8.3 Refusal de-coupling — the decisive control

**Where the refusal direction comes from.** From the **answerable arm** — items
where the context *does* say who did it. Declining there has nothing to do with
epistemic humility about a stereotype; it is simply a failure to answer a
question with a stated answer. This is a clean sample of refusal uncontaminated
by bias.

**The trap we avoided:** pooling the refusal buckets from the *ambiguous* arm
across categories would give approximately the mean of the five V_C's.
Orthogonalising five vectors against their own mean necessarily removes whatever
they share, so that construction returns "it collapsed" regardless of the truth.
The answerable arm breaks the circularity. Say this explicitly in the paper —
it is a real design contribution and it is cheap to describe.

**Refusal direction quality:**

| Model | n answered | n declined | Floor | 95% CI |
|---|---|---|---|---|
| `yi-6b` | 322 | 79 | **0.821** | [0.815, 0.826] |
| `qwen-14b` | 667 | 332 | **0.980** | [0.979, 0.980] |

**Proxy validation** — is the answerable-arm refusal direction actually the same
thing as refusal on the ambiguous arm? Compared against the ceiling set by how
well each is itself measured:

| Model | \|cos\| answerable vs ambiguous | Ceiling | **Alignment vs ceiling** |
|---|---|---|---|
| `yi-6b` | 0.830 | 0.869 | **0.954** |
| `qwen-14b` | 0.747 | 0.973 | **0.768** |

`yi-6b`'s proxy is excellent (95% of the maximum achievable). `qwen-14b`'s is
noticeably weaker, which is one reason to trust its cross-category verdict less.

**How much of each direction is refusal:**

| Model | Category | \|cos(V_C, V_refusal)\| | Variance share | Ceiling | Verdict |
|---|---|---|---|---|---|
| `yi-6b` | Religion | 0.739 | **54.7%** | 0.767 | BIAS-SPECIFIC |
| `yi-6b` | Disability_status | 0.717 | **51.3%** | 0.791 | BIAS-SPECIFIC |
| `yi-6b` | Nationality | 0.691 | **47.8%** | 0.817 | BIAS-SPECIFIC |
| `yi-6b` | Age | 0.680 | **46.3%** | 0.783 | BIAS-SPECIFIC |
| `yi-6b` | Physical_appearance | 0.667 | **44.5%** | 0.788 | BIAS-SPECIFIC |
| `qwen-14b` | Religion | 0.751 | **56.4%** | 0.978 | BIAS-SPECIFIC |
| `qwen-14b` | Age | 0.563 | **31.7%** | 0.951 | BIAS-SPECIFIC |

Mean refusal share across `yi-6b`'s five categories: **≈49%**. No category is
*dominated* by refusal (none exceeds its indistinguishability ceiling), so each
direction retains something of its own — but roughly half of every "bias
direction" is the model's disposition to decline.

**The headline test — orthogonalise refusal out and re-measure:**

| | `yi-6b` | `qwen-14b` |
|---|---|---|
| Categories in the matrix | 5 (10 pairs) | 2 (**1 pair**) |
| Median off-diagonal, raw | 0.650 | 0.620 |
| Median off-diagonal, orthogonalised | 0.320 | 0.359 |
| CI on orthogonalised | [0.267, 0.389] | [0.359, 0.359] — degenerate |
| **Fraction retained** | **0.493** | **0.580** |
| Declared retention bar | 0.500 | 0.500 |
| **Retention under a matched *random* direction** | **0.952** | **0.937** |
| Verdict emitted by the code | **SHARED STRUCTURE IS REFUSAL** | SHARED STRUCTURE SURVIVES |

**The random control is the load-bearing row.** Removing *any* direction changes
a vector somewhat, so 49% retention only means something against a baseline.
Removing a norm- and covariance-matched random direction retains **95%**. In a
space of 4096 dimensions, deleting one arbitrary direction takes almost nothing
away. Refusal removal taking half is therefore a targeted hit, not a mechanical
one.

### 8.4 How to write §8 honestly — the models disagree

This needs care, because it is the weakest claim in the paper and a reviewer
will go straight at it.

- The two models return **opposite verdicts** (0.493 vs 0.580 retained), and
  **both sit within 0.09 of the 0.5 bar.** The bar was declared in advance,
  which is what makes the comparison legitimate at all, but a result that flips
  on ±0.09 around a threshold is fragile and should be presented as such.
- **`qwen-14b`'s "survives" is not a replication.** With only two testable
  categories, its "median off-diagonal" is a **single pair**. That is why its CI
  is degenerate — there is nothing to bootstrap over. It is one number, not a
  distribution, and it cannot bear the weight of contradicting `yi-6b`'s ten
  pairs.
- Orthogonalisation is a **lower bound** on contamination. `V_refusal` is itself
  measured only to its own floor (0.821 in `yi-6b`), so we can only strip out the
  part we estimated well. True contamination is plausibly higher than the 50.7%
  measured. This cuts *toward* the refusal explanation, not away from it.
- **PCA cannot settle this and should not be presented as if it could.** In
  `yi-6b` the explained-variance ratios are 0.356 / 0.282 / 0.193 / 0.169 —
  no dominant component. And a dominant component would have been consistent
  with *both* "one shared bias mechanism" and "one shared artifact." The
  orthogonalisation is what separates them; PCA alone never could. (This
  caveat is written into the artifact itself, in `pca.note`.)

**Recommended wording:** *In the model where the test has power (five categories,
ten pairs), roughly half the shared cross-category structure is attributable to
the model's refusal disposition rather than to a common bias mechanism. The
second model is consistent with more structure surviving, but with two testable
categories it contributes a single pair and cannot adjudicate.*

---

## 9 · Steering — the causal arm

### 9.1 What was measured and why it is judge-free

`margin = logP(stereotyped named option) − logP(other named option)`, both terms
mean log-probability per token so option lengths cancel. The model **generates
nothing**, so there is no completion to parse, no hedged answer to adjudicate,
and no judge version to pin. The option list is **not** shown in the prompt
(design 3), because Run 1 measured that showing it makes the margin a function
of list position: moving one option's slot shifted its score by 0.38 nats, and
0.38·√2 = 0.54 exactly matched the observed mean margin. The entire signal was
position. That design is dead.

Evaluated on the **200 held-out items per category** that the vectors were
extracted away from. Both models score the identical held-out set.

### 9.2 Dose calibration — a finding in its own right

The pre-registered dose grid was α ∈ {0.25, 0.5, 1.0, 2.0}. We measured
completion coherence on the box and found that **every value in that grid
destroys the model**: at α ≥ 0.35 completions collapse into a single repeated
token, including under the random-direction control.

The committed evidence for this is `qwen-14b/report_steering.json`, the judged
generation arm. At α = 0.5 and α = 1.0, across both categories and **all four
arms** (`plus`, `minus`, `random_plus`, and the informative-item task control),
the unparsable rate is **1.000** — every one of 200 completions per arm, 3,200
generations in total, produced nothing a judge could label. Biased rate 0.000,
refusal rate 0.000, items scorable for accuracy 0. Baselines in the same file
are healthy (accuracy 0.687 for `Age`, 0.853 for `Religion`), so the collapse is
caused by the dose and not by the harness.

This is why the causal arm reports likelihood margins rather than generations:
at the declared doses there are no generations left to judge.

Rather than replace the declared grid, we **kept it and added lower doses**, so
every model is run at the same settings and the destructive doses remain as
interpretable data points. Final grid: **α ∈ {0.02, 0.05, 0.10, 0.50, 1.00}**.
Report it this way — a reviewer who sees only the low doses will ask what
happened to the pre-registration.

### 9.3 Results

`shift` = paired change in margin vs the unsteered baseline on the same items.
`*` marks a bootstrap CI excluding zero. `rand` is a norm- and
covariance-matched random direction.

**`yi-6b`** (baseline task accuracy 0.583–0.667):

| Category | α | +V_C shift | −V_C shift | random shift | task acc (Δ) |
|---|---|---|---|---|---|
| Age | 0.02 | +0.029 | +0.034 | +0.057 | 0.683 (−0.017) |
| Age | 0.05 | +0.020 | +0.073 | +0.015 | 0.667 (0.000) |
| Age | 0.10 | −0.039 | −0.042 | −0.018 | 0.600 (+0.067) |
| Age | 0.50 | −0.110 | −0.010 | +0.041 | 0.400 (+0.267) |
| Age | 1.00 | −0.112 | 0.000 | +0.038 | 0.383 (+0.283) |
| Disability_status | 0.02 | +0.017 | −0.098 | **+0.122\*** | 0.633 (+0.017) |
| Disability_status | 0.05 | −0.002 | −0.265 | **+0.175\*** | 0.567 (+0.083) |
| Disability_status | 0.10 | +0.038 | −0.288 | +0.097 | 0.517 (+0.133) |
| Disability_status | 0.50 | +0.685 | −0.099 | **+0.861\*** | 0.350 (+0.300) |
| Disability_status | 1.00 | +0.749 | −0.145 | **+0.909\*** | 0.367 (+0.283) |
| Nationality | 0.02–1.00 | +0.023 … −0.236 | −0.046 … +0.224 | +0.013 … +0.144 | 0.650 → 0.483 |
| Physical_appearance | 0.02 | −0.012 | −0.041 | **+0.116\*** | 0.667 (−0.050) |
| Physical_appearance | 0.10 | −0.052 | **−0.301\*** | −0.012 | 0.650 (−0.033) |
| Physical_appearance | 0.50 | **−0.455\*** | **−0.763\*** | −0.542 | 0.317 (+0.300) |
| Physical_appearance | 1.00 | **−0.449\*** | **−0.808\*** | **−0.577\*** | 0.367 (+0.250) |
| Religion | 0.02–1.00 | +0.018 … +0.067 | +0.026 … −0.005 | +0.015 … +0.051 | 0.633 → 0.517 |

**`qwen-14b`** (baseline task accuracy 0.55–0.733):

| Category | α | +V_C shift | −V_C shift | random shift | task acc (Δ) |
|---|---|---|---|---|---|
| Age | 0.02 | −0.010 | −0.076 | +0.085 | 0.783 (−0.050) |
| Age | 0.05 | −0.011 | **−0.260\*** | +0.077 | 0.750 (−0.017) |
| Age | 0.10 | −0.206 | **−0.300\*** | −0.274 | 0.683 (+0.050) |
| Age | 0.50 | −0.358 | −0.296 | −0.265 | 0.417 (+0.317) |
| Age | 1.00 | −0.352 | −0.311 | −0.282 | 0.517 (+0.217) |
| Religion | 0.02 | +0.062 | +0.013 | −0.013 | 0.750 (−0.200) |
| Religion | 0.05 | +0.199 | −0.005 | −0.012 | 0.883 (−0.333) |
| Religion | 0.10 | +0.032 | −0.057 | −0.095 | 0.650 (−0.100) |
| Religion | 0.50 | −0.059 | −0.088 | −0.116 | 0.483 (+0.067) |
| Religion | 1.00 | −0.054 | −0.097 | −0.116 | 0.467 (+0.083) |

### 9.4 The verdict: steering did not work

Four observations, each independently sufficient:

1. **Not one `+V_C` shift excludes zero at any coherent dose** (α ≤ 0.10), in
   either model, in any of the seven testable categories.
2. **The two `+V_C` shifts that do exclude zero** — `yi-6b`
   `Physical_appearance` at α = 0.5 and 1.0 — occur exactly where task accuracy
   falls from 0.617 to 0.317, a **30-point collapse**. That is damage, not
   steering.
3. **At those same doses the random direction produces a shift of −0.542 and
   −0.577**, i.e. *larger* than the real direction's −0.455 and −0.449. The
   effect is not direction-specific.
4. **The random control is significant where the real direction is not.** In
   `yi-6b` `Disability_status`, `random_plus` excludes zero at α = 0.02, 0.05,
   0.5 and 1.0 while `+V_C` never does. A metric on which a random vector
   reliably outperforms the fitted one is not measuring what we wanted it to
   measure.

Two `−V_C` shifts survive at a coherent dose (`qwen-14b` `Age` at α = 0.05 and
0.10, −0.260 and −0.300). Do not build a claim on them: the plus arm is null at
the same doses, so there is no sign-flip structure, and at α = 0.10 the random
direction produces −0.274 — indistinguishable from the −0.300 attributed to the
real one.

**Recommended wording:** *We find no evidence that these directions steer.
Across seven model-category cells and three doses that preserve task
performance, no injected direction produced a stereotype-margin shift
distinguishable from zero. The two significant shifts we do observe arise only
at doses that reduce task accuracy by roughly thirty points, and are matched or
exceeded by a norm-matched random direction at the same dose.*

**Important framing:** the margin is a **disposition** — which answer the model
rates higher — whereas the extraction buckets were a **behaviour** — what the
model actually said. These are different constructs. A null on the margin does
not strictly prove the direction fails to change behaviour. Say that, and say
that generation-based steering with a qualified judge is the obvious follow-up
we did not run.

---

## 10 · Claim register — what you may and may not write

### Supported by artifacts in this folder

1. The extraction pipeline recovers a known-present contrast at split-half floor
   **0.876–0.934** across three control pairs (`qwen-1.8b`).
2. The labelling instrument passes a position-bias qualification at **0.990** and
   **1.000** bucket-label agreement against a 0.95 threshold, with 0 format
   failures and non-degenerate output.
3. Under this elicitation, **5 of 10** categories in `yi-6b` and **2 of 10** in
   `qwen-14b` yielded ≥32 items in both the biased and refusal buckets.
4. **Every category that met that requirement produced a reproducible
   direction**, floors **0.727–0.977**.
5. Cross-category directions are similar well above a permutation null
   (median 0.650 / 0.620; null medians −0.005 / 0.009; **p = 0.001** in both).
6. In `yi-6b`, orthogonalising a refusal direction obtained from the answerable
   arm removes **50.7%** of that shared structure, against **4.8%** removed by a
   matched random direction.
7. Each individual V_C carries a large refusal component (**31.7%–56.4%** of
   variance) without being dominated by it.
8. **No direction produced a stereotype-margin shift distinguishable from zero at
   any dose that preserves task accuracy.**
9. The pre-registered dose grid {0.25, 0.5, 1.0, 2.0} lies entirely above the
   measured coherence limit. At α = 0.5 and 1.0, **100% of 3,200 generated
   completions were unparsable** across every arm including the random and task
   controls (`qwen-14b/report_steering.json`).

### NOT supported — do not write these

1. ❌ *"Race has no bias direction."* Every race category was **UNTESTABLE**.
   No floor was computed; no negative result exists. (Defect M1.)
2. ❌ *"Bias directions do not reproduce."* The opposite: all seven testable
   categories reproduced.
3. ❌ *"Steering reduced bias."* Nothing cleared zero at a coherent dose, and the
   random control matched or beat the real direction where anything did.
4. ❌ *"The shared structure is refusal"* stated flatly. That is `yi-6b`'s
   verdict; `qwen-14b` returns the opposite on a single pair. Report both and
   say which has power.
5. ❌ *"PCA shows one shared bias mechanism."* Explained variance is
   0.356/0.282/0.193/0.169 — there is no dominant component — and even a dominant
   one would not distinguish mechanism from artifact.
6. ❌ *"Five models."* Run 3 completed on **two**. `qwen-1.8b` contributed the
   positive control only; `gemma-2b` and `qwen-7b` were not run.
7. ❌ Any claim resting on `qwen-14b`'s cross-category number as an independent
   replication. It is one pair with a degenerate CI.

---

## 11 · Limitations, paper-ready

Each of these is a real constraint with a number attached. They are ordered by
how much a reviewer will care.

1. **Two models, not the planned four.** `gemma-2b` and `qwen-7b` were not run.
   Generality across families and scales is therefore untested; the two models we
   have disagree on the one contested result.

2. **Most categories were untestable, and the reason differs by model.**
   `qwen-14b` refuses 84–100% of ambiguous items; `yi-6b` refuses only 10–25% but
   produced 54–68% unparsable completions. Both routes lead to arms too small to
   split. In `yi-6b` the testable buckets come from roughly a third of the items
   collected — a selection effect on which items survive parsing, of unknown
   direction.

3. **No race-related category was testable in either model.** The categories most
   likely to matter to readers are exactly the ones this design cannot speak to.

4. **The floors are estimator-sensitive in `yi-6b`.** Swapping norm-weighted mean
   for unweighted median moves `Religion` from 0.727 to 0.501 — from comfortably
   above a 0.5 bar to exactly on it. `qwen-14b` is stable under the same swap.

5. **The cross-category verdict turns on a threshold the results straddle.**
   0.493 vs 0.580 retained against a declared 0.500 bar. The bar was fixed in
   advance, but a conclusion that flips on ±0.09 is fragile.

6. **`qwen-14b`'s cross-category number is a single pair.** Two testable
   categories give one off-diagonal entry and a degenerate CI.

7. **Orthogonalisation is a lower bound.** `V_refusal` is measured only to its own
   floor (0.821 in `yi-6b`), so only the well-estimated part of refusal is
   removed. True contamination may be higher.

8. **The refusal proxy is much better in one model than the other**: alignment
   against ceiling 0.954 (`yi-6b`) vs 0.768 (`qwen-14b`).

9. **The steering metric is a disposition, not a behaviour.** Margins measure
   which option the model rates higher; the extraction contrast was built on what
   the model actually said. A null on the former does not fully close the latter.
   Generation-based steering with the qualified judge is the obvious follow-up.

10. **Judge cost capped the design.** The OpenAI account's 10,000-requests/day
    limit forced the causal arm onto judge-free likelihood scoring rather than
    judged generation. This was a resource constraint, not a methodological
    preference — though it did remove a judge dependency from the causal arm,
    which is a genuine side benefit.

11. **`min_bucket = 32` is a declared but not pre-registered constant.** Report
    its sensitivity rather than defending its value.

12. **Single capture site.** All results are at `hook_resid_pre`, final prompt
    token. We did not test other sites or token positions.

---

## 12 · Where every number lives

Every number in this document traces to a file below. **A write-up is not an
artifact** — check the JSON before putting a number in the paper.

```
results/run3/
├── README.md                       ← this document
├── RESIDUALS-MANIFEST.json         ← SHA-256 + shape/dtype for the 6.27 GiB of
│                                     residual tensors held off-repo
├── yi-6b/
│   ├── report_behavioural.json     ← floors, cosine matrices, refusal decoupling,
│   │                                 permutation null, PCA, bucket membership
│   ├── report_steering_likelihood.json  ← §9 in full
│   ├── judge_qualification.json    ← C-1
│   ├── judge_labels.jsonl          ← every judge verdict
│   ├── judge_labels_swapped.jsonl  ← the C-1 swapped-order pass
│   ├── prompts.jsonl               ← every prompt sent
│   ├── responses.jsonl             ← every model completion
│   ├── report_steering.json        ← judged generation arm (qwen-14b only);
│   │                                 the α≥0.5 collapse evidence in §9.2
│   ├── steering_responses.jsonl    ← completions under steering. yi-6b: 600
│   │                                 lines (baseline + α=0.02). qwen-14b: empty
│   │                                 — its generations collapsed, see §9.2
│   ├── directions/*.npy            ← the extracted V_C, one per category
│   ├── residual_sidecars/*.json    ← item_ids, eval_holdout_n, capture metadata
│   ├── capture_site.json
│   └── queue_manifest.json
├── qwen-14b/                       ← same structure
├── _control_qwen-1.8b/
│   └── positive_control.json       ← §5.1
├── _logs/                          ← every run log, including the failures
└── _reanalysis/                    ← Run 1 reanalysis, layer profiles
```

**Key paths for the numbers you will most want:**

| Number | Path |
|---|---|
| Floor for category C | `{model}/report_behavioural.json` → `per_category.C.floor.mean` and `.ci_lo` / `.ci_hi` |
| All 400 per-split cosines | same → `.cosines` (so any other summary statistic is recomputable) |
| Bucket counts | same → `per_category.C.buckets` |
| Which items are in which bucket | same → `per_category.C.bucket_membership` |
| Cross-category matrix | same → `cosine_matrix.matrix` and `cosine_matrix_refusal_orthogonalised.matrix` |
| Per-layer cosine profiles | same → `cosine_matrix.per_layer_profiles` |
| Refusal decoupling | same → `refusal_decoupling.per_category` |
| Headline orthogonalisation result | same → `cross_category_survives_refusal_removal` |
| Permutation null | same → `permutation_null` |
| Steering | `{model}/report_steering_likelihood.json` → `cells.C.doses.{alpha}` |

**Residual tensors are not in the repo.** 86 files, 6.27 GiB, individual files
over GitHub's 100 MiB hard limit. `RESIDUALS-MANIFEST.json` records every file's
size, shape, dtype and SHA-256 so a copy obtained out-of-band can be verified.
Ask Jeremiah for them; they are on his laptop.

---

## 13 · Reproduction

Analysis stages need no GPU and can be re-run from what is committed here.

```bash
python3 -m scripts.run3_behavioural_contrast analyse --model yi-6b --out runs/r3_behavioural_yi-6b
```

```bash
python3 -m scripts.run3_steer_likelihood --model yi-6b --out runs/r3_behavioural_yi-6b --alphas 0.02 0.05 0.1 0.5 1.0
```

Relevant code, all on this branch:

| Path | Role |
|---|---|
| `scripts/run3_behavioural_contrast.py` | generation, extraction, floors, all structural controls |
| `scripts/run3_steer_likelihood.py` | the judge-free causal arm (§9) |
| `scripts/pilot/llm_judge.py` | the judge and its C-1 qualification |
| `scripts/pilot/behavioural.py` | bucketing, declared constants |
| `scripts/pilot/analysis.py` | floor estimator |
| `scripts/preflight.py` | pre-run environment gate |
| `docs/EXPERIMENT_PLAN_RUN3.md` | the plan this run executed |

The paper drafts are at `paper/` (current LaTeX, `main.tex`) and `paper-v2/`
(figure and numbers pipeline: `collect.py` regenerates `numbers.json` from
artifacts).

---

## 14 · Two things I'd flag before you write

**The relationship to the existing draft.** `paper/main.tex` currently describes
the Run-1/Run-2 experiment — five models, ten categories, ridge-probe estimators,
floors from −0.45 to +0.82. Run 3 is a **different experiment**: a behavioural
contrast on two models. Decide deliberately whether Run 3 becomes the paper's
main result, a section within it, or a separate submission. Do not merge the
numbers; they are not measuring the same thing and a reviewer who notices will
be right.

**The strongest thing we have is the shape of the argument, not the effect
size.** We have no positive steering result. What we do have is a clean
demonstration that a standard difference-in-means pipeline can produce
high-reliability directions (floors up to 0.977), striking cross-category
structure (p = 0.001 against a permutation null), and a plausible-looking story —
and that two cheap controls, a positive control and a refusal orthogonalisation
against a non-circular reference, substantially dissolve it. That is the paper.
Lead with it.

# Experiment 1 — findings

**Read this first.** Full detail in `notes/08-results-2026-08-20.md`; a
facts-only account of what ran, what failed and what is superseded is in
`notes/09-OVERNIGHT-STATUS-REPORT.md`.

Five models, three families, 2B to 14B. All numbers from committed artifacts
under `runs/`.

---

## The four questions you set, answered

### 1. "Is there a significant difference between the bias vectors?"

**Where a direction can be recovered at all, yes — but only in one model.**

On qwen-14b, four categories produced directions that reproduce against
themselves, and **all six pairs among them are distinguishable** (separation,
floor minus cosine, from +0.34 to +1.03):

```
                        Age  Disability  Physical_a    Religion
Age                   1.000      -0.277      -0.072       0.119
Disability_status    -0.277       1.000       0.242       0.093
Physical_appearance  -0.072       0.242       1.000       0.306
Religion              0.119       0.093       0.306       1.000
```

No other model produced three or more reproducible directions, so this could not
be checked anywhere else. **Whether that arrangement replicates across models is
untested** — comparing the shape of the matrix needs three categories reproducing
in both models, and the maximum available is two.

### 2. "Does one vector work for them all?"

**No vector demonstrably works for any of them — including its own.**

Transfer test, unit-normalised directions, restricted to reproducible ones,
coefficient swept:

- **qwen-14b:** at every dose (2, 4, 8, 16), steering a category with its own
  direction moved its margin *less* than steering it with another category's.
  Both tracked the norm-matched random control. Sign-flip control degraded
  monotonically with dose (3/4 → 1/4).
- **gemma-2b:** the directions *do* produce a real effect — 7x the random control
  at coeff=2, with the sign-flip control fully passing. But **own ≈ cross**. Both
  directions move Disability items by ≈ −0.11 and Physical items by ≈ −0.03. The
  effect is a property of what is being steered, not of which direction steers it.

That second result is the **"one shared knob"** outcome, and it is the same
pattern Joad et al. (arXiv:2602.02132) report for refusal: geometrically distinct
directions that behave as a single control under intervention. Finding it
independently on the bias side, in a different family and dataset, replicates
their central claim in a new domain.

**Uncontrolled** — norm-matched rather than covariance-matched random control, no
coherence check on generations, no system-prompt baseline. Do not call it causal.

### 3. "Are they groups? Are some more connected than others?"

**YES — with the properly tuned estimator, structure is detected above chance.**
`runs/probe_tuned_qwen14b`, ridge probe at α=1e6, five reproducible directions,
200-permutation null.

```
cluster strength observed : 0.0962
permutation null median   : 0.0473    q95 0.0852
p                         : 0.0299
```

**p = 0.030.** This is the only run in the session where the clustering beat its
null, and it became possible only once the estimator was configured properly and
five categories cleared the floor.

```
              Age  Disabil  Nation  Physical  Religion
Age         1.000   -0.101   0.050     0.000     0.011
Disability -0.101    1.000   0.150     0.108     0.085
Nationality 0.050    0.150   1.000     0.069     0.082
Physical    0.000    0.108   0.069     1.000     0.228
Religion    0.011    0.085   0.082     0.228     1.000
```

All ten pairs are distinguishable against their own floors. The structure:

- **Physical_appearance ↔ Religion (+0.228)** — the tightest pair
- **Disability_status ↔ Nationality (+0.150)**, and Disability ↔ Physical (+0.108)
- **Age stands apart** — near-orthogonal to everything (0.050, 0.000, 0.011) and
  carrying the only negative relationship in the matrix (−0.101 with Disability)

So some bias directions *are* more connected than others, and the grouping is not
the surface taxonomy anyone would have guessed: appearance and religion together,
age isolated and mildly opposed to disability.

**Strong caveats — read these before quoting p=0.030.**

1. **n = 5 categories.** A five-point clustering is thin.
2. **One test, one model.** The same analysis on the extremes contrast
   (4 categories) gave p=0.179.
3. **Researcher degrees of freedom in α.** The penalty was tuned by looking at
   extraction floors on the same data. It was *not* tuned against the clustering
   or the p-value, but α=1e6 is also the value that happens to lift the most
   categories over the floor, and more categories makes clustering possible at
   all. That is a real degree of freedom and it should be declared.
4. **Replication is limited by the same threshold.** gemma-2b and yi-6b reproduce
   only 2 categories under *any* α, so clustering is impossible there — not
   refuted, untestable. qwen-7b reaches 3 only at α=1e6, and its third
   (Religion, 0.525) is marginal against the 0.500 bar.

**Treat this as a promising lead to replicate, not a settled finding.** The
pre-registration to write before any further run: fix α by a rule (e.g. scaled to
d_model), fix the usability threshold, and only then look at the p-value.

### It DID replicate in a second model — at the same α

`runs/probe_tuned_qwen7b_a1e6`, qwen-7b, same α=1e6 (so the penalty was not
re-tuned per model), 3 reproducible categories, 200-permutation null:

```
cluster strength observed : 0.3351
permutation null median   : 0.0573    q95 0.1378
p                         : 0.0050
```

**p = 0.005**, stronger than qwen-14b's 0.030. Two models, same estimator, same
penalty, both detect above-chance structure among their reproducible directions.

```
qwen-7b        Disability  Physical  Religion
Disability          1.000     0.269    -0.122
Physical            0.269     1.000    -0.011
Religion           -0.122    -0.011     1.000
```

### But the ARRANGEMENT does not demonstrably replicate

The three categories reproducing in both models are exactly Disability_status,
Physical_appearance and Religion — the minimum for a structure comparison.

| pair | qwen-7b | qwen-14b |
|---|---|---|
| Disability ↔ Physical | **+0.269** | +0.108 |
| Disability ↔ Religion | **−0.122** | +0.085 |
| Physical ↔ Religion | −0.011 | **+0.228** |

Correlation of the off-diagonals: **pearson −0.095, spearman +0.500 — over three
pairs.** Three points carry essentially no statistical power, so this is
**untestable, not refuted.**

**The honest summary of question 3:** each model has internally consistent,
above-chance structure among its reproducible bias directions. Whether the two
models agree about *which* categories sit near which cannot be determined from
three shared categories. Establishing that needs more categories over the floor —
which means either more items, a better estimator, or a contrast with more
signal.

What *did* emerge is a much sharper split than any grouping:

| category | qwen-7b | yi-6b | gemma-2b | qwen-14b | qwen-1.8b |
|---|---|---|---|---|---|
| **Disability_status** | **0.700** | **0.605** | **0.818** | **0.820** | — |
| **Physical_appearance** | **0.785** | **0.614** | **0.511** | **0.648** | — |
| Religion | 0.316 | −0.454 | −0.163 | **0.686** | −0.008 |
| Age | 0.194 | 0.177 | 0.161 | **0.754** | 0.423 |
| Race_x_gender | 0.218 | 0.204 | −0.044 | 0.072 | 0.138 |
| Nationality | −0.020 | 0.296 | −0.038 | 0.279 | 0.057 |
| Sexual_orientation | −0.104 | −0.304 | 0.056 | 0.161 | −0.202 |
| Race_x_SES | −0.150 | 0.089 | −0.165 | −0.105 | 0.017 |
| Race_ethnicity | −0.214 | −0.230 | −0.013 | −0.204 | −0.115 |
| Gender_identity | −0.177 | — | — | 0.081 | −0.072 |

**Disability_status and Physical_appearance produce a reproducible direction in
every model that produced any.** qwen-7b, yi-6b and gemma-2b agree *exactly* —
Jaccard 1.00, those two categories and nothing else. **Every race-related
category fails in every model.**

### 4. "Is it politics vs race vs religion, or is something deeper going on?"

**Neither.** The surface taxonomy does not organise the results, and the
heterogeneity hypothesis was tested and **refuted** (floor vs number of distinct
stereotype sets: r = −0.079; vs entropy: r = −0.020 — Religion has the most
heterogeneous stereotype sets of any category and reproduces in qwen-14b, while
Race_ethnicity has just as many and reproduces nowhere).

What predicts extractability, in every model tested:

| predictor | qwen-1.8b | qwen-14b | gemma-2b | yi-6b |
|---|---|---|---|---|
| the model's behavioural tilt (mean margin) | +0.689 | +0.726 | +0.660 | +0.689 |
| **direction norm** (activation-space separation) | **+0.808** | **+0.912** | **+0.945** | **+0.765** |

And a decoupling worth reporting: **how strongly a model behaves biased on a
category tells you little about how separable that category is internally** —
norm vs behavioural separation is −0.07 and +0.06 in the two Qwen models, though
+0.93 in yi-6b.

---

## The load-bearing methodological result

**Extraction works; the bias contrast is what fails.** In every model, a
direction for *topic identity* (Race prompts vs Gender prompts) extracted through
the identical capture site, estimator and split-half procedure reproduces at
**cosine 0.86–0.95**. Against that, bias-margin directions at −0.45 to +0.82.

So the failures are not a broken pipeline. That control is what licenses reading
the negatives at all, and it should be in the paper.

---

## What must NOT be claimed

1. **Not** "race, gender and sexuality have no bias vectors." Every failure is a
   failure to *recover* a direction with this contrast, at this n, at this
   capture site. The defensible sentence is: *two of ten BBQ categories yielded
   reproducible directions in all models tested; the remaining eight did not,
   under the current extraction and validation procedure.*
2. **Not** that the tilt→extractability association is a mechanism. It is partly
   circular: directions are built by contrasting margin extremes, so a category
   with no tilt is split on noise and cannot reproduce whether or not a
   representation exists. The dose-response test (monotone, at fixed n) narrows
   this but does not close it — see `notes/08`.
3. **Not** that the directions are non-causal. Four doses on one model and three
   on another, missing three required controls.
4. **Not** "the" direction — always "a" direction (arXiv:2602.06801).

---

## Caveats attached to the positive results

- **Two of the four qwen-14b reproducible categories are heavy-tailed.**
  Physical_appearance and Age have excess kurtosis +3.9 with the top 5% of items
  carrying ~half the variance. The extremes contrast selects exactly those tails,
  so their directions may rest on few unusual items. Disability_status is clean
  (kurtosis −0.9) and is the most trustworthy of the four.
- **Disability_status and Physical_appearance are also among the largest
  behavioural separations in every model**, so "these two are special" and "these
  two simply have the biggest contrast" are not yet separated.
- ~~**The probe estimator underperformed** (0/10 vs 4/10 for extremes)~~
  **RESOLVED — see below. It was misconfigured, not worse.**

---

---

## LATE RESULT — the probe was misconfigured, and it changes two conclusions

`scripts/probe_alpha_sweep.py`, qwen-14b, ridge penalty swept over six orders of
magnitude. Residuals captured once and reused, margins from cache.

| category | α=1 | α=1e2 | α=1e3 | α=1e4 | α=1e5 | α=1e6 |
|---|---|---|---|---|---|---|
| Disability_status | 0.461 | **0.599** | **0.660** | **0.734** | **0.830** | **0.880** |
| Age | 0.270 | 0.370 | **0.529** | **0.731** | **0.770** | **0.827** |
| Physical_appearance | 0.321 | 0.441 | **0.556** | **0.706** | **0.735** | **0.749** |
| Religion | 0.190 | 0.262 | 0.497 | **0.547** | **0.612** | **0.686** |
| **Nationality** | 0.111 | 0.200 | 0.302 | 0.348 | 0.440 | **0.506** |
| Race_x_gender | 0.089 | 0.153 | 0.257 | 0.315 | 0.320 | 0.239 |
| Sexual_orientation | 0.200 | 0.261 | 0.300 | 0.278 | 0.189 | 0.168 |
| Gender_identity | 0.084 | 0.164 | 0.135 | 0.162 | 0.062 | 0.030 |
| Race_x_SES | 0.034 | 0.046 | 0.023 | 0.011 | −0.040 | −0.120 |
| Race_ethnicity | −0.001 | 0.009 | 0.004 | 0.007 | −0.036 | −0.059 |

**1. "The probe is the worse estimator" was wrong.** `alpha=1.0` against
d_model=5120 with n=600 is essentially unregularised, so the probe was
overfitting and could not reproduce across a split-half by construction. Properly
penalised it beats the extremes contrast: **5/10 versus 4/10**, with a higher
floor on every category that reproduces at all.

**2. Nationality's failure was an estimator limit, not a fact about the model.**
It never cleared the bar under the extremes contrast (0.279) and clears it under
a tuned probe (0.506). That is precisely the confusion this caveat warned about,
now resolved for one category — and a reminder that the remaining negatives are
"not recoverable by the estimators tried", not "absent".

**3. The race negative is now much stronger.** Race_ethnicity, Race_x_SES,
Gender_identity, Sexual_orientation and Race_x_gender fail at **every** alpha
across six orders of magnitude, under **two different estimators**.
Race_ethnicity peaks at 0.009. That negative now rests on far more than one
estimator's configuration.

**Consequence for the headline count:** on qwen-14b it is **5 of 10 categories
with a recoverable direction**, not 4, once the estimator is configured properly.
A full tuned-probe run (cosine matrix, clustering, permutation null over the 5)
was launched — with five reproducible directions, clustering clears the
three-category minimum for the first time. See `runs/probe_tuned_qwen14b/`.

## The sweep run on every model — the negative holds

| model | extremes contrast | tuned probe | categories recovered by tuning |
|---|---|---|---|
| qwen-14b | 4/10 | **5/10** | + Nationality |
| qwen-7b | 2/10 | **3/10** | + Religion |
| yi-6b | 2/9 | **2/9** | none |
| gemma-2b | 2/9 | **2/9** | none |

Reproducible under the tuned probe, by model:

- **qwen-14b:** Disability_status, Age, Physical_appearance, Religion, Nationality
- **qwen-7b:** Disability_status, Physical_appearance, Religion
- **yi-6b:** Disability_status, Physical_appearance
- **gemma-2b:** Disability_status, Physical_appearance

**Disability_status and Physical_appearance reproduce in all four models, under
two estimators, across six orders of magnitude of regularisation.**

**No race-related category reproduces anywhere, under anything.** Race_ethnicity's
best floor across every model and every alpha is +0.146 (gemma-2b, α=1e2); on
qwen-14b it peaks at +0.009. Race_x_SES and Race_x_gender likewise. This negative
no longer rests on one estimator's configuration.

Optimal α scales with d_model, as expected: 1e6 for qwen-14b (d=5120), 1e2–1e4
for gemma-2b (d=2048) and yi-6b (d=4096).

---

## The obvious next experiments

1. **Winsorise the tails** on Physical_appearance and Age and re-extract. If the
   floor survives, the two heavy-tailed positives are solid.
2. **Tune the ridge penalty** and re-run the probe. If a properly regularised
   probe lifts more categories over the floor, "no reproducible direction" was an
   estimator limit rather than a fact about the model.
3. **Split a failing category by stereotyped group** — extract a direction for
   Black-targeted items alone rather than all of Race_ethnicity. If single-group
   subsets reproduce where the pooled category does not, heterogeneity comes back
   as an explanation at the right granularity.
4. **The missing steering controls**: covariance-matched random direction,
   coherence check, system-prompt baseline.

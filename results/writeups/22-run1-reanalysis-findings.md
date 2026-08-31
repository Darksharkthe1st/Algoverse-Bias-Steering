# 22 — Seven analyses of run 1 that had never been done

**Written 2026-08-29. No GPU. Everything here comes from files already in
`runs/`.** Reproduce with:

```
python -m scripts.run1_reanalysis          # repo/scripts/run1_reanalysis.py
```

Output is also written to `runs/_reanalysis/run1_reanalysis.json`.

---

## Why these were possible at all

Defect **S5** says residuals were not cached, and that is true — so no new
direction can be extracted without a GPU. But three other things *were* saved and
nobody used them:

- `runs/*/direction_<Category>.npy` — **every fitted direction**, 46 of them,
  shape `(n_layers, d_model)`
- `report.json` → `extraction_floor.cosines` — **the ten raw split-half cosines**
  behind every floor, not just the summary
- `runs/_margins_cache/*.json` — **per-item margins and abstention margins**
- `report.json` → `floor_vs_n` — **a sample-size sweep**, run and never analysed

That is enough for seven analyses, three of which materially strengthen the paper,
and one of which caught an error I was about to make (§G).

---

## The three that belong in the paper

### F. The headline does not depend on which summary statistic you use

**This is the most useful thing in this document.** Run 1 gated on the `q05` of
ten split-half cosines; `notes/13` §4 switches run 2 to the mean. A reviewer is
entitled to ask whether `q05` was chosen to get the answer. It was not, and this
demonstrates it:

| statistic | cells ≥ 0.50 (of 46) | categories clearing in ≥2 models |
|---|---|---|
| min | 9 | Disability_status, Physical_appearance |
| **q05** (run 1) | 10 | Disability_status, Physical_appearance |
| median | 12 | + Religion |
| **mean** (run 2) | 13 | + Age, Religion |
| **bootstrap CI lower bound** | 10 | Disability_status, Physical_appearance |

- **Disability_status and Physical_appearance clear in ≥2 models under all five
  summaries**, including the most conservative one.
- **No race-related category clears 0.50 under any statistic.** Across 15 race
  model-category cells the maximum is **+0.384**, on the *most permissive*
  summary.
- **Only 3 of 46 cells change verdict** between the run-1 and run-2 statistics,
  and all three move fail → pass, so the run-2 choice is the more generous one.

### A. Every floor now has an interval — and one headline cell is shaky

The ten raw cosines make the interval recoverable, which closes **S1** for
run-1 data retroactively. Bootstrapping each cell's ten splits (20,000 resamples)
and asking how often a redraw flips its 0.50 verdict:

| model | category | q05 | P(verdict flips on redraw) |
|---|---|---|---|
| yi-6b | **Disability_status** | +0.605 | **27%** |
| qwen-7b | Religion | +0.316 | 11% |
| yi-6b | Nationality | +0.296 | 7% |

The other 43 cells are stable. **The 27% matters**: `yi-6b/Disability_status` is
one of the four cells behind "Disability clears in every model that produced a
direction." One of those four is close to a coin flip on redraw.

*This weakens the headline slightly and should be reported anyway.* It is
precisely the instability S1 predicts, measured rather than asserted, and a paper
that finds this in its own result and says so is more believable than one that
does not look.

### B. More data would not have fixed it

The first reviewer attack on any negative result is *"you needed more data."*
Run 1 swept n and stored the answer:

| model | category | n | **mean** floor | median floor | q05 |
|---|---|---|---|---|---|
| qwen-14b | Religion (clears the bar) | 172 | +0.654 | +0.658 | +0.532 |
| | | 240 | **+0.817** | +0.813 | +0.789 |
| | | **change at 1.40× data** | **+0.163** | +0.156 | +0.257 |
| qwen-1.8b | Race_ethnicity (fails everywhere) | 172 | −0.063 | −0.016 | −0.268 |
| | | 320 | **−0.061** | −0.085 | −0.217 |
| | | **change at 1.86× data** | **+0.002** | −0.069 | +0.051 |

**More data helps a lot where there is something to find, and does essentially
nothing where there is not.** The conclusion holds under all three summaries: on
Religion every one moves up substantially; on Race_ethnicity the mean is flat,
the median goes *down*, and the floor stays negative throughout.

> *An earlier version of this table quoted the median column while labelling it
> "mean floor". Caught on the verification pass. The change figures were always
> computed from means and were correct; the endpoint values were not. All three
> statistics are now shown so the reader can check rather than trust.*

> **Do not overstate this.** The sweep covers one category per model and only two
> values of n, so it cannot say where either curve plateaus. It *bounds* the
> attack; it does not refute it. An earlier version of this analysis fitted a
> log-linear curve to the two points and duly concluded that Race_ethnicity would
> need 10¹⁰⁰ items — a line through two points is not a fit, and that number is
> arithmetic dressed as evidence. It has been removed.

---

## The three worth knowing but not headlining

### C. What the design could actually have detected — the power analysis

`notes/14` §6.4 lists "a negative with a power analysis showing what effect size
the design could have detected" as one of four things that would lift this to
conference quality. It had never been done. It costs nothing:

Split-half SD across 46 cells: **median 0.145, 90th pct 0.211, max 0.242.**

| n_splits | 95% CI half-width (median sd) | (90th pct sd) |
|---|---|---|
| **10** (run 1) | 0.090 | **0.131** |
| 100 | 0.029 | 0.041 |
| **400** (run 2) | 0.014 | **0.021** |

**At run 1's `n_splits = 10`, two floors had to differ by roughly 0.26 to be
distinguishable** — most of the usable range of a cosine. The design was never
powered for fine distinctions. The coarse one it *can* make — control at
0.86–0.92 versus most categories below 0.5 — is the only one the paper should
lean on, and it is comfortably outside that resolution.

Resolving a 0.05 difference needs `n_splits ≈ 273`, which is affordable only on
cached residuals. That is the concrete argument for the caching requirement.

### D. Abstention: suggestive, not established

Defect **N3**: on 23.5% of qwen-14b items the model's top choice is *"can't
answer"*, yet the item is still ranked by a margin between two options it
rejected. Nobody asked whether that predicts which categories fail.

**Pooling across models gives r = −0.11 and hides the effect** — abstention is
mostly a *model* property (yi-6b averages 5.0% across categories, qwen-14b
23.9%), so pooling mixes a large between-model effect into a between-category
question. That is the exact confound that invalidated the correlation retracted
in `notes/16`. Within model:

| model | r(abstention, floor) | n | permutation p |
|---|---|---|---|
| gemma-2b | −0.422 | 9 | 0.260 |
| qwen-14b | −0.101 | 10 | 0.792 |
| qwen-7b | −0.178 | 10 | 0.621 |
| yi-6b | −0.536 | 9 | 0.136 |

Mean **−0.309**, negative in **4/4** models, sign test **p = 0.125**, smallest
individual p = 0.136.

**Suggestive, not established — say it exactly that way.** The mechanism, if
real: the contrast ranks items by a margin between two *named* options, so on an
item where the model prefers to abstain that margin is a difference between two
options it rejected — noise with a sign. A category with many such items builds
both poles partly from that noise.

It earns a sentence because it makes a **falsifiable prediction**: the
annotation-derived contrast never ranks anything, so abstention should not
degrade it at all. Run 2 tests that directly.

### E. Cross-model structure replication: still untestable

`notes/10` reports this as untested. The saved directions let the question be
asked over all categories: mean cross-model Pearson of the cross-category cosine
matrices is **+0.394** (qwen-14b vs qwen-7b highest at +0.680).

**That number should not be reported as replication.** It is computed over
cosines between directions most of which do not reproduce against themselves —
closer to a measure of shared noise than shared structure. The test that would
separate them is to restrict to category pairs where all four directions
reproduce, and **that cannot be run**: with 0–4 reproducing directions per model,
no model pair has three such category pairs.

So `notes/10`'s "untested rather than refuted" stands, and this reanalysis
confirms the reason rather than removing it.

---

## Paste-ready LaTeX

Appendices are free, so C, D and E can go there in full. F and A belong in the
main text; B fits either.

```latex
\paragraph{The result does not depend on the summary statistic.}
Run~1 gated on the $q_{05}$ of ten split-half cosines. Because that choice was
made by us, we report every reasonable alternative. Under the minimum, the
$q_{05}$, the median, the mean, and the lower end of a bootstrap 95\% interval,
Disability\_status and Physical\_appearance clear the bar in at least two models
in all five cases, and \emph{no} race-related category clears it in any of
them---the largest value over 15 race model-category cells is $+0.384$, obtained
under the most permissive summary. Only 3 of 46 cells change verdict between the
most and least conservative statistic, and all three move from fail to pass.

\paragraph{Every floor carries an interval, and one of ours is unstable.}
Bootstrapping the ten split-half cosines behind each floor ($20{,}000$
resamples) gives an interval for every cell and, more usefully, the probability
that a redraw would flip its verdict. Forty-three of 46 cells are stable. Three
are not, and one of those matters: \texttt{yi-6b}/Disability\_status clears at
$q_{05}=0.605$ but would fail on $27\%$ of redraws. Disability\_status is one of
our two positive results, so we state plainly that one of the four models
supporting it does so unstably.

\paragraph{More data would not have closed the gap.}
On a category that clears the bar (\texttt{qwen-14b}/Religion) increasing $n$
from 172 to 240 raised the mean floor by $+0.163$; on one that fails in every
model (\texttt{qwen-1.8b}/Race\_ethnicity) increasing $n$ from 172 to 320 moved
it by $+0.002$ and left it negative. The sweep covers one category per model at
two values of $n$ and so cannot locate a plateau; we report it as a bound on the
sample-size explanation rather than a refutation of it.

\paragraph{What the design could resolve.}
The split-half standard deviation has median $0.145$ and 90th percentile $0.211$
across 46 cells. At the ten splits used here, the 95\% interval on a mean floor
has half-width $\approx 0.13$, so two floors must differ by roughly $0.26$ to be
told apart. The comparison this paper rests on---a control at $0.86$--$0.92$
against categories below $0.5$---is far outside that limit, but no finer
distinction among the failing categories is supported.
```

---

## What this changes about the paper

1. **Add F.** It costs four sentences and removes a whole line of attack.
2. **Add the 27% flip on yi-6b/Disability_status.** It slightly weakens the
   headline and substantially strengthens the paper.
3. **Add B** where the sample-size objection is discussed.
4. **Use `paper/figures/floors_ci.pdf`** instead of `floors.pdf` — same data,
   with the intervals shown. Plotting a q05 over ten draws as a bare bar is the
   thing S1 objects to.
5. **C, D, E to the appendix.** Free space, and D's falsifiable prediction is a
   good future-work sentence.

---

## G. Where the directions live — and a mistake I made and caught

Added 2026-08-29. Reproduce with:

```
python -m scripts.layer_profile_analysis
```

### The question

`notes/19` §6.2 flagged that the primary layer is undeclared for all five of our
models, and proposed `round(0.47 × n_layers)` — the reference paper's own depth
fraction, borrowed from two models we do not run. Borrowing a constant from
another paper is exactly the kind of undeclared parameter this project keeps
getting burned by, so I tried to measure it from the 46 saved directions.

### The mistake

The measurement says every one of the 46 directions peaks at the **final** layer,
with **zero variance**, and has a norm-centroid depth of **0.829**. My first
conclusion was: 0.47 is badly wrong, use 0.83.

**That conclusion is wrong and I caught it before it went anywhere.** Two checks
kill it:

| check | result | meaning |
|---|---|---|
| Is the norm profile monotonic in depth? | **97.5%** of layer-to-layer steps increase | that is what residual-stream scale growth looks like by itself |
| Is the profile *shape* different for directions that reproduce vs those that do not? | cosine between the two mean profiles **≥ 0.9991** at every depth | a real direction and a noise direction have indistinguishable profiles |

A transformer's residual stream accumulates contributions, so its norm grows with
depth, and a per-layer difference of means inherits that growth for free.
**A statistic that cannot tell signal from noise cannot locate signal.** The
0.829 is a property of the architecture, not of bias.

### The measurement that does work

Mean |cosine| between category directions at each layer. Cosine ignores
magnitude, so anything surviving here is not a scale artifact:

| model | layers | peak layers | peak depth |
|---|---|---|---|
| **qwen-14b** | 40 | 22, 23, 24 | **0.56–0.62** |
| **qwen-7b** | 32 | 17, 18, 19 | **0.55–0.61** |
| yi-6b | 32 | 23, 24 | 0.74–0.77 |
| gemma-2b | 18 | 14, 15, 16 | 0.82–0.94 |
| qwen-1.8b | 24 | 1, 13, 15 | noisy |

The two models with the most reproducible directions — qwen-14b and qwen-7b —
both peak at **depth 0.55–0.62**, near the borrowed 0.47 and nowhere near 0.83.

**Verdict: keep 0.47, but justify it from this profile rather than from the
reference paper, and keep the all-layer summary as the headline.**

### This also qualifies the N1 closure — worth knowing before run 2

`notes/13` §8 fixes the layer summary as a **norm-weighted mean**, justified on
the grounds that *"per-layer norms span orders of magnitude, so an unweighted
median treats a near-zero-norm layer as equal to the highest-signal one."*

That justification assumes **high norm implies high signal**. The shape
comparison above shows it does not: reproducing and non-reproducing directions
have the same norm profile to a cosine of 0.999. Weighting by a near-monotone
function of depth makes the headline effectively a late-layer read.

**The practical impact is small** — `notes/12` N1 measured norm-weighted mean vs
unweighted median to differ by ≤ 0.033 — so this is a flaw in the *justification*,
not in the number. But:

1. the justification should be restated honestly in the paper and in `notes/13`;
2. both summaries should keep being reported, which `notes/13` already requires;
3. **it is a live question for run 2**, where residuals will be cached and the
   per-layer split-half cosine can be computed directly. Then "which layer
   carries signal" becomes measurable rather than inferred, and the weighting can
   be chosen on evidence.

### Why this section is in the notes at all

Because the first version of this analysis produced a confident, plausible,
wrong answer — and it was caught by asking one question: *would this statistic
look any different if there were no signal at all?* That is the same question
that catches the option-list confound, the first-mention parser, and the
specificity control's first two rule attempts. It is the most reusable thing in
this project.

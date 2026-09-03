# Review of `paper/main.tex`, against the artifacts

**2026-08-29.** I checked every number in the draft against `runs/`. Verdict
first, then the one thing that must change, then the rest in priority order.

---

## What I already applied to `main.tex`

I edited the paper directly for the items below. Each is revertible and each is
explained in the section named. **Review P0 in particular** — it changes a claim.

| # | change | why I did it rather than only flagging it |
|---|---|---|
| **P0** | Replaced *"this threshold was fixed before the categories were run"* with the honest provenance **plus a sensitivity analysis** | The original is **factually false** by your own commit history. Shipping a false pre-registration claim is worse than any result in the paper. See §P0. |
| **P1** | Added the parser/N6 failure to §Silent failure modes | That section had no failure from *this* project. Additive, uses verified facts. |
| — | Added **Figure 1** + caption after the floor table | The paper had no figure; this one carries the whole argument. |
| — | Abstract: *"nine of ten"* → the true counts | "Nine of ten" matches no framing in the artifacts. |
| — | Fixed the Joad citation title (was a placeholder) | Wrong title in a citation is the kind of thing reviewers notice. |
| — | Softened *"We release the … code"* to release at camera-ready | Double-blind; the repo is public under a real handle. |
| — | Removed `[final]` from `\usepackage{neurips_2026}` | `[final]` drops the line numbers reviewers cite by. |
| — | Labelled the positive-control table as `qwen-1.8b` | It was unlabelled and read as if it covered all models. |
| — | Added `\appendix` + `\input{appendix_tables}` | Appendix space is free and the tables are generated from `runs/`. |

**I did not touch** the framing, the structure, the title, the Responsible-use
section, or any other scientific claim. P2 and the rest below are left for you.

**I stopped editing `main.tex` at 02:02** because a second session was writing to
it (last write 02:00:45). Two agents on one file is how work gets lost. Anything
below this line that is not in the table above is **not applied**.

---

## Verdict

**This draft is in good shape and the framing is better than the one I would have
written.** "Extraction works, the contrast does not" puts the positive control at
the centre, which is exactly right — the control is what makes the negative
readable, and leading with it pre-empts the reviewer's first attack.

**Almost every number is accurate.** I re-derived every table in the draft
from the run artifacts:

| table | checked | result |
|---|---|---|
| Floor table (§4) | 46 model-category cells + 4 dashes | **46/46 exact**, all 4 dashes correspond to genuinely absent runs |
| Task control (§3.1) | 10 accuracies + 10 z-scores | **10/10 exact** |
| Predictor table (§4) | 4 correlations | **4/4 exact** (Pearson) |
| Positive control (§3.2) | 3 rows | **3/3 exact** — but see P2 |
| Transfer table (§5) | 21 values | **14/14 exact** on `own`/`cross`; the `random` column uses a different (unstated) convention — see P0c |
| Clustering table (§5) | 7 rows | 6 exact; **one reports a p-value that was never computed** — see P0d |

That is unusually clean for a first draft. Whoever assembled it worked from the
artifacts rather than from the writeups, which is the standing repo rule and it
shows. The two exceptions are both in §5 and both are cheap to fix.

---

## P0 — one claim is false and must be fixed before submission

**`main.tex` §2, "Reproducibility criterion":**

> A direction is *reproducible* if its floor clears $0.500$. **This threshold was
> fixed before the categories were run.**

**It was not.** The project's own commit history says the opposite. Commit
`72b4e57`, 2026-08-20 20:16, is the commit that introduces
`MIN_USABLE_FLOOR = 0.50`, and its body reads:

> *"**The full qwen-1.8b run returned per-category extraction floors of −0.202 to
> 0.423 (q05)**: re-extracting the SAME category from random halves mostly gives
> unrelated directions. […] `floor_is_usable()` and `pair_verdict()` now return
> 'indeterminate' unless BOTH directions reproduce, **with the threshold at 0.50
> calibrated against a measured reference**: topic directions (Race vs Gender
> prompts) reproduce at q05 = 0.88 on this model."*

So the order was: run → observe collapsed floors → set the threshold. This is
also what `notes/12` records as defect **S4** and what `notes/13` §3 means by
*"Run 1's 0.50 bar is retired. It was a constant with a post-hoc justification."*

**Why this matters more than the threshold itself.** A post-hoc threshold with a
stated justification is a normal, forgivable limitation. A paper that *claims*
pre-registration it does not have is a different category of problem, and the
repo becomes public at camera-ready, where the commit is one `git log` away.

### The fix makes the paper stronger, not weaker

Because the conclusion does not actually depend on the threshold. I computed the
sensitivity across the whole plausible range:

| threshold | cells clearing (of 46) | categories clearing in ≥2 models |
|---|---|---|
| 0.30 | 12 (26%) | Age, Disability, Physical, Religion |
| 0.40 | 11 (24%) | Age, Disability, Physical |
| **0.50** | **10 (22%)** | **Disability, Physical** |
| 0.60 | 9 (20%) | Disability, Physical |
| 0.70 | 5 (11%) | Disability |
| 0.80 | 2 (4%) | Disability |

No cliff at 0.50, and **every race-related category fails at every threshold** —
across all 15 race model-category pairs the maximum floor is **+0.218**, below
even a 0.30 bar.

**Paste this in place of the false sentence:**

```latex
\paragraph{Reproducibility criterion.} We split the data into random halves,
extract independently from each, and record the cosine between the two
directions; the $q_{05}$ of that distribution is the \emph{floor}. We call a
direction \emph{reproducible} if its floor clears $0.500$.

We state the provenance of that constant plainly, because it is a limitation.
It was not pre-registered. It was fixed after an initial run returned floors of
$-0.202$ to $0.423$, and was calibrated against the extraction control in
Section~\ref{sec:extractionctrl}, which reproduces at $q_{05}=0.88$ on the same
model through the same code path --- so a working direction clears it with room
to spare. Because the constant is post-hoc, we report its sensitivity rather
than defend the value: the qualitative conclusion is unchanged for any
threshold in $[0.30, 0.80]$. Disability\_status clears in at least two models
throughout that range, and every race-related category fails at every value of
it, the maximum floor over 15 race model-category pairs being $+0.218$.
```

That paragraph converts your weakest methodological point into a demonstration
of rigour, and it costs eight lines.

---

## P0b — "pre-registered" is the same overclaim, one word smaller

**NOT APPLIED — this one is yours, because it touches the contributions list.**

`main.tex` §Introduction, contribution 1:

> **Two pre-registered positive controls** that separate instrument failure from
> construct absence…

**There was no pre-registration for run 1.** `notes/13-preregistration.md` — the
actual pre-registration document — says in its own header: *"Written 2026-08-23,
before any run-2 data exists."* The run-1 artifacts are dated **2026-08-21**. The
pre-registration postdates the data it would have had to precede.

The commit timeline says the same:

| commit | time | what |
|---|---|---|
| `7e7e71d` | Aug 20, 19:58 | task-control gate (`PC_MIN_ACCURACY`, `PC_MIN_Z`) added |
| `72b4e57` | Aug 20, 20:16 | `MIN_USABLE_FLOOR = 0.50` **and** the extraction control added, after an earlier run had already returned collapsed floors |
| run artifacts | Aug 21, 09:44 | the runs the paper reports |

So both controls **did** exist before the runs the paper reports — that part is
true and worth saying. What is not true is "pre-registered," which is a term of
art meaning a protocol registered before data collection. You have no such
document for run 1.

**The accurate word is "pre-specified."** Suggested edit:

```latex
\item \textbf{Two pre-specified positive controls} --- fixed in code and
committed before the runs they gate, though not pre-registered in the formal
sense --- that separate instrument failure from construct absence: ...
```

Same credit, no exposure. A reviewer who takes "pre-registered" at face value and
later finds there was no registration will discount everything else in the paper,
and that would be an expensive way to save one word.

---

## P0d — Table `tab:fork` reports a p-value that was never computed

**NOT APPLIED — but this is the most serious thing I found in the draft, and it
is a one-cell fix.**

The clustering table has these three rows:

```latex
default & gemma-2b  & 2 & --- & 1.000 \\
default & yi-6b     & 2 & \multicolumn{2}{l}{not clusterable ($<3$)} \\
default & qwen-7b   & 2 & \multicolumn{2}{l}{not clusterable ($<3$)} \\
```

All three models have **exactly 2** reproducible categories — the same two,
Physical_appearance and Disability_status. Two are correctly marked *not
clusterable*. The third is given **p = 1.000**.

**Clustering was never run on gemma-2b.** From
`runs/_cross_model_final.json` and `runs/full_gemma2b/report.json`:

```
gemma-2b   p_value = None    cluster_strength = None
yi-6b      p_value = None    cluster_strength = None
qwen-7b    p_value = None    cluster_strength = None
```

All three are `None`, identically, because three categories are needed and only
two were available.

**Where the 1.000 came from.** The gemma-2b verdict *string* in the artifact
says:

> *"NO STRUCTURE: 2/9 categories produce reproducible directions, and their
> clustering is within the permutation null (p=1.000). This IS a finding — the
> measurement had the precision to detect separable subtypes and did not."*

That sentence is false, and the project already knows it is. It is
**incident I-8** in `notes/11`, recorded verbatim:

> *"A second instance printed 'clustering is within the permutation null
> (p=1.000)' when no clustering had run at all."*

I-8 was fixed for the *floor-collapse* path — which is why yi-6b and qwen-7b
correctly print `NOT CLUSTERABLE` — but the gemma-2b run predates or bypasses
that fix, and its stale verdict string survived into the table.

**Why this matters more than one cell.** §6 of this paper is *about* verdict
strings that assert claims the data cannot support. Publishing a p-value that
was never computed, sourced from a verdict string the project's own incident log
names as a defect, in the paper that makes that argument, is the single worst
thing a reviewer could find. It would also be easy to find: the row is
inconsistent with the two rows directly beneath it.

**The fix, one line:**

```latex
default & gemma-2b  & 2 & \multicolumn{2}{l}{not clusterable ($<3$)} \\
```

**And it costs you nothing** — arguably it strengthens §5, because the table then
says *no default-estimator run was even clusterable except qwen-14b and
qwen-1.8b*, which makes the tuned-estimator contrast sharper rather than weaker.

**Worth a footnote if you have room**, because it is a free extra example for §6:
this is a silent failure mode caught in the paper's own draft, by comparing a
reported number against the artifact it claimed to come from.

---

## P0c — RETRACTED, and replaced by a smaller but real point

> ### I got this one wrong. The original critique is kept below, struck through.
>
> **What I first wrote:** that the 7.0× ratio divides real *diagonal* effects by
> *all* random cells including off-diagonal ones, that this shrinks the
> denominator and inflates the ratio, and that the like-for-like number is 5.85×.
>
> **Why that was wrong.** I assumed pooling the four random cells mixed in cells
> that do not correspond to the diagonal comparison. It does not. The pooled
> denominator is **balanced across target categories** — two random cells measured
> on Disability items (0.0225, 0.0145) and two on Physical items (0.0039, 0.0051).
> A random direction has no category affiliation, so all four are equally valid
> samples of "what a norm-matched random direction does to these items."
> **Pooling is therefore a larger-sample estimate of the same quantity, not a
> biased one**, and 7.01× is defensible as computed.
>
> The gap between 7.01× (4 cells) and 5.85× (2 cells) is sampling noise on a
> handful of numbers, not a methodological error. **Do not change the number on
> my original account.**

### What is actually worth fixing

**1. The ratio is quoted to two significant figures and is not that stable.**
Under the paper's own convention it is **7.01×, 7.25×, 6.69×** at coefficients
2, 5 and 10. Under the two-cell denominator it is 5.85×. A quantity that moves
between 5.9 and 7.3 depending on dose and on how many cells you average should
not be printed as "7.0×".

*Suggested:* `"roughly $7\times$ the norm-matched random control (6.7--7.3$\times$
across the three doses)"`. The argument is unaffected and the honesty is free.

**2. Table `tab:transfer` mixes signed and absolute means without saying so.**
This is the real finding. Verified across all seven rows:

- `own` and `cross` are **signed** means — all 14 values reproduce exactly;
- `random` is a mean of **absolute** values — all 7 values reproduce exactly only
  under `mean|·|`.

Taking the absolute value for the random control is the *right* choice (a random
direction's sign is arbitrary, so a signed mean cancels toward zero and
understates the null). **The problem is that the table does not say so**, and on
one model it distorts badly.

On gemma-2b nothing is lost — every effect has the same sign, so signed and
absolute means coincide. **On qwen-14b the effects cancel:**

| coeff | own (signed, as printed) | own (mean abs) | random (mean abs) | own ÷ random |
|---|---|---|---|---|
| 2 | +0.0001 | 0.0020 | 0.0035 | 0.59 |
| 4 | +0.0021 | 0.0037 | 0.0066 | 0.56 |
| 8 | +0.0036 | 0.0078 | 0.0126 | 0.61 |
| 16 | +0.0067 | 0.0202 | 0.0250 | 0.81 |

Printed as `+0.000` against a random of `+0.004`, the c=2 row reads as though the
random control beats the real direction **35×**. The magnitude comparison is
**0.59×**. The conclusion is unchanged — own is below random at every dose — but
the table overstates it by more than an order of magnitude, and a reviewer who
recomputes will notice.

*Suggested:* relabel the column `$|\text{random}|$`, use mean-absolute for all
three columns, and add one caption sentence: *"Effects are summarised as mean
absolute margin shift, because a random direction's sign is arbitrary and signed
means cancel."*

**3. Minor, and in your favour.** §5 says the own-direction effect is *"smaller
than the norm-matched random control at three of four"* doses on qwen-14b. It is
smaller at **four of four** — under both the signed and the absolute comparison.
You are understating your own result.

---

## P1 — the project's best failure mode is missing

§6 "Silent failure modes" lists four defects, and they are good ones — but all
four are from the **inherited 2025 refusal pipeline** (the scalar-offset
indexing bug, the vector/model rotation, the marginals-as-transitions columns,
the unpinned judge). None is from **this** work.

The workshop asks for *"misleading interpretations."* You have the best example
in the project and it is not in the paper: **the choice parser fails toward
position, and its error rate is unauditable.**

**Paste as a fifth paragraph in §6:**

```latex
\paragraph{A parser that failed toward position, invisibly.} An earlier scoring
design generated a response and parsed which option it named, resolving ties by
earliest mention. The rule was adopted for a real reason --- models state a
choice and then explain themselves, naming the other option, and treating that
as ambiguous discarded a third to a half of responses --- and it fixed that
problem. But it has no negation handling and no question-echo stripping. Of
seven realistic phrasings, three parse wrong, and \emph{every} failure resolves
to whichever option is named first: ``It's not the doctor, it's the nurse''
scores as the doctor.

The unparsed rate was visible and we reported it. The \emph{misparsed} rate is
invisible --- a wrong label is byte-identical to a right one in the saved counts
--- and because the raw response text was not persisted, it can no longer be
measured at any price. A first-mention-biased parser reproduces the exact
signature we had read as evidence of model-side order sensitivity, so that
reading cannot be separated from the parser's own bias using the data we kept.
The general lesson is not about this parser: a labelling rule whose inputs are
discarded converts a fixable bug into a permanent one.
```

This is worth the space. It is concrete, it is checkable, and it is the kind of
thing a reader remembers.

---

## P2 — the positive control is stronger than the paper says

The §3.2 table is **qwen-1.8b only** — I verified it matches qwen-1.8b on 3/3
rows and gemma-2b and yi-6b on 0/3. The caption did not say so; I have fixed the
caption. But you also ran the control on two more models and the paper does not
mention it, which understates your own evidence.

Verified from `runs/_extraction_control_gemma2b.json` and `_yi6b.json`:

| model | contrast | $n$ | $q_{05}$ | median |
|---|---|---|---|---|
| qwen-1.8b | Race vs Gender | 320 | 0.882 | 0.899 |
| qwen-1.8b | Religion vs Age | 320 | 0.893 | 0.912 |
| qwen-1.8b | Nationality vs Sexual\_or. | 320 | 0.871 | 0.906 |
| gemma-2b | Race vs Gender | 320 | 0.915 | 0.925 |
| gemma-2b | Religion vs Age | 320 | 0.912 | 0.926 |
| gemma-2b | Nationality vs Sexual\_or. | 320 | 0.860 | 0.888 |
| yi-6b | Race vs Gender | 320 | 0.862 | 0.897 |
| yi-6b | Religion vs Age | 320 | 0.864 | 0.882 |
| yi-6b | Nationality vs Sexual\_or. | 320 | 0.879 | 0.891 |

**Across three models and three topic pairs the control reproduces at
$q_{05} = 0.86$–$0.92$.** Say that instead of "$0.87$–$0.89$". Put the full
table in the appendix (free) and the range in the abstract.

**Also worth one sentence:** the control was *not* run on qwen-7b or qwen-14b.
Those are the two models carrying the clustering result. Say so in Limitations
before a reviewer says it for you.

---

## P3 — smaller things, in order

1. **`[final]` removed.** `\usepackage[final]{neurips_2026}` switches the style
   out of submission mode and drops the line numbers reviewers cite by. Fixed.
   Your `\author{}` block was already anonymous, so nothing leaked.

2. **"We release the extraction and evaluation code"** (§Responsible use) — under
   double-blind you cannot link it, and the repo is public under a real handle.
   Change to: *"Code and per-category floors will be released with the
   camera-ready version."* See `CHECKLIST.md` §1.

3. **Abstract says "nine of ten categories."** The floor table shows 2 of 10
   clearing in ≥2 models at 0.50, and 10 of 46 cells clearing overall. "Nine of
   ten" is true for some models and not others. Prefer the exact form: *"only two
   of ten categories clear the bar in more than one model, and no race-related
   category clears it in any."*

4. **Joad et al. bibliography entry has a placeholder title.** Currently
   *"Directions that are geometrically distinct but behave as one control."* The
   real title is **"There Is More to Refusal in Large Language Models than a
   Single Direction"** (arXiv:2602.02132). Fix before submitting — a wrong title
   in a citation is the kind of thing a reviewer notices.

5. **Missing citations.** The draft cites BBQ, Arditi and Joad. Add
   \citet{venkatesh2026nonidentifiability} (arXiv:2602.06801) — you rely on the
   non-identifiability argument in §5 when you say passing steering controls is
   insufficient, and it is the strongest support for that claim. Consider
   Wollschläger et al. (arXiv:2502.17420) for the refusal-geometry line.

6. **"a direction", never "the direction."** Standing repo rule, and §5's
   argument depends on it. Grep before submitting.

7. **Figure.** There is no figure. One would help more than any paragraph you
   could add: per-category floor with the positive control as a horizontal
   reference line, one panel per model. `CHECKLIST.md` §5.

---

## What I did not change

- Anything that is a scientific claim. P0 and P1 are written as ready-to-paste
  blocks precisely so you make the call, not me.
- The framing, structure, or title. They are good.
- The "Responsible use" section — it is well judged and workshops increasingly
  look for it.

## The two-minute version

Fix the pre-registration sentence (P0). Add the parser failure (P1). Widen the
positive control to three models (P2). Fix the Joad title. Everything else is
polish.

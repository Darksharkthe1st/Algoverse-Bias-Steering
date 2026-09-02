# 19 — Closing the open holes, and an audit of the run-2 plan

**Written 2026-08-28. Planning only; no GPU used, nothing rented.**

This document does three things:

1. Reports every contradiction found between the run-2 planning documents, with
   file and line, each one checked against code or data rather than reasoned
   about.
2. Closes the three holes named as blocking: the specificity-control threshold
   (§3), the steering evaluation (§4), and the judge-validation schedule (§5).
3. Names **five further holes** found while doing so (§6). Three are in the
   mandatory path and one would have invalidated the whole run.

**Precedence.** This file is additive. It supersedes nothing. Where it corrects a
number in `17` or `18`, the original stays and is marked. Reading order becomes
**17 → 18 → 19 → 13 → 14 → 11 → 12**.

**Nothing here changes the primary contrast.** That decision is closed. What
changes is what the *controls* are, what the *thresholds* are, and which
parameters were still undeclared.

---

## 1. Method

Every factual claim below was recomputed from an artifact or executed against the
code. Claims I could not verify are marked `UNVERIFIED`. Three checks came back
negative — I looked and found nothing — and those are reported in §2.9 rather
than dropped.

---

## 2. Contradictions between the planning documents

### 2.1 · C1 — `13` §11 names the void estimator, and the file's own header blesses it

**Severity: high. This is the one most likely to be read as current.**

`notes/13-preregistration.md:377-380`:

> **Exactly one primary analysis:** §3.1 applied with the `extremes` estimator at
> n=600 over 9 categories.

`extremes` is marked **VOID as primary** at `13:59-63`, and `13:485` makes the
annotation contrast primary over all matched pairs in 10 categories.

The trap: `13:17` declares *"Sections 3, 4, 7, 9, 10, 11 stand unchanged and
remain better than the reference paper's equivalents."* So the header explicitly
certifies §11 as current, and §11's closing paragraph names the void estimator,
the void n, and the void category count. A reader following the header's own
instruction lands on superseded text believing it is authoritative.

**Fix:** amend §11's last paragraph to name the annotation contrast, and strike
§11 from the "stand unchanged" list at `13:17`.

### 2.2 · C2 — residual storage is 9.6× the planned figure

**Severity: high. It breaks a pre-rental checklist item.**

`notes/14-run-plan.md:126-143` sizes residual storage from **n=600 × 9
categories = 5,400 items per model → ≈16 GB fp32.**

But `13:496` amended sampling to *"all matched pairs; minimum 32 per arm"*, and
BBQ ships **25,814 matched pairs = 51,628 items per category set**, verified by
counting every `.jsonl` in `repo/datasets/BBQ_Prompt_Sets/`.

| model | layers | d_model | `14` §1.2 says | actually |
|---|---|---|---|---|
| qwen-1.8b | 24 | 2048 | 1.06 GB | **10.15 GB** |
| gemma-2b | 18 | 2048 | 0.80 GB | **7.61 GB** |
| yi-6b | 32 | 4096 | 2.83 GB | **27.07 GB** |
| qwen-7b | 32 | 4096 | 2.83 GB | **27.07 GB** |
| qwen-14b | 40 | 5120 | 4.42 GB | **42.29 GB** |
| **total** | | | **11.9 GB** | **114.2 GB** |

With the topic-control residuals scaled the same way, **≈153 GB, not ≈16 GB.**

Three statements fall with it:

- `14:223` pre-rental checklist — *"16 GB free on the laptop for the sync
  target"*. Needs ~200 GB with headroom. (Checked: `C:` has 1,079 GB free, so it
  is satisfiable — but the checklist item as written would be ticked wrongly.)
- `14:140` — *"≈15 min at 20 MB/s"*. At 114 GB that is **≈1.6 hours**, and the
  10-minute continuous sync has to keep pace with the capture rate, not merely
  finish eventually.
- `14:143` — *"16 GB is 3% of [the 0.5 TiB SSD]"*. It is **≈28%**.

**Unstated implementation consequence:** the largest single array is
Race_x_gender on qwen-14b — `15,960 × 40 × 5120 × 4 B` = **13.1 GB in one
`.npy`**. The analysis phase does 400 split-halves on the laptop. That must be
`np.load(..., mmap_mode='r')`, or the analysis OOMs on a machine that never sees
the GPU. No document says so.

### 2.3 · C3 — the 12-hour kill criterion is calibrated on the void sampling

**Severity: high, and it is the only wall-clock guard the plan has.**

`14:233-235` derives the op count as *"margins 16,200 + generation 5,400 +
residual capture 5,400 + task control 4,050 + topic control 1,920 = **32,970**"*
— every term sized at n=600.

`14:43-58` correctly says money no longer matters but that **wall-clock still
governs**, with a 12-hour kill at `14:275`. Under the amended sampling, residual
capture alone is **51,628 forwards per model**, ~9.6× the figure the 5.7-hour
table rests on.

So the mechanism meant to force a re-plan is set against an estimate an order of
magnitude too small. **A run proceeding perfectly will trip it.**

**Fix:** re-derive `14` §5's hours from the amended sampling before renting, and
re-set the kill threshold from that. §0.3 already voids the dollar column; it does
not void the *hours* column, and it should.

### 2.4 · C4 — a retracted novelty claim is still live in `14`

`notes/17-reference-paper-and-contrast.md:194-197` instructs that the claim *"the
extraction floor is a validity check this literature skips"* is **false with
respect to the reference paper and must be removed everywhere it appears.**

`notes/14-run-plan.md:420-421` still carries it:

> the field currently reports cosines against zero.

That sentence sits in §6.2, the drafted paper framing — the worst place for it to
survive. Joad et al. Appendix C Table 8 is an extraction floor computed exactly
the way we compute ours (`17:76-95`).

Checked and clean: the claim does **not** survive in `notes/08`, `README.md`, or
`extraction-floor.html`. `notes/11:470` describes what the project did rather than
claiming novelty, and is fine.

### 2.5 · C5 — `16` carries no banner, and its live text contradicts `17`

**Severity: high. Verified by grep: `notes/16` contains no reference to N6, to
`18`, or to any suspension.**

`17:350-365` suspends the consistency/floor convergence claim: *"Do not report
that as converging evidence until the parser is validated."*

`notes/16-method-1-reexamined.md:34-42` still says:

> **The two methods agree.** … This is converging evidence and it strengthens the
> headline result. It should be reported.

`16`'s own retraction (`16:72-94`) retracts only the *Spearman correlation*, on
different grounds — mixed models, wrong estimator. The qualitative convergence
claim above it is untouched and unmarked.

The handover instruction *"read `16` WITH `18`; see caveat"* is doing all the
work, and the caveat lives in a different file. Anyone reading `16` alone reads a
live instruction to report a suspended claim.

**Fix:** put the banner in `16` itself.

### 2.6 · C6 — `README` describes a git state that is no longer true

`README.md:83-93` states the branch is unpushed:

| README says | actual |
|---|---|
| *"Both local, neither pushed"* | `jz/bias-taxonomy` tracks `origin/jz/bias-taxonomy`, **0 ahead / 0 behind** |
| *"pushed? **NO — deliberately**"* | pushed |
| *"Nothing has been pushed to GitHub"* | pushed |
| *"21 files / ~5,100 lines ahead of `origin/main`"* | **294 files, 143,175 insertions**, 16 commits |
| `README.md:78` — *"notes\ 12 documents"* | **20** |

The 294 / 143,175 figure is dominated by `third_party/bbq/additional_metadata.csv`
(86,157 lines), so "21 files / 5,100 lines" was probably counting hand-written
files only. Worth restating precisely rather than leaving two incompatible
numbers in the entry-point document.

### 2.7 · C7 — `17` §9.3 miscounts the metadata file

`17:385` says `additional_metadata.csv` *"carries `target_loc` (86,157 rows)"*.

The file has **86,157 lines** but **58,556 CSV records** — several fields contain
embedded newlines. Immaterial to the design; it is a cited number, and the project
rule is that numbers trace to an artifact.

### 2.8 · C8–C17 — the rest of the register

Each is expanded where it is closed; listed here to keep the register in one place.

| id | contradiction | closed in |
|---|---|---|
| **C8** | `13:500` fixes the primary layer for `gemma-2-9b-it` and `Llama-3.1-8B` — **neither is in the model list** (`14:38`). The mid-layer for our five models is undeclared. | §6.2 |
| **C9** | `13:505` and `13:508` still carry live rows conditional on the SAE stage, voided at `17:394` and `14:91`. The authoritative table contains rows for a cut stage. | §6.6 |
| **C10** | `13:320` says *"Confirm this is the same position our loader already used."* Nobody confirmed. **It is not.** `bbq_score.py:296` captures index **−1**; the spec requires **−2**. | §6.1 |
| **C13** | `13:506` adopts α ∈ {5,10,20,30,60} from a paper whose operator is `x + α·r`. The repo's operator (`steering.py:103-124`) adds `(α/n_layers)·r_layer` at **every** layer. Different quantity. | §4.2 (B-4) |
| **C14** | *"matched on `question_index`"* (`17:238`, `13:73`, `13:485`) is not a 1:1 key. | §6.3 |
| **C15** | `17:258` — the arms *"differ slightly in length"*. Measured: **2.0–2.3×**. | §3.1 |
| **C16** | `11:149` cites direction norms *"Frobenius 100→314"*. `runs/full_qwen14b/direction_norms.json` gives **100.3→504.8**. The "5×" in the same sentence is right (5.03×); the endpoints are not. | — |
| **C17** | `14:20-27` lists five per-model passes and omits the **topic-control capture** — yet `14:136` budgets storage for it and `14:272` makes it a step-1 kill criterion. The pass list cannot produce an artifact the kill criterion requires. | — |

### 2.9 · Checks that found nothing — reported, not padded

- **Live `0.50` usable-floor thresholds.** Grepped every note and the README. All
  surviving mentions are run-1 history, the superseded banner, or the *task
  control's* `accuracy ≥ 0.50` (`13:302`) — a different constant, correctly
  retained. No stale gate found.
- **The retracted novelty claim outside `14`.** Not present in `notes/08`,
  `README.md`, or `extraction-floor.html`. Already cleaned there.
- **`13` §§3, 4, 9, 10** — usability criterion, `n_splits=400`, split
  construction, permutation null. Re-read against `17` and `18`. Consistent with
  the amended design; they need no change beyond the arm-label substitutions
  `13:502-503` already make.

---

## 3. Hole (a) — the specificity control

### 3.1 The confound is much larger than `17` says

`17:258` calls it *"slightly in length and specificity."* Measured across all ten
categories from the shipped `.jsonl` files:

| quantity | value |
|---|---|
| disambiguated ÷ ambiguous, context characters | **2.22 – 2.65×** |
| disambiguated ÷ ambiguous, full prompt | **1.95 – 2.33×** |
| disambiguated contexts beginning with their ambiguous partner **verbatim** | **25,814 / 25,814 = 100.00%** |

The primary contrast is therefore, mechanically, *"prefix"* versus *"prefix + one
appended sentence."* The appended sentence is not content-free — it names the
social groups (`"The grandson was struggling with the app and not really
listening the advice from the grandfather."`) — but the length delta is
**2.2–2.6× and near-constant across categories**, which is precisely the condition
under which a length-driven direction would look identical everywhere.

That makes the specificity control the load-bearing test of the design, not a
formality. `17:287` already says *"This control is the whole experiment in
miniature"*; the measurement says it more strongly than the text does.

### 3.2 Verdict on the candidate rule: **reject**

> *Candidate: the control fails if the MEDIAN cross-category cosine exceeds the
> LOWEST within-category floor.*

Three objections, two of them fatal.

**(i) The bar moves with the data, in the wrong direction.** The threshold is the
*lowest* within-category floor — the worst-measured category sets the bar, so a
category that measures badly makes the control **easier** to pass. In run 1 the
lowest floors were negative (Race_ethnicity **−0.204**,
`runs/full_qwen14b/direction_norms.json`), and a negative bar is cleared by any
cross-category cosine whatsoever. **This is the S3/S4 disease — a criterion
defined by the quantity it gates — reintroduced in the act of removing it.** It is
the same objection the pre-registration itself makes to run 1's 0.50.

**(ii) It cannot separate the two hypotheses it exists to separate.** A high
cross-category cosine has two readings: *we are measuring sentence length*
(fatal), or *bias really is one mechanism* (the research question's positive
answer). **No statistic computed only from cross-category cosines of this one
contrast can distinguish them**, at any threshold. The control as specified is
therefore not decisive regardless of the number attached. This is a design defect,
not a threshold defect, and it is the real problem.

**(iii) The median is the wrong summary for a partial collapse.** With ten
categories there are 45 pairs. If five collapse together and five stand apart, the
median lands mid-range and returns a muddy verdict where the informative answer is
*"these five collapsed."*

### 3.3 What replaces it — measure the confound instead of inferring it

**A-1 · Build a pure length direction, inside one arm.**

Within the ambiguous arm alone, context length already varies a lot — measured
p90/p10 of **1.55–3.42** per category, comparable to the 2.1–2.3× between-arm gap.
So:

```
d_len(C) = mean(resid | C, ambig, context length in TOP tercile)
         − mean(resid | C, ambig, context length in BOTTOM tercile)
```

Both terms are ambiguous items, so `context_condition` is held exactly fixed and
length is the only systematic difference. Pool across categories for `d_len_bar`,
the best available estimate of *"what reading a longer context does to the
residual stream."*

**A-2 · The decision rule — no constant, each category judged against itself.**

> ### Two earlier versions of this rule were wrong, and the pilot is what showed it
>
> Both are kept below, marked, because the reason each failed is the useful part.
>
> **Rejected v1 — `|cos(d_C, d_len_bar)| >= CI_lower(own floor)`.** It can never
> fire. A large length confound is a big, stable, *shared* component, so it
> inflates the floor by as much as it inflates the alignment. On the pilot's
> planted pure-length data — a direction with **no bias content at all** — it read
> `|cos| = 0.92` against a floor CI-lower of `0.966` and returned **PASS**. An
> entirely artifactual direction, certified clean. Comparing two quantities driven
> by the same cause is not a control.
>
> **Rejected v2 — "does C still beat its negative control once `d_len_bar` is
> projected out?"** Better, and causally the right question, but **attenuated**:
> `d_len_bar` is itself only measured to a self-floor of ~0.75, so projecting it
> out removes only the part of the confound that was successfully estimated. On
> the same pure-length data the projected floor was still `0.805` against a
> control of `0.378` — reported as reproducing. *You cannot fully project out a
> confound you can only estimate noisily.*
>
> Both are still **computed and reported** as descriptive numbers. Neither is the
> decision.

**CURRENT RULE.** The specificity control **fails for category C** iff

```
| cos( d_C , d_len_bar ) |  >=  sqrt(  CI_lower(floor of C) × CI_lower(floor of d_len_bar)  )
```

The right-hand side is the **largest cosine two noisy estimates of the same
underlying direction could be expected to show**, given how well each reproduces
against itself. If the observed alignment reaches that ceiling, the two are
indistinguishable at the precision available — the direction *is* length.

No constant appears anywhere: both terms are floors this pipeline already
computes, and the geometric mean is the standard correction for attenuation by
measurement error. The control **fails overall** if the median category fails.

This still fixes objection (i): each category is compared against **its own**
floor, so a badly-measured category can no longer lower the bar for the others —
it only fails itself.

**Measured on the pilot's three planted scenarios** (`scripts/pilot/`, 20 pairs ×
2 categories, torch-free, seconds):

| planted truth | specificity control | cross-category median cos | correct? |
|---|---|---|---|
| distinct categories, no confound | **PASS** | +0.057 | yes |
| categories collapsed onto one, no confound | **PASS** | +0.926 | yes — real structure, just not category-specific |
| **pure length, no bias content** | **FAIL** | +0.987 | yes |

**A-2b · The two instruments answer different questions.** The pilot forced this
distinction and it is worth stating plainly, because the candidate rule in §3.2
conflated them: the **specificity control** catches a direction that is an
*artifact*; the **cross-category matrix** catches *category collapse*. A high
cross-category cosine is equally consistent with "bias is one mechanism", which
is a **result**, not a defect. Neither instrument substitutes for the other, and
that is exactly why the candidate — which tried to detect length using
cross-category cosine — could not work at any threshold.

**A-3 · The convergent test that settles objection (ii).**

Every BBQ row carries **both** `context_condition` and `question_polarity`, and
the 2×2 is **perfectly balanced in all ten categories** (verified: e.g. Age
920/920/920/920). One residual capture therefore yields both contrasts, and the
polarity arm costs **zero extra GPU**.

And the polarity contrast has essentially no length confound. Measured over all
25,814 polarity pairs:

| | ambig-vs-disambig | neg-vs-nonneg |
|---|---|---|
| context identical between arms | 0% | **100.00%** |
| answer options identical | 100% | **100.00%** |
| full-prompt length ratio | 1.95 – 2.33× | **0.985 – 1.040** |

`17:270` already notes the polarity contrast is *"even more tightly matched."* The
measurement says it is not merely tighter — it is **exact** on context and options
and within 4% on length. It is currently demoted to *"a comparison, not the
headline"* (`17:273`). **Its real job is to be the length-clean twin of the
primary**, and the plan does not use it that way.

Run the identical floor and cross-category machinery on both, and read the 2×2:

| primary (length-confounded) | polarity (length-clean) | reading |
|---|---|---|
| cross-cat **high** | cross-cat **high** | **bias is one mechanism.** A real unitary answer, not an artifact |
| cross-cat **high** | cross-cat **low** | **the primary is measuring length.** Design fails; stop and re-plan |
| cross-cat **low** | cross-cat **low** | **bias is several mechanisms.** H1's target result |
| cross-cat **low** | cross-cat **high** | anomalous. Report, do not interpret, stop |

This is the piece the current plan lacks entirely, and it is the only thing that
tells "one mechanism" apart from "no mechanism."

**A-4 · Sensitivity check on the control itself, pre-declared.** `d_len_bar` must
reproduce against its own split-half floor. If the length direction is itself
noise, A-2 compares against noise and is vacuous. Declared now so it is not
discovered later.

A-4 turned out to be load-bearing rather than a formality: its floor **is** the
`CI_lower(floor of d_len_bar)` term inside A-2's ceiling. So the control
automatically becomes *harder* to trip when the confound is poorly estimated,
which is the correct behaviour, and it happens without a constant.

**Build `d_len_bar` inside each category and then pool. Never rank a pooled,
cross-category set by length.** The pilot showed why: ambiguous
Disability_status contexts average 106 characters and Physical_appearance 134,
so a pooled "long" tercile came out **9:4 Physical** and the "short" tercile
**10:3 Disability**. The resulting "length" direction was substantially
`d_Physical − d_Disability` — **a category direction wearing a length label.**
That is defect N4's disease in a new place: *a contrast built by ranking a
pooled, topic-heterogeneous set encodes topic, not the ranked variable.* Ranking
inside a category holds topic fixed, so length is what varies.

**A-5 · Pre-declared remedy, so a failure is not a dead end.** If A-2 fails,
report the primary a second time with the length component removed:

```
d_C^⊥ = d_C − ( d_C · d̂_len_bar ) d̂_len_bar
```

recompute floors and cross-category cosines on `d_C^⊥`, and report **both**, never
the projected one alone. **Stated up front:** if the bias signal is itself largely
shared across categories, this projection removes signal along with confound, so
`d^⊥` results are a *lower bound* and can never be the headline. Declaring the
remedy now is what stops it being a post-hoc rescue later.

---

## 4. Hole (b) — the steering evaluation

**The question:** what counts as a stereotyped response, and what scores it? Is
BBQ's likelihood scoring sufficient with no judge?

### 4.1 Verdict: sufficient for the headline, and better than a judge — but not sufficient alone

**Sufficient, on three grounds, each checked against the code:**

1. `score_answers` (`repo/src/bias_steer/bbq_score.py:92-126`) already accepts
   `fwd_hooks`. Its docstring at lines 100-102 states the intent outright:
   *"steering can be measured as a SHIFT IN THE MARGIN, with no generation and no
   judge anywhere in the loop."* Run 1 used exactly this path and produced clean
   numbers (`runs/full_qwen14b/transfer_test_norm_c*.json`).
2. It is deterministic, reproducible, carries **no judge version**, and has **no
   position bias** — the three properties `18` spends its whole length worrying
   about. It sidesteps N6 rather than mitigating it.
3. It answers *"what counts as stereotyped"* with **BBQ's own answer key**
   (`target_loc`), not our rubric:
   `margin = logP(stereotyped person) − logP(other named person)`. That is the
   strongest available provenance, and it is what incident I-4 (`11:163-171`) says
   to prefer.

**Not sufficient alone, on one ground.** `11:410-412` requires a coherence check:
*"Without a coherence check on the generated text, 'the model got worse at
everything' is not excluded and no result is causal."* A margin shift cannot
distinguish *the direction moved bias* from *the direction broke the model*. At a
dose large enough to move a margin, models degenerate — and likelihood scoring
will happily report a large, clean margin shift from a model emitting garbage.

### 4.2 The specification

**B-1 · Primary steering metric — judge-free.** Margin shift under steering, via
`score_answers` with `fwd_hooks`, on the controlled test set below. No generation,
no judge, no parser in the primary path.

**B-2 · The controlled test set.** Port Joad et al. §2.7 (`17:118-133`). Their
cells are (harmful / benign) × (refused / complied). Ours:

```
(ambiguous / disambiguated)  ×  (unsteered margin > 0 / unsteered margin < 0)
```

Equal n per cell, so **the unsteered model is stereotyped on exactly 50% of the
set by construction** and any movement is attributable to the intervention — the
property that makes their design elegant. Built from cached run-2 margins; costs
nothing.

**B-3 · Success criterion — absent from every document, declared here.**

No file states what makes the steering stage succeed or fail. `13:506` and
`14:81-89` specify the *operator* and the *test set* and stop. Two separate,
separately falsifiable claims:

> A direction **steers** iff its margin shift on its own category exceeds that of
> a **covariance-matched random direction** at the same dose, with disjoint
> bootstrap CIs over items.
>
> A direction **steers specifically** iff its shift on its own category exceeds
> its shift on every other category, again CI-disjoint.

This is not a formality. Run 1's artifacts say the *first* claim is already in
doubt. Comparing each category's own direction against a **norm-matched random
direction on the same category** (`runs/full_qwen14b/transfer_test_norm_c*.json`,
`runs/full_gemma2b/...`):

| model | category | real ÷ random, across the dose sweep |
|---|---|---|
| qwen-14b | Physical_appearance | 2.6 – 4.4× — clears, consistently, monotone |
| qwen-14b | **Age** | **0.05 – 0.30× — the random direction moves the margin 3–20× *more***, and the real direction pushes the *wrong way* at every dose |
| qwen-14b | Disability_status | 0.5 – 4.4×, sign flips across doses |
| qwen-14b | Religion | 0.4 – 1.1×, sign flips at c=16 |
| gemma-2b | Disability_status, Physical_appearance | 4.8 – 6.3×, consistent — but **sign inverted**, and at c=10 it wipes out 56% of the baseline margin |

Absolute sizes on qwen-14b are tiny: **0.03 nats at c=16 against a baseline margin
of 0.79** — a 4% shift. `README.md:253-254` records this as *"steering showed no
category specificity at any dose"*, which is accurate but understates it: on the
headline model, **three of four directions perform at or below their own random
control.**

Without a pre-declared criterion, run 2 reproduces this table and there is no rule
for calling it. With one, "no direction steers above its control" is a clean,
reportable negative.

**B-4 · The dose grid must be re-derived, not copied.** `13:506` adopts
α ∈ {5,10,20,30,60}. Those are doses for `x + α·r` with a single unit direction.
The repo's operator (`steering.py:103-124`) adds `(α / n_layers) · r_layer` at
**every** layer, so the same α is divided by 18 on gemma-2b and 40 on qwen-14b —
**a 2.2× different per-layer dose between our two extreme models, and neither
equals the paper's.** Run 1's own working range was α ∈ {2,4,8,16} (`11:406-408`).
Copying 60 across is a unit error.

Worse, `α / n_layers` is a *constant* perturbation at every layer, while residual
norms vary by orders of magnitude across layers and 21 of 40 layers carry under
10% of the direction's peak norm (`12:141`). The dose is dumped equally into
layers where the direction is signal and layers where it is noise.

> **Rule.** Express the dose relative to the residual: define `α_rel` such that the
> added vector's norm at the **primary layer** equals
> `α_rel × median ||resid|| at that layer`, with
> `α_rel ∈ {0.05, 0.1, 0.2, 0.4, 0.8}`. Dimensionless, identical in meaning across
> models, computable from cached residuals **before** any steering pass runs.
> Record the repo-unit coefficient each `α_rel` maps to, per model, in the
> manifest.

This is `11` §8.1's own rule — *"put the dose in the coefficient, never in the
vector"* — applied one level up, to the coefficient's units.

**B-5 · The coherence check, and the only place a judge is needed.** Generate at
each dose on a fixed 100-prompt subset, save verbatim, score coherence.

> **Pre-declared kill:** any dose whose mean coherence falls below the unsteered
> baseline's CI is **excluded from the steering result entirely, before its margin
> shift is read.**

This is where a judge earns its place, and it is a far easier judging task than
choice extraction — no option list, no position bias, no `target_loc`.

**B-6 · Ablation.** `x' = x − (x·r)r` appears at `13:506` with no metric and no
criterion anywhere. Same criterion as B-3, opposite expected sign.

---

## 5. Hole (c) — judge validation

### 5.1 Three things in the code that change the design

**c-i · `parse_verdict` has the same defect as `parse_choice`.**
`repo/src/bias_steer/judge.py:32-37` returns the label *"appearing earliest"* in
the text after `ANSWER:`. A judge replying `ANSWER: not the doctor, the nurse`
resolves to **the doctor**. **Fixing N6 with a judge, then parsing the judge's
reply with an earliest-mention rule, reintroduces N6 one level up.**

> **Required:** the judge emits a bare label, and `parse_verdict` **rejects** any
> reply containing more than one label after `ANSWER:`, counting it unparsed
> rather than resolving it. That converts a silent misparse into a visible count —
> the exact property `18:61-63` praises about the unparsed rate.

**c-ii · Temperature is never set.** `judge.py:92` calls
`create(model=model, messages=messages)` and nothing else, so it runs at the API
default. `18:121-122` requires temperature 0. Not implemented. No seed either.

**c-iii · The judge model is an unpinned alias.** `config.py:92` —
`model: str = "gpt-4o-mini"`. A rolling alias silently re-points. `18:132` requires
a pinned id. It must be a **dated snapshot string**, stored with every label.

### 5.2 C-1 · Order-swap qualification — runs first, before any judged number is read

- **Sample:** 200 responses, stratified 20 per category, seed 0, from the first
  model's generation pass.
- **Procedure:** judge each twice — option list in the item's order, then reversed
  — everything else byte-identical.
- **Statistic:** flip rate, with a Wilson 95% CI.
- **Pass/fail: the judge is disqualified if the flip rate's 95% CI lower bound
  exceeds 0.02.**

  *Why 0.02, and why n=200 — both from one requirement, not from convention.* The
  flip rate is a **pure instrument artifact**: same response, same question, only
  presentation order changed, so the true rate is **zero by construction**. The bar
  is set by what the instrument must resolve. The quantity being measured is
  person-consistency near 58% — an 8-point departure from 50%. An instrument
  contributing more than 2 points of flip contributes a quarter of the entire
  effect. And n=200 is the smallest n whose CI half-width (~2 points at p=0.02) can
  resolve the bar it is being tested against. Change one and the other moves.
- **If disqualified:** fall to the next judge on a pre-declared ordered list. If
  none qualifies, **the judge is dropped and only the deterministic parser is
  reported, with its accuracy stated.** Same shape as the ridge probe's `C` drop
  rule at `13:141-143`.
- **Also run it on `parse_choice`.** It will score a 0.00 flip rate. **That is not
  evidence the parser is sound** — see C-2 — and the report must say so, or a clean
  number gets read as a pass.

### 5.3 C-2 · The mirror-pair test — replaces `18` item 4, which is vacuous

`18:83-84` specifies: *"Feed it the same response with the option order swapped;
the label must not change."*

**Executed this session against the real function.** `parse_choice` returns an
identical label under **all six permutations** of the option list, and **0 of 7**
of `18`'s own sentences flip. The specified test passes 7/7 while the parser is
wrong on 3/7. **Running it would produce false assurance on the exact defect it
was written to catch.**

The reason is mechanical: `parse_choice` scans the **response text** for each
option and takes the earliest *in the response*. The order of the `answers` list
never enters, except in exact-position ties, which break by string length
(`bias_taxonomy.py:369`).

Which also means **`18:48-50`'s stated mechanism is wrong as written**:

> *"Swap the order in the prompt, the model gives the same answer in the same
> words, and the parser flips its label."*

The parser cannot flip on identical text. The real chain has three links: **prompt
order → the order the model echoes the options in its own response → the parser's
label.** `18`'s *conclusion* survives intact — question-echo phrasings do track
prompt order, so the label does follow position — but anyone implementing the fix
from the stated mechanism writes the wrong test, which is what happened.

**The test that bites:** mirror pairs — two responses of identical grammar with
the mention order reversed and therefore *different* correct answers. Measured on
four such pairs built from `18`'s own vocabulary:

| response | correct | parser |
|---|---|---|
| "It's not the doctor, it's the nurse." | nurse | **doctor** |
| "It's not the nurse, it's the doctor." | doctor | **nurse** |
| "Between the doctor and the nurse, I'd say the nurse." | nurse | **doctor** |
| "Between the nurse and the doctor, I'd say the doctor." | doctor | **nurse** |

**0 / 8 correct.** A first-mention rule gets *both* halves of every negation and
echo pair wrong, in mirror image — which is the signature, and it is invisible to
any test that does not pair.

- **Set:** 24 pairs (48 responses), 6 per phrasing family — plain statement,
  negation, question-echo, "neither" — with real option texts substituted.
- **Pass rule:** an instrument passes iff it labels **both** halves correctly on
  ≥90% of pairs. A first-mention rule scores ~50% on a balanced set by
  construction, and 0% on the negation and echo families.
- Pure CPU, free, and it belongs in `tests/test_bias_taxonomy.py` next to `18`'s
  seven sentences.

### 5.4 C-3 · Hand-labelling — how many, sampled how

Two strata, because they answer different questions.

| stratum | n | what it buys |
|---|---|---|
| **disagreement** | all, capped at **300**, proportional by category when capped | *relative* accuracy of parser vs judge; the high-yield set `18:151-155` identifies |
| **random control** | **200**, stratified 20/category, seed 0, drawn independently of agreement | *absolute* accuracy for both, with a real CI |

**The random stratum is not optional.** The disagreement set cannot estimate
accuracy on the agreeing majority, where **both instruments can be wrong
together** — `"Neither the doctor nor the nurse; it doesn't specify"` is exactly
that case, and both a substring rule and a careless judge can miss it.

- **Total ≤ 500 labels.** At ~15 s each, ~2 hours of one person's time.
- **Report** parser accuracy and judge accuracy, each with a Wilson 95% CI, from
  the random stratum, as first-class numbers beside every judged result —
  `18:76-79`: *"If accuracy is not reported, the labels are not evidence."*
- **Second labeller:** a 50-response overlap, requiring **Cohen's κ ≥ 0.8**. Below
  that the labels are not a ground truth and the audit is reported inconclusive.
  Declared now.
- **Blinding:** the labeller sees response text and option list only — never which
  instrument said what, never `target_loc`. Enforced structurally, the way
  `repo/wp25/README.md` does it for the battery sheets (*"no model column exists in
  these files"*).

### 5.5 C-4 · Where the judge's identity lives

Add to `report.json` (`14` §2 schema, `14:160-178`):

```
judge: { model_id, api_snapshot_date, temperature, seed,
         rubric_sha256, prompt_template_sha256,
         n_labelled, flip_rate, flip_rate_ci,
         parser_accuracy, parser_accuracy_ci,
         judge_accuracy, judge_accuracy_ci,
         disagreement_rate, kappa_overlap }
```

and write the labels themselves to `judge_labels.jsonl`:

```
{ item_id, response_sha256, parser_label, judge_label,
  judge_raw_reply, human_label|null, order_variant }
```

`response_sha256` ties every label to the exact verbatim text, so labels can be
recomputed years later from the stored responses. That is the whole lesson of N6.
**Never compare labels across differing `rubric_sha256` or `model_id`.**

### 5.6 C-5 · Schedule

| when | step | gate |
|---|---|---|
| pilot · CPU · free | mirror-pair + 7-sentence regression tests | must pass before FREEZE |
| pilot · CPU · free | `parse_verdict` multi-label rejection test | must pass before FREEZE |
| after model 1's generation pass | order-swap qualification, n=200 | disqualified → judge dropped, run continues |
| after all generation passes | dual labelling; disagreement rate | — |
| analysis · CPU | hand-labelling ≤500, κ check | required before any judged number is reported |
| analysis · CPU | recompute person-consistency with the qualified instrument | closes `18` item 5 |

Nothing here is on the GPU critical path except the generation pass, which is
already pass 3 (`14:25`).

---

## 6. Five further holes, found while closing the first three

### 6.1 · (d) The capture index is wrong in the code, and the spec never got confirmed

**Blocking. This alone would have invalidated the comparability claim.**

`13:317-322` requires capture at *"the chat-template token immediately preceding
the assistant's response (index −2)"* and adds: *"Confirm this is the same position
our loader already used; if it is not, the reference paper's position wins."*

**Nobody confirmed it, and it is not.** `bbq_score.py:296`:

```python
stack = torch.stack([cache[n][:, -1, :] for n in names], dim=1)
```

Index **−1**, the final token of the chat-formatted prompt.

There is also a genuine ambiguity to resolve, not merely a constant to change: for
a prompt built with `add_generation_prompt=True` (`bbq_score.py:81`), the *last*
prompt token already **is** "the token immediately preceding the assistant's
response." So Joad et al.'s "index −2" may refer to indexing a sequence that
already contains the response — in which case **our −1 is their −2 and the code is
right.** Which reading holds depends on the exact chat template, and it differs per
model family.

> **Required, and it is cheap:** tokenize one chat-formatted prompt per model,
> print the last five token strings, and record in the manifest which index
> corresponds to the decision state — **per model**. Pure CPU; needs only a
> tokenizer, not torch. Then set the index once and freeze it.

The pilot cannot certify the capture path without this, and this is the item most
likely to have silently produced an entire run at the wrong site.

### 6.2 · (e) The primary layer is undeclared for every model we actually run

**An undeclared free parameter in the headline number — the S4 defect class.**

`13:500` fixes the primary layer as *"gemma-2-9b-it: 20; Llama-3.1-8B: 15–16"*.
**Neither model is in the list.** `14:38` fixes the list at qwen-14b, qwen-7b,
qwen-1.8b, yi-6b, gemma-2b. The headline read therefore has no defined layer.

The paper's own choice is a *depth fraction*, not a layer number:

| their model | layer | fraction |
|---|---|---|
| gemma-2-9b-it | 20 of 42 | 0.476 |
| Llama-3.1-8B | 15–16 of 32 | 0.469 – 0.500 |

> **Proposed rule, fixed before data:** `primary_layer = round(0.47 × n_layers)` —
> reproducing the paper's own depth fraction rather than importing its layer
> numbers.

> ### ⚠ QUALIFIED 2026-08-29 by measurement — read `notes/23` before using this
>
> The 0.47 above is still the recommendation, but the reasoning behind it was
> weak (it was borrowed from two models we do not run) and I tried to replace it
> with a measured value. **That attempt failed, instructively.**
>
> All 46 saved directions peak at the **final** layer, with zero variance, and
> their norm-centroid depth is **0.829**. Taken at face value that would say the
> signal lives at the top and 0.47 is badly wrong. It is not evidence, for two
> measured reasons:
>
> - the norm profile is **97% monotonically increasing** in depth, which is what
>   residual-stream scale growth looks like on its own; and
> - the profile **shape is the same for directions that reproduce and directions
>   that do not** — cosine between the two mean profiles is **≥0.9991** at every
>   depth. A statistic that cannot tell signal from noise cannot locate signal.
>
> The scale-free version — mean |cosine| between category directions per layer,
> which ignores magnitude — peaks at depth **0.56–0.62 on qwen-14b and
> qwen-7b**, the two models with the most reproducible directions. That is near
> 0.47 and nowhere near 0.83.
>
> **So: keep 0.47 as the declared fraction, but justify it from the scale-free
> profile rather than from the reference paper, and keep the all-layer summary as
> the headline.** Reproduce with `python -m scripts.layer_profile_analysis`.

| model | n_layers | primary layer |
|---|---|---|
| gemma-2b | 18 | **8** |
| qwen-1.8b | 24 | **11** |
| yi-6b | 32 | **15** |
| qwen-7b | 32 | **15** |
| qwen-14b | 40 | **19** |

This differs from naive `n_layers // 2` for **all five** models, so it is a real
choice and must be declared rather than defaulted. All layers are still stored
(`13:328`); this fixes only which one carries the headline.

### 6.3 · (f) `question_index` is not a 1:1 matching key

**Would have silently produced a cross-product.**

`17:238`, `13:73` and `13:485` all specify the primary as *"matched on
`question_index`."* But `question_index` takes only **25–50 distinct values per
category**, not one per scenario. A literal join on it:

| category | true matched pairs | pairs from a `question_index` join |
|---|---|---|
| Sexual_orientation | 432 | **7,680** |
| Age | 1,840 | **187,136** |
| Race_x_gender | 7,980 | **1,954,800** |

The real scenario key is
**`(question_index, question_polarity, ans0, ans1, ans2)`**, which yields 1,828
keys in Age — **1,816 exactly (1 ambig, 1 disambig)** and 12 at (2,2). BBQ also
ships the pair as consecutive `example_id`s (0 / 1 in the verified Age sample),
which is the simplest correct join.

The *group means* are unaffected — BBQ is a complete factorial and the arm counts
are exactly equal in every category — so the **primary estimate is correct either
way**. What breaks is any code that materialises the pairing; and the claim at
`13:242-243` that *"every item in one arm has a partner in the other"* is true only
under the full key, and is (2,2) rather than (1,1) for a small fraction.

> **Also undeclared, and it matters more:** `13:502` says splits are *"stratified
> by arm."* But if the two arms of one scenario land in **different halves**, the
> two half-directions are estimated from **different scenarios**, and the floor
> absorbs scenario-sampling variance that has nothing to do with reproducibility —
> N5's disease one level up. **Proposal: split by scenario pair**, so both arms of
> a scenario always travel together. Then the two halves are independent *samples
> of scenarios*, which is what the floor is meant to be measuring.

### 6.4 · (g) Storage, wall-clock and the kill criterion

Covered as C2 and C3 in §2.2–2.3. Restated as a hole because all three need a
number before renting:

1. Correct `14` §1.2 to **≈114 GB fp32** (≈153 GB with the topic control), and
   `14:223` to **≈200 GB free**.
2. Re-derive `14` §5's hours from the amended sampling and **re-set the 12-hour
   kill from that number**.
3. Require `mmap_mode='r'` in the analysis path — the largest single array is
   **13.1 GB**.

`14:139-141` rejected fp16 because *"precision in the stored object is the one
thing that cannot be recovered later."* That reasoning is unaffected by the size
change and I am not proposing to revisit it — the constraint is laptop disk, and
1,079 GB is free, so fp32 stands.

### 6.5 · (h) The steering stage had no success criterion

Covered as B-3 in §4.2. Listed here so the register is complete: this was an
undeclared decision rule in a **mandatory** stage.

### 6.6 · Two stale rows in the authoritative table

`13:505` (*"plus `google/gemma-2-9b-it` if the SAE stage is in scope"*) and
`13:508` (the whole SAE row) are conditional on a stage voided at `17:394` and
`14:91`. §15 is *the* authoritative parameter table; it should not carry live rows
for a cut stage. Mark both void in place, per the keep-superseded rule.

---

## 7. What I did not do, and why

- **Did not re-open the primary contrast.** Closed decision. Everything above
  changes controls, thresholds and undeclared parameters only.
- **Did not change the ridge probe.** Still the recommendation to keep it as
  specified, with the `C` drop rule at `13:141-143`. Nothing found this session
  changes that.
- **Did not touch `repo/wp25/`.** Different workstream (WP-25 battery
  stratification), built 2026-08-28, untracked, and its own README says it needs
  its own branch and PR.
- **Did not push anything. Did not rent anything.**

---

## 8. The weakest point in the plan, after all of the above

Not the specificity threshold — §3 gives that a rule that cannot be gamed.

**It is that the design's mandatory control is the one most likely to fail, and
the plan currently reads its failure as a nuisance rather than as the result.**

The primary contrast is *prefix* versus *prefix + one sentence*, in 100% of 25,814
pairs, at a length ratio of 2.0–2.3× that is **near-constant across categories**.
If the cross-category cosines come back high, the honest reading is genuinely
ambiguous between "bias is one mechanism" and "we measured sentence length," and
§3's A-3 is the only thing in the design that can tell them apart.

That is why the polarity contrast should stop being a footnote. It is the only
length-clean measurement available, it costs nothing, and without it a high
cross-category result is uninterpretable.

**Second weakest:** the steering stage. Run 1's numbers say three of four qwen-14b
directions do not beat their own random control at any dose tested, and the stage
is mandatory. §4's B-3 makes that a reportable negative instead of an ambiguous
table — but the plan should expect the negative rather than be surprised by it.

---

## 9. The pilot — built, run, and what it found

`repo/scripts/pilot/`, driven by

```
python -m scripts.pilot.run_pilot --out runs/pilot
```

20 scenario pairs × 2 categories, seconds, no GPU, no model download, no network.

### 9.1 What makes it a pilot rather than a smoke test

The stub backend synthesises residuals with a **known planted structure**, so
every control has a right answer and the pilot asserts the control *produces*
it. The whole pipeline runs three times:

| planted truth | what it isolates | required verdict |
|---|---|---|
| `distinct` | orthogonal categories, negligible confound | specificity **PASS**, cross-category low |
| `collapsed` | categories share one direction, no confound | specificity **PASS**, cross-category **high** |
| `pure_length` | no bias content at all, length only | specificity **FAIL** |

Run 1 never did this. Its controls ran once, on real data, with no ground truth —
so a control that silently always passed would have looked exactly like a control
that worked. **Two of the controls in this plan did silently always pass**, and
that is how they were caught.

### 9.2 Tier 1 gate — all eight checks pass

```
  distinct     specificity=PASS  cross-category median |cos| = +0.057
  collapsed    specificity=PASS  cross-category median |cos| = +0.926
  pure_length  specificity=FAIL  cross-category median |cos| = +0.987

  [PASS]  specificity control passes when no confound is planted
  [PASS]  specificity control FAILS when the direction is pure length
  [PASS]  specificity control does not misfire on real-but-shared structure
  [PASS]  cross-category matrix separates distinct from collapsed
  [PASS]  verifier passed on all runs
  [PASS]  queue manifest reports all steps OK
  [PASS]  A-4 self-check finds a length direction when one is planted
  [PASS]  A-4 self-check reports none when none is planted
```

Test suite: **246 passed, 1 skipped, 4 xfailed.** The four xfails are N6's
documented defects, marked `strict=True` so that fixing the parser turns them
into failures and forces the markers to be removed. The defect cannot be closed
silently and cannot be forgotten.

### 9.3 What the pilot caught, in order

Every one of these was in code or in a rule that had already been written down
and reviewed, and none would have been visible without running it.

1. **A-2 v1 could never fire.** §3.3. A confound inflates the floor and the
   alignment equally.
2. **A-2 v2 was attenuated.** §3.3. You cannot fully project out a confound you
   can only estimate noisily.
3. **The pooled length direction encoded category, not length.** §3.3 A-4. N4's
   disease in a new place.
4. **`example_id` restarts at 0 in every BBQ file.** A residual cache keyed on it
   alone merges categories silently. With ten categories and 51,628 items per
   model that is a near-certain collision on every id, and the direction it
   produces looks entirely normal. Fixed by `pairing.item_key` →
   `"<category>:<example_id>"`.
5. **A pilot drawn from the head of a BBQ file has almost no length variation.**
   BBQ orders rows by scenario template, so the first 20 Disability_status pairs
   span 132–144 characters against 75–141 for the arm as a whole. Sampling is now
   evenly spaced across the file. *General lesson: a pilot sample that does not
   span the real distribution certifies nothing about a control that depends on
   it.*
6. **`notes/18` item 4's parser test is vacuous.** §5.3, and it is now an
   executable test rather than a claim.

### 9.4 What the pilot does NOT certify — tier 2 is not run

**The pilot is not green, and it must not be described as green.**

Everything above is torch-free. The following is written (`backends.HFBackend`,
`backends.probe_capture_index`) and **has not been executed**, because this
laptop has no torch, no transformers and no tokenizers — by design, per
`README.md` §4:

- tokenisation and the chat template,
- **the capture index**, hole (d) in §6.1 — the −1 vs −2 question,
- a real forward pass and the shape of what comes back.

That is precisely where the defect most likely to invalidate the run lives.
`HFBackend` therefore **refuses to construct** without an explicit
`capture_index`; a default there would be exactly the library-default failure
that incident I-2 cost a reversed conclusion.

`probe_capture_index(hf_id)` answers hole (d) in about ten seconds per model
using only a tokenizer — no weights, no torch. It should have been run before
run 1.

### 9.5 Where the closures live in code

| closure | file |
|---|---|
| scenario pairing, `item_key`, split-by-pair, length terciles | `scripts/pilot/pairing.py` |
| planted-truth stub, `HFBackend`, `probe_capture_index` | `scripts/pilot/backends.py` |
| floors with bootstrap CI, negative control, §3.3 specificity control, projection | `scripts/pilot/analysis.py` |
| queue runner with real exit codes and declared outputs | `scripts/pilot/queue.py` |
| post-run verifier — the termination gate | `scripts/pilot/verifier.py` |
| driver, confound checklist, gate P2 | `scripts/pilot/run_pilot.py` |
| N6 regression + mirror-pair tests | `tests/test_bias_taxonomy.py` (appended) |
| queue/verifier negative paths, pairing assertions | `tests/test_pilot_infrastructure.py` |

Written fresh rather than by editing `bias_taxonomy.py`: the shipped functions
produced every run-1 artifact, and changing them in place would make run 1
unreproducible. Three of them also carry defects the pre-registration closes
(S1, N1, N5), so run 2 needs different behaviour, not patched behaviour.

---

## 10. What I need from you before the pilot can go green

Four things, in the order they block.

1. **Torch, or a CPU box.** Tier 2 needs `torch` + `transformers` locally
   (~2.5 GB) or a cheap CPU instance. I did not install anything on your machine
   while you were asleep. *For hole (d) alone, `pip install transformers jinja2`
   is enough — no torch, no weights.*
2. **Hole (d), the capture index.** Once a tokenizer exists, this is a ten-second
   check per model and then a frozen decision. Nothing should be captured before
   it is answered.
3. **A ruling on the disputed numbers in `14`** — storage ≈114 GB not 16 GB, and
   the wall-clock table that the 12-hour kill criterion is calibrated against.
   I have not edited `13` or `14`; both are frozen and §2 lists what would change.
4. **The judge model id**, for §5.5. `config.py:92` defaults to `"gpt-4o-mini"`,
   a rolling alias. It needs a dated snapshot, and the choice is yours.

Not blocking, but worth a decision: whether the polarity contrast is promoted
from "a comparison" to the primary's control twin (§3.3 A-3). It costs no GPU
time, and without it a high cross-category result cannot be interpreted.

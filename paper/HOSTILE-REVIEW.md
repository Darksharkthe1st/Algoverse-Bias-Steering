# Hostile review of `main.tex`

**Written 2026-08-29.** Read as an unsympathetic reviewer at an interpretability
workshop, looking for the fastest route to "reject" or "weak accept". Ordered by
how much damage each attack does.

`notes/14` §6.3 has a version of this exercise, but it was written for run 1's
framing before this draft existed. This is against the actual draft.

**Overall: I would accept this, after the fixes below.** The framing is good and
§5 and §6 are unusually honest in a way that reads as competence rather than
weakness. Five attacks land. All five close from data you already have, and one
of them (A1b) is a single line.

*Corrected while writing this: my first draft of this review said "the numbers
check out." That was true of every table I had checked at the time and became
false when I checked the last one — see A1b. Almost all of them do check out
(46/46 floors, 10/10 task-control, 4/4 predictors, 3/3 steering ratios, 14/14
transfer own/cross values), but "almost all" is the accurate word — one
clustering cell is not a number at all.*

---

## A1 — "Your positive control is too easy." · **LANDS. Fix it, it's cheap.**

> *The extraction control is a topic-identity contrast — race-themed prompts vs
> gender-themed prompts. Topic is trivially linearly decodable from a residual
> stream; a bag-of-words model would find it. That your pipeline recovers topic
> at 0.86–0.92 tells me it isn't broken in some gross way. It tells me nothing
> about whether it could recover something as subtle as a bias direction. Your
> negative is therefore uninformative about anything harder than topic.*

**This is the strongest attack on the paper and the draft does not answer it.**
The control section (§3.2) argues the pipeline "recovers a direction it should
recover," which is exactly the thing the reviewer is conceding and then
dismissing.

**You already have the answer and you are not using it.** The same pipeline, on
the same contrast type, recovers **bias** directions in two categories, in every
model where those categories survived the behavioural task control:

| category | gemma-2b | yi-6b | qwen-7b | qwen-14b |
|---|---|---|---|---|
| Disability_status | 0.818 | 0.605 | 0.700 | 0.820 |
| Physical_appearance | 0.511 | 0.614 | 0.785 | 0.648 |

These are not topic contrasts. They are the *same* per-category bias margin that
fails on race, run through the *same* machinery. (qwen-1.8b is absent because
both categories failed its behavioural task control and were dropped before
extraction, not because they failed to reproduce.) So the instrument demonstrably
recovers directions of the construct class under test — not merely an easier one.

**That converts A1 from a hole into a two-tier control argument**, which is
stronger than what the paper currently claims:

- *tier 1* — topic identity at 0.86–0.92: the pipeline is not broken;
- *tier 2* — Disability and Physical at 0.51–0.82: the pipeline recovers **bias**
  directions when they are there;
- therefore the race-wide failure is not "our instrument is too blunt for this
  construct."

**Suggested addition to §3.2**, four sentences and no new data:

```latex
A reviewer may object that topic identity is trivially decodable, and so too
weak a control to license a negative about anything subtler. The bias contrast
itself answers this. The identical machinery, applied to the same per-category
margin that fails on every race-related category, recovers Disability\_status
in all four models where that category cleared the behavioural control
($0.605$--$0.820$) and Physical\_appearance in all four ($0.511$--$0.785$). The
instrument therefore recovers directions of the construct class under test, not
merely of an easier one, and the race-wide failure cannot be attributed to a
blunt instrument.
```

---

## A1b — "You report a p-value you never computed." · **LANDS HARDEST. One line.**

> *Table 5 gives gemma-2b a clustering p of 1.000 with 2 reproducible categories,
> and marks yi-6b and qwen-7b — also 2 categories — as not clusterable. Which is
> it? Either two categories can be clustered or they cannot.*

They cannot, and gemma-2b's clustering **was never run**: `p_value` and
`cluster_strength` are both `None` in `runs/_cross_model_final.json`, identically
to yi-6b and qwen-7b. The 1.000 came from a stale verdict *string*, which
`notes/11` already logs as **incident I-8**.

Full detail and the one-line fix in `REVIEW.md` §P0d. This is the most damaging
attack available against the draft, and the cheapest to close.

---

## A2 — "Ten splits, and you gated on a quantile of them." · **LANDS. You now have the answer.**

> *Your decision statistic is the 5th percentile of ten numbers, read as a point
> value. That has enormous sampling error. How do I know your `reproduces` column
> isn't noise?*

The draft's honest threshold paragraph helps but doesn't close this, because it
addresses the *value* of the threshold rather than the *precision* of the thing
being thresholded.

**Closed by `notes/22` §A and §C**, neither of which is in the draft yet:

- every floor now has a bootstrap 95% interval;
- **43 of 46 cells are stable to redraw**; the three that aren't are named,
  including `yi-6b/Disability_status` at a **27%** flip probability;
- the power analysis says the design resolves differences above ≈0.26, and the
  comparison the paper rests on — control 0.86–0.92 versus categories below 0.5 —
  is far outside that.

**Reporting the 27% yourself is worth more than hiding it.** A reviewer who finds
an unstable cell in a paper that claims stability discounts the whole thing; a
reviewer who is told about it by the authors, with the number, reads the paper as
careful. Paste-ready LaTeX is in `notes/22`.

---

## A3 — "You picked q05 to get this answer." · **CLOSED, but the answer isn't in the draft.**

Now fully answered by `notes/22` §F: under min, q05, median, mean and the
bootstrap CI lower bound, the two positives clear in ≥2 models in **all five**
cases and **no race category clears 0.50 under any**. Four sentences. Add them.

---

## A4 — "Your steering table mixes conventions." · **LANDS, but not the way I first thought.**

> *Table 4 prints `own` as a signed mean and `random` as a mean of absolute
> values. On qwen-14b at coeff 2 that shows +0.000 against +0.004, which reads as
> the random control beating the real direction by a factor of 35. Recomputed on
> a common convention it is 0.59. Which is it?*

**Both columns are individually defensible and the conclusion is unchanged** —
own is below random at every dose on qwen-14b under either convention. But the
table does not say the conventions differ, and on qwen-14b the effects cancel, so
the signed `own` column understates the real magnitude by up to 20×.

*I originally wrote this attack up as "the 7.0× ratio uses a mismatched
denominator." **That was wrong and is retracted** — the pooled random denominator
is balanced across target categories, so it is a larger-sample estimate of the
same quantity, not a biased one.* The surviving points are smaller: state the
convention, use one convention for all three columns, and quote the ratio as
"roughly 7×" rather than "7.0×" given it moves between 6.7 and 7.3 across doses.
Full detail in `REVIEW.md` §P0c.

---

## A5 — "Split-half instability means high variance, not absence." · **Defended.**

> *A direction could be a real but noisily-estimated thing. You have shown you
> cannot recover it reliably, not that it isn't there.*

The draft handles this correctly in Limitations — *"it does not say bias is
absent from the models"* — and the title (*"extraction works, the contrast does
not"*) makes the same distinction. No change needed. **Do not weaken this;** it is
one of the paper's better instincts and reviewers reward it.

---

## A6 — "Old, small models." · **Concede, already conceded.**

Qwen1.5, Yi-6B, gemma-2b, 1.8B–14B. The draft says so. Fine for a workshop, and
the cross-family replication is a genuine strength. One sentence noting that the
*method* is model-agnostic would help.

---

## A7 — "Why is split-half stability the right criterion at all?" · **Defended, could be sharper.**

The paper argues it from the noise-floor logic and cites the reference paper doing
the same. That is sufficient. It would be sharper still to note that
\citet{joad2026} use within-category stability as a *precondition* for their
distinctness claim — i.e. the criterion is not ours, and the field's own leading
example of this analysis already relies on it.

---

## A8 — "One capture site, one estimator." · **Conceded in Limitations.** Fine.

---

## The one thing I would add if there were space

Nothing. The paper is over its page budget already (`PAGE-BUDGET.md`), and A1's
fix is four sentences that pay for themselves several times over. If anything has
to go to make room, cut Table `tab:floors` — Figure 1 now shows the same 46
numbers better and the table survives in the appendix.

## Priority order

1. **A1b** — one line. A p-value that was never computed is currently in Table 5.
2. **A1** — four sentences, closes the strongest conceptual attack, uses data you
   already have.
3. **A4** — state the table's summary convention; quote “roughly 7×”, not “7.0×”.
4. **A3** — four sentences from `notes/22` §F.
5. **A2** — the 27% flip, and the power sentence.

All five are additions or corrections to text, none needs new data, and together
they take about an hour.

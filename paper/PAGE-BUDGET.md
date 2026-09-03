# Page budget

Re-measure any time:

    py paper/pagecount.py paper/main.tex

Heuristic: 575 words/page, 0.22 page/table, 0.35 page/figure. **Compile once and
check the real PDF** — this is an estimate, and at the deadline you want the
actual number, not mine.

## Where it stands (2026-08-29, 06:5x)

| | est. pages |
|---|---|
| body text (2887 words) | 5.02 |
| 4 tables | 0.88 |
| 1 figure | 0.35 |
| **total** | **6.25** |
| **limit** | **5.00** |
| **over by** | **1.25** |

References and appendices are excluded and do not count, so most of this is
solved by moving rather than deleting.

## Already cut (was 6.44)

- **Three of the six silent-failure-mode paragraphs** moved to a new
  Appendix~B (`Three further silent defects`): the vector/model rotation, the
  marginals-as-transitions columns, and the unpinned judge. The three that stay
  are the ones with artifacts behind them or the strongest content — the scalar
  index bug, the position-biased parser, and the selectivity rule.
- **The predictor table** became one sentence. It was ten numbers.
- **The extraction-control table** became one sentence. It was six numbers, and
  the all-models version is already in the generated appendix.

That recovered ~0.5 page. Structure verified after each edit: all `\ref`s
resolve to a `\label`, and table/figure/tabular environments all balance.

## The biggest remaining lever

**Table~\ref{tab:floors} and Figure~\ref{fig:floors} present the same data.**
The table gives per-category floors for 10 categories × 5 models; the figure
plots exactly those floors with the control band overlaid. Keeping both costs
**1.23 pages** for one result.

Pick one. My recommendation is **keep the figure, drop the table**:

- the figure carries the control band visually, which is the paper's whole
  argument, and a reader gets "every race category sits at or below zero" from
  it in one glance;
- the full numeric table already exists in the generated appendix
  (`make_appendix_tables.py` emits per-category floors for all models), so
  nothing is lost;
- it saves 0.88 rather than 0.35.

That alone lands you at ~5.4.

## Then, in order

1. **Delete the "Negative split-half cosines deserve comment" paragraph** (~55
   words). Its content is already the last two sentences of the figure caption,
   verbatim in substance. Pure duplication.
2. **Trim the positive-control table to the boundary rows** (~0.1 page). The
   argument needs the two that fail and two or three that pass, plus "the
   remaining categories pass at 55.3–62.0%; full table in appendix".
3. **Compress the Setup threshold paragraph** (~40 words). The provenance and
   the sensitivity range must stay — that is the P0 fix and it is load-bearing —
   but it can say the same thing in fewer words.
4. **Tighten the confound paragraph** (~40 words). The point is one sentence:
   norm predicts reproducibility, so an unnormalised transfer test would confound
   specificity with dose, and ours is normalised.

Doing the recommendation plus 1–2 lands at roughly **4.9**. That is too close to
the line to trust an estimate, so compile before deciding whether to do 3 and 4.

## What must not be cut

- **Both positive controls.** They are why the negative result is readable at
  all. Without them the paper is "we tried something and it did not work."
- **The threshold-provenance paragraph in Setup.** Claiming a pre-registration
  you do not have is the one mistake that would sink the paper, and the repo goes
  public at camera-ready where the commit is one `git log` away.
- **The forking-path section.** It is the most honest thing in the paper and the
  CFP explicitly asks for exactly this.
- **The cross-category and dose columns** of the transfer table. They are the
  entire argument that standard controls are insufficient.
- **The responsible-use statement.** Missing it is stated grounds for desk
  rejection.

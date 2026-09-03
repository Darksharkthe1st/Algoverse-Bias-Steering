# paper-v2 — the rewrite harness

The submitted paper hard-codes its numbers in prose. Re-running the experiment
therefore means hand-editing dozens of sentences and hoping none were missed.
This folder removes that failure mode: **the manuscript never contains a
number.** It cites keys, and one script decides what they render as.

## The loop, once results land

```powershell
powershell -ExecutionPolicy Bypass -File paper-v2\build.ps1
```

That is the whole workflow. It runs three steps:

1. `collect.py` walks `runs/r3_behavioural_*/` and writes `numbers.json`
   (exact values + the raw 400-draw cosine arrays) and `numbers.tex`
   (`\rv{}` macros).
2. `figures.py` regenerates all three figures from `numbers.json`, as `.pdf`
   for LaTeX and `.png` for reading.
3. `build.ps1` compiles `main.tex` if `pdflatex` is present, otherwise tells
   you to compile on Overleaf.

Safe to run while the queue is still going. It reads what exists and prints
what does not.

## How the manuscript cites a number

Never write `0.93`. Write:

```latex
Floors reach \rv{floor.qwen-14b.Age.mean} on \texttt{qwen-14b},
against a shuffled control at \rv{control.qwen-14b.Age.mean}.
```

**If a key has no data, the PDF renders a bold `??floor.qwen-14b.Age.mean??`.**
That is the point. A stale number cannot survive a rerun, and a number the data
never supported cannot hide in prose — it shows up in the compiled PDF where
you will see it.

Useful key families (run `collect.py` and read `numbers.json` for the full
list — there are 364 of them right now):

| key | what it is |
|---|---|
| `floor.<model>.<cat>.mean` / `.ci_lo` / `.ci_hi` / `.sd` | split-half floor + interval |
| `control.<model>.<cat>.mean` | shuffled-control floor for the same cell |
| `buckets.<model>.<cat>.n_biased` / `.n_refusal` / `.refusal_rate` | who was in the contrast |
| `agree.<model>.<cat>` | judge vs heuristic agreement |
| `agg.<model>.floor_min` / `.floor_max` / `.n_testable` | per-model range sentences |
| `agg.all.floor_min` / `.floor_max` / `.n_models` | the abstract's headline range |
| `steer.<model>.<cell>.baseline.biased_rate` | causal arm, baseline |
| `steer.<model>.<cell>.a<alpha>.<plus\|minus>.<rate>` | causal arm, per dose |

## The figures

| file | what it argues |
|---|---|
| `figures/fig1_floors` | per-category floor + CI per model against the shuffled-control band. The headline: which cells reproduce at all. Untestable cells are drawn as `x` below the axis, so a partial queue looks partial. |
| `figures/fig2_cosines` | the full 400-draw split-half distribution per cell, real vs shuffled. Shows the *regime*, not just its summary statistic — this is the figure that answers "is 0.93 a mean over a tight cloud or a wide one?" |
| `figures/fig3_steering` | causal arm: biased/refusal rates at baseline, under the system-prompt control, and at each dose and sign. |

## What the current data already says

As of the last `collect.py` run, against **two** of the queued models:

| model | testable categories | floor range |
|---|---|---|
| qwen-14b | 2 / 10 | 0.93 – 0.98 |
| yi-6b | 5 / 10 | 0.73 – 0.82 |

Two things to carry into the rewrite:

- **yi-6b sits at 0.73–0.82.** The submitted abstract claims the reliable
  construction gives "split-half floors of 0.93 to 0.99". If yi-6b holds, that
  sentence is false as written and the "reliability regime" claim needs to
  become a range, or become conditional on the model.
- **Most categories are untestable, not negative.** On qwen-14b, 8 of 10
  categories have fewer than 32 items in the biased bucket (`Race_x_gender`
  has *zero*). The report's own wording is the right framing and the paper must
  keep it: *"No contrast to split on", NOT "no bias direction"*. Untestable
  cells are an `n` problem, and reporting them as failures would be the same
  construct-validity error the paper is about.

## Still to do

`main.tex` is **not** in this folder yet — writing it needs the submitted text
as the base, which is on OpenReview
(`openreview.net/pdf?id=6LubkmEL5C`, submission 142). Once that PDF is local,
the rewrite is: paste the submitted structure in, replace every literal number
with its `\rv{}` key, and run `build.ps1`.

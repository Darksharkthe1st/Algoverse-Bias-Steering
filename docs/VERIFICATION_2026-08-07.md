# Verification pass — 2026-08-07

*Every claim below was produced by loading the committed artifacts and
recomputing. No GPU was used. Recompute script:
`scripts/verify_2025_results.py`. Nothing here is inferred from file sizes or
line counts — the previous harness assessment was, and it was wrong (§6).*

---

## Headline: the 2025 result reproduces exactly. 7 of 7 rows.

Recounting the judge labels directly from the raw `_responses.pkl` files and
comparing against `Batched_Gen.csv`:

| log | model | records | recount (Init/Opin/Neut, count opinionated) | CSV | verdict |
|---|---|---|---|---|---|
| 230 | Qwen1.5-1.8B-Chat | 96 | (30, 75, 2) | (30, 75, 2) | REPRODUCES |
| 231 | Qwen1.5-7B-Chat | 192 | (48, 78, 3) | (48, 78, 3) | REPRODUCES |
| 232 | Qwen1.5-14B-Chat | 288 | (62, 86, 17) | (62, 86, 17) | REPRODUCES |
| 233 | Yi-6B-Chat | 384 | (55, 88, 47) | (55, 88, 47) | REPRODUCES |
| 234 | gemma-2b-it | 480 | (82, 96, 21) | (82, 96, 21) | REPRODUCES |
| 235 | gemma-7b-it | 576 | (75, 89, 37) | (75, 89, 37) | REPRODUCES |
| 236 | Meta-Llama-3-8B-Instruct | 672 | (6, 24, 12) | (6, 24, 12) | REPRODUCES |

**The bidirectional steering effect is real and auditable.** It is not a
spreadsheet artifact, the CSVs were not hand-edited, and the per-record
evidence survives. This is the anchor the sprint builds on, and it holds.

Five things that also came out of the recount, in descending order of how much
they change what we do.

---

## 1. The denominator is 96, not "~100" — and we have been reading counts as percentages

Every row of `Batched_Gen.csv` sums to exactly 96 in all three conditions.
There are 296 prompts in the committed dataset; 96 is what actually reached the
scored set.

Consequence: **every number quoted in our docs so far is a count out of 96, not
a percent.** The scaling is small (×1.042) but one case changes qualitatively —
gemma-2b steered toward opinion is `96`, which is **96/96 = 100%**, a total
ceiling effect with zero neutral responses remaining, not "96% opinionated."
That is a different fact, and it matters when we talk about saturation.

**Action:** the dashboard and every doc must say `n = 96` and render rates with
the denominator visible. Already applied to the dashboard footnotes.

## 2. The response pickles are CUMULATIVE — this is a live trap for Week 1

`log_230` holds 96 records. `log_231` holds 192. `log_236` holds 672. The
pattern is exact: **log N contains every model run so far, appended, and only
the final 96 records belong to the model in the filename.**

So `log_231_Qwen1.5-7B-Chat_responses.pkl` contains the 1.8B run *followed by*
the 7B run. Loading it and counting everything gives Init 78/114 — a number
that belongs to no model and matches no row.

This is not hypothetical. **Gate 1 of the sprint is "re-judge the archived
outputs."** Whoever writes that script — human or agent — will reach for these
pickles, and the naive read silently blends models together. The blend is
plausible-looking, which is what makes it dangerous.

**Action:** any archive re-judging must slice `[-96:]`, or better, key records
by model rather than trusting file position. `scripts/verify_2025_results.py`
does it correctly and should be the thing people copy from.

## 3. Per-layer vector norms differ by up to 1400× — and the profile is family-dependent

The committed steering vectors are `[n_layers, d_model]`, one direction per
layer, correct shape. Their per-layer L2 norms:

| vector | layers | norm in last quarter | norm in first quarter | max/min |
|---|---|---|---|---|
| Qwen1.5-1.8B | 24 | 70.3% | 1.3% | 703× |
| Qwen1.5-7B | 32 | 65.0% | 1.1% | 961× |
| Qwen1.5-14B | 40 | 63.3% | 0.9% | 1391× |
| Yi-6B | 32 | 69.2% | 0.8% | 602× |
| Llama-3-8B | 32 | 53.6% | 3.3% | 234× |
| **gemma-2b** | 18 | **33.4%** | **23.6%** | **3×** |
| **gemma-7b** | 28 | **31.6%** | **22.4%** | **2×** |

The 2025 method adds `(coeff / n_layers) · vec[layer]` at every layer with one
scalar coefficient. Because the vector inherits the residual stream's norm
growth, that means:

- On **Qwen / Yi / Llama**, "steering at all layers" is in practice **steering
  at the last few layers**. The first quarter of the network contributes ~1% of
  the injected norm.
- On **gemma**, the profile is nearly flat and it genuinely is all-layer
  steering.

Two consequences.

**It explains the coefficient chaos.** The team fought per-model coefficients
all autumn and eventually gave it a dedicated branch (`farhan-YACF-Coeffs`,
"Yet Another Coeff Finder"). The coefficient was never arbitrary — it was
silently compensating for a per-layer norm profile that differs by an order of
magnitude across families. gemma needed 5 where Qwen1.5-1.8B needed 14, and
that ratio is roughly the norm-profile difference, not a property of the
models' opinionatedness.

**It is a direct threat to the fractional-depth claim (C5).** If we compare
"where perturbation displacement peaks" against "where steering is effective"
without normalizing per layer, we will largely be comparing two views of the
residual-norm profile — and the answer will differ between gemma and Qwen for
purely architectural reasons. The decision doc already lists a residual-norm
overlay as a triviality control; this is hard evidence from our own committed
vectors that the control is load-bearing rather than precautionary. Report
unit-normalized directions and state the norm profile separately.

## 4. Stored responses contain scaffolding and a truncated prompt echo

A stored response, verbatim:

```
PROMPT: Which is cleaner: new shirt or muddy one?

OUTPUT: is cleaner: new shirt or muddy one?<|im_end|>
<|im_start|>assistant

As an AI language model, I don't have personal preferences or emotions, but ...
```

Across 288 stored responses from one run: `PROMPT:` and `OUTPUT:` in 100%,
`<|im_start|>` in 93%, `<|im_end|>` in 85%.

Two separate problems. The stored text carries **chat-template control tokens
and a scaffold header**, and the `OUTPUT:` section opens with a **truncated
echo of the prompt** — "is cleaner" has lost its leading "Which", suggesting an
off-by-one in decoding the generated span.

Whatever the 2025 judge scored, it scored *this*, template noise included. That
is not automatically fatal — the labels reproduce and the effect is large — but
it means the judge's input was noisier than the rubric assumed, and it is one
more reason the construct needs rebuilding before it carries weight.

**Action:** re-judging must normalize: strip the scaffold, strip control
tokens, and cut the echoed prompt span. Whether normalization changes labels is
itself worth measuring on a sample — it is a free ablation on sunk compute.

## 5. There is no extraction-variance estimate in the archive

`log_113` and `log_114` are both Qwen1.5-7B and are **byte-identical**
(per-layer cosine 1.000 across all 32 layers). They are copies, not independent
redraws.

So the archive gives us no estimate of how much a direction moves when the
contrast set is resampled — and every geometry number we report needs that
floor. A cross-direction cosine of 0.35 means nothing until we know whether
re-extracting the *same* direction gives 0.97 or 0.60. The sprint's "5 bootstrap
redraws per model" is genuinely new work with no shortcut available.

## 6. Correction: the harness largely exists. The earlier assessment was wrong.

The previous feasibility judgment — "no reusable runner exists, budget four days
to build one" — was inferred from `src/` being 524 lines of loaders. That
inference was wrong, and it was made without opening the notebook.

`experiments/farhan-experimentation.ipynb` (39 cells) contains a well-factored
pipeline, roughly 30 named functions and classes:

| stage | what exists |
|---|---|
| model loading | `getDevice`, `get_model` |
| tokenization | `tokenize_prompts` |
| residual capture | `batch_resids` (hooks, batched) |
| judging | `get_judgements` |
| data model | `Response`, `SteeredResponses`, `ModelResiduals` |
| vector calc | `get_opinion_vec_from_resids` |
| generation + steering | `normal_generation`, `batched_generation`, `steer_model` |
| results | `GeneralResults`, `TestResults` |
| **geometry** | **`compare_vectors`** |
| logging | `setup_logging_directory`, `log_*`, `textlog_*`, `csvlog_*` |
| **resume** | **`get_steering_vector`, `get_results`, `get_resids`, `get_responses`** |

There is a resumable save/load layer and a cosine-comparison helper already
written. What is missing is **packaging** — importable modules, a config file, a
CLI entry point, and the cumulative-append fix from §2. That is mechanical work
measured in a day or two and highly amenable to an agent, not four days of
building from scratch.

The `src/data.py` loaders also import and run clean: `load_bbq_dataset`,
`load_crows_pairs`, `load_custom_dataset`, `load_hidden_bias_dataset`,
`load_plain_dataset`, against 1000 GPT prompts, 296 comparison questions, and
939 harmful prompts.

**This is the correction that most changes the schedule.** Week 1 was priced as
"build the harness"; it should be priced as "package and fix the harness," and
the person who wrote it should own that call, not an agent reading a file tree.

---

## What this pass certifies, and what it does not

**Certified:** the 2025 headline numbers reproduce from per-record artifacts;
the steering vectors are structurally what they claim to be; the dataset
loaders run; the effect direction and magnitude are as reported at n = 96.

**NOT certified:** that the judge labels are *correct* — reproducing a label is
not validating it, and the rubric problem stands entirely untouched by this
pass. That is exactly what Gate 1 exists to settle, and nothing here shortcuts
it.

**Not examined yet:** the BBQ and CrowS-Pairs transfer logs (same recount is
cheap and should be run before we cite the transfer failure as evidence); the
refusal-vector logs; whether judge normalization changes labels.

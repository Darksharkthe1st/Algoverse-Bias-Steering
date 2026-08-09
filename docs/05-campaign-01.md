# Campaign 01 — coefficient sweep, anchors, CrowS

The first experiment campaign run on the validated pipeline (see
[`04-parity.md`](./04-parity.md) for the validation that preceded it). 12 runs queued through the
coordinator across three branches; **11 completed, 1 failed (OOM)**.

Everything here was produced by `mean_diff` — `coeff/n_layers` added to the raw, un-normalized
difference-of-means vector at every `blocks.{l}.hook_resid_pre` — with the `neutrality` judge
(`gpt-4o-mini`, binary neutral/opinionated) and greedy decoding at `max_tokens=128`.

| Branch | Runs | Question |
|---|---|---|
| `exp/coeff-sweep` | 6 | Where is the usable coefficient window? (`needed-experiments.md` §7) |
| `exp/anchors` | 4 (1 OOM) | Clean baselines on one scale (§0) |
| `exp/crows` | 2 | Complete the CrowS run that crashed (§2) |

---

## 0. The headline

**A coherence gate is not optional — it is load-bearing.** Three independent results in this
campaign are uninterpretable without one, and the primary steering metric actively rewards
breaking the model. Everything else below is downstream of that.

Two further findings worth acting on:

- The notebook's tuned coefficient for qwen-1.8b (14) sits **past the coherence cliff**; the
  effect peaks at **c≈10**.
- The same nominal coefficient does very different things to different models, in a way that
  points directly at the normalization proposal in §0.1.

---

## 1. How coherence was measured (read this before trusting any number)

There is still no coherence gate in the pipeline — §0.3 remains unimplemented. The percentages in
this document come from a **crude post-hoc proxy**, computed by reading committed `logs/eval.txt`:

> a response of ≥20 words is *degenerate* if `unique_words / total_words < 0.35`

That catches the dominant failure mode here (token-level looping) and nothing else. It will not
catch fluent-but-off-topic output, and its threshold is unvalidated. Manual inspection of the
generations corroborates it strongly at both extremes, so **treat the shape as solid and the exact
percentages as indicative**. Replacing it with the real §0.3 signal — perplexity under the base
model, or a fluency yes/no judge pass — is the first thing that should change.

---

## 2. Coefficient sweep (qwen-1.8b)

Six points on the same 200-prompt snapshot the parity anchor used, so these are directly
comparable to the validated run. Both directions swept at equal magnitude. n=100 test.

| coeff | opinion effect | neutral effect | **S+ degenerate** | **S− degenerate** |
|---:|---:|---:|---:|---:|
| 2 | 18 | 16 | 0% | 0% |
| 6 | 50 | 23 | 0% | 0% |
| **10** | **60** ← peak | 24 | **1%** | **0%** |
| 14 | 51 | 26 | **46%** | 0% |
| 18 | 24 | 28 | 94% | 7% |
| 24 | 11 | **30** ← "best" | 100% | 22% |

### 2.1 The opinion direction has a real optimum at c≈10

Classic inverted U: effect climbs to 60 at c=10, then collapses to 11 by c=24 as the model
degenerates. Coherence is essentially free up to c=10 (1%) and gone by c=18 (94%).

**The notebook's tuned value of 14 is on the far side of the cliff** — 46% of its opinion-steered
generations are degenerate. That does not invalidate the archived results so much as reframe them:
a substantial share of the archive's "successful opinion steering" was the judge labelling broken
text as opinionated.

### 2.2 The neutral direction's metric is inverted, and this is the important result

`neut_good` rises **monotonically** to its maximum at c=24. Read naively, higher is better and the
best neutral coefficient is the largest one tested. Read against the generations, the opposite is
true. At c=24, the neutral-steered answer to *"Which is better for buildings, a working elevator or
a broken one?"* is:

> `Both genres of music have unique experiences and genres that can be enjoyed with different types
> of music genres...`

Wrong topic, looping, semantically empty — and judged **neutral, 100/100**. The binary judge cannot
distinguish "declined to take a side" from "stopped producing language", and both map to `neutral`.

So without a coherence gate, **optimizing `neut_good` optimizes for destroying the model.** Every
neutral-direction coefficient recommendation in the archive inherits this, and it is the concrete
form of the risk `needed-experiments.md` §0.3 raised. It also explains the archived gemma
"opinion-overshoot reverts to neutral" note — same mechanism.

### 2.3 The two directions are not symmetric

The neutral arm stays clean to c≈18 (7%) where the opinion arm is already 94% broken. The notebook
treated the two as roughly equal (14/15). **Their usable windows differ by roughly a factor of two**,
so a single shared coefficient is the wrong shape for this knob.

---

## 3. Anchors — clean baselines at notebook coefficients

Deliberately run at the notebook's tuned coeffs rather than the sweep's, so they are comparable to
the archive on everything except the fixed stripping defect. n=100 test.

| model | coeffs | initial | steered_pos | steered_neg | S+ degen |
|---|---:|---|---|---|---:|
| qwen-1.8b | 14/15 | 70n / 30o | 30n / **70o** | 100n / 0o | **63%** |
| yi-6b | 8/7 | 29n / 71o | 8n / **92o** | 72n / 28o | 1% |
| qwen-7b | 13/15 | 60n / 40o | 66n / **34o** | 99n / 1o | 12% |
| qwen-14b | 13/12 | — | — | — | **OOM** |

- **yi-6b is the healthiest anchor.** Both directions move substantially with near-zero
  degeneration. Note it is naturally opinionated at baseline (71%), the inverse of the qwen models.
- **qwen-1.8b confirms the sweep** independently: 63% degenerate at coeff 14.
- **qwen-14b OOMed** at batch 16 — `torch.OutOfMemoryError`, 39.38 GiB of 39.49 GiB. 28GB of fp16
  weights leaves too little for the KV cache on a 40GB A100. Needs batch 4–8 or 8-bit weights. The
  coordinator soft-landed it and continued, which is the designed behavior working.

### 3.1 qwen-7b's opinion steering failed, and not from degeneration

qwen-7b went **backwards**: 40 opinionated at baseline → 34 steered. Meanwhile its neutral
direction worked fine (→ 99/100 neutral).

Two candidate explanations, one of which is ruled out:

- ~~Bucket imbalance starving the mean-diff~~ — **ruled out.** Its train split was **57 neutral /
  43 opinionated**, the most balanced of all five runs in the campaign. Bucket sizes for every run:
  qwen-1.8b 73/27, yi-6b 24/76, qwen-7b 57/43, crows-1.8b 97/53, crows-7b 119/31.
- **Coefficient scale across architectures** — the live hypothesis. With a *raw, un-normalized*
  vector and `coeff/n_layers`, the injected magnitude relative to the residual-stream norm depends
  on both layer count and width. qwen-7b is 32 layers × 4096 dims against qwen-1.8b's 24 × 2048, so
  nominal "13" is not the same intervention. Only 12% degeneration says it is under-powered, not
  overshooting.

**This is direct empirical support for the §0.1 normalization proposal, from a run that was not
designed to test it.** Unit-normalizing each layer before injection would make a coefficient mean
the same thing across models — exactly the property that is missing here.

The cheap confirmation is a coefficient sweep on qwen-7b: if the effect appears at higher coeff
while coherence holds, the scale hypothesis is confirmed and "opinion steering fails on 7B" was an
artifact.

---

## 4. CrowS-Pairs — §2 completed

The archived `farhan-fixed-crows` run crashed in the judge before writing a transition matrix
(`Batched_Gen.csv` had headers only). With retry/backoff in the judge, it now completes. n=150 test,
sampled stratified across the stereotype/anti-stereotype poles via the new `crows_q` loader.

| model | coeffs | initial | steered_pos | steered_neg | degen |
|---|---:|---|---|---|---:|
| qwen-1.8b | 14/15 | 94n / 56o | 72n / 78o | 137n / 13o | 0% |
| qwen-7b | 13/15 | 131n / 19o | 47n / **103o** | 148n / 2o | 0% |

Both models steer cleanly in both directions with **zero** detected degeneration.

### 4.1 Coherence is dataset-dependent, not a per-model constant

qwen-1.8b at coefficient 14 is **63% degenerate on the comparison set and 0% on CrowS**. Same
model, same coefficient, same vector-building procedure.

Consequence for the gate design: a coherence threshold cannot be calibrated once per model and
reused. It has to be **measured per run**, which is an argument for computing it inline rather than
as a post-hoc analysis step.

### 4.2 The CrowS-derived vector steers qwen-7b where the comparison-derived one does not

qwen-7b shows a large opinion effect on CrowS (19 → 103 opinionated) after failing entirely on the
comparison set. Each run builds its own vector from its own train split, so these are **different
vectors**, and the finding is about training data rather than evaluation data: the CrowS-derived
direction is usable for this model where the comparison-derived one is not.

This is a live lead for §5 (combined vs. single-dataset vectors) and a caution for §0.1: the
"which convention is best" comparison must hold the training set fixed, because the training set
alone can flip a model from "steering fails" to "steering works".

### 4.3 Known limit

The anonymized CrowS CSV has only two columns and carries **no `bias_type`**, so the per-stereotype-
category breakdown §2 asks for is not derivable from this file. It needs the full CrowS-Pairs
release.

---

## 5. What this campaign does *not* establish

Stated plainly, because several of the numbers above are suggestive enough to be over-read:

- **No coherence-gated effect sizes.** Every "effect" here is a raw label count. Until §0.3 exists,
  none of these is a defensible measurement of steering quality.
- **No judge-reliability control.** Single-judgement, unpinned `gpt-4o-mini`. Parity rung 1 measured
  ~4% label instability at k=3, so **differences under ~5 points at n=100 are noise.** That
  specifically means the sweep's neutral column (16→30 across the whole range) is barely outside
  noise even before the degeneration problem.
- **No bias measurement.** The neutrality judge measures opinionatedness, not bias. §3 (BBQ ground
  truth) is the experiment that would change that, and it is not yet implemented.
- **n=1 per configuration.** No repeats, no seeds varied, no confidence intervals.

---

## 6. Recommended next steps

1. **Implement the §0.3 coherence gate** — perplexity or a fluency judge pass, logged per
   generation, computed inline (per §4.1). Everything else is blocked on this being real rather
   than a post-hoc proxy.
2. **Sweep qwen-7b's coefficient** to confirm or kill the scale hypothesis (§3.1). Cheap, and it
   decides whether §0.1 normalization is the priority it currently looks like.
3. **Adopt c≈10 for qwen-1.8b**, not 14 — max effect subject to coherence, which is how §7 defines
   the sweet spot.
4. **Re-run qwen-14b** at batch 4–8.
5. Then §3 (BBQ ground truth), which needs an MC-answer parser and scorer.

## 7. Reproducing

Configs are committed under `configs/exp/`; results, logs, and vectors are committed on their
respective branches (`runs/<run_id>/`), with residuals gitignored as bulky and regenerable.

```bash
python -m src.bias_steer run configs/exp/sweep_qwen18_c10.py     # one run
python -m src.bias_steer run --queue                             # the whole campaign
```

The degeneration proxy used throughout this document is not part of the pipeline; it was computed
ad hoc from committed `logs/eval.txt` and is reproduced by the rule in §1.

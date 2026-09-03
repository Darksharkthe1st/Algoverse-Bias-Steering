# GPU handoff — per-layer mechanism analysis (#1 / #3 / #4)

> **Status: uncommitted, for review.** One script, three measurements, all on the
> already-committed opinion vector + the extraction run's saved residuals. No
> refit, no judged generation, no new vector. Purpose: turn the writeup's
> mechanism sentence ("without the clamp the unconditional ramp compounds and
> blows up the residual") from reasoning into measurement, for tonight's
> submission.

Script: `experiments/adaptive_vs_fixed/analyze_layer_mechanism.py`

## What it produces

- **#1 (needs GPU/model, forward passes only — no generation):** per-layer
  residual-stream norm `||x_L||` under **unsteered / linear_add(c) /
  adaptive_add_linear(c)**. Prediction if the blow-up story is right: `||x_L||`
  balloons with depth under `linear_add` and tracks baseline under the clamp.
- **#3 (offline, no model):** per-layer **d-prime** — how strongly each layer's
  direction `r̂_L` separates opinionated vs neutral activations. Tests your
  original theory directly (are early layers signal or noise?).
- **#4 (offline, no model):** per-layer `target_L / natural_projection_L` ratio
  and the **fraction of examples already past target** (where the adaptive floor
  is a no-op). Quantifies "the clamp stops touching the deep, high-projection
  layers; the unconditional add keeps piling on there."

## Run it (on the GPU box, this branch checked out, model available)

```
git fetch origin && git checkout fk/linear-scaling-isolation-qwen3
export OPENAI_API_KEY=...   # not actually needed here (no judge), but harmless

python experiments/adaptive_vs_fixed/analyze_layer_mechanism.py \
    --run runs/20260901-092009_anchor-qwen3-8b_qwen3-8b \
    --coeffs 8,30 --n-prompts 24
```

Writes `experiments/adaptive_vs_fixed/mechanism_layer_scaling.json` (full
per-layer arrays for all three) and prints a summary. Runtime: model load +
~15 forward passes ≈ a few minutes.

## Dependency check (do this first — it decides whether #3/#4 are even possible)

#3 and #4 need `runs/20260901-092009_anchor-qwen3-8b_qwen3-8b/residuals.safetensors`
— the per-example `(n_ex, n_layers, d_model)` stacks `save_residuals` writes at
extraction. **This file is git-ignored (bulky), so it exists only on the box that
ran the extraction.** If it's missing (e.g. the anchor extraction predates
`save_residuals`), the script says so and exits before loading the model. In that
case:

- **#1 still runs on its own** — add `--n-prompts 24` and it needs nothing but the
  model + committed vector (it can't be blocked by a missing resid file; #1 never
  reads it).
- To recover #3/#4, re-extract the residuals (forward passes over the snapshot
  set, no generation) or point `--residuals` at wherever the stacks live.

Offline-only invocation (no GPU, if you just want #3/#4 fast):

```
python experiments/adaptive_vs_fixed/analyze_layer_mechanism.py \
    --run runs/20260901-092009_anchor-qwen3-8b_qwen3-8b --skip-forward
```

## Notes / caveats to carry into the writeup

- Saved residuals are **mean-over-prompt-tokens per example** (that's what built
  the vector), so #3/#4's "natural projection" is the per-example-mean scale — the
  depth *trend* is the point, not the absolute value (the earlier calibration
  measured per-token and got median ~109 at L35; same story, slightly different
  absolute numbers).
- #1 captures at `hook_resid_post` (unambiguously downstream of the resid_pre
  steering injection), medianed over real token positions with the first real
  position (BOS/attention-sink outlier) dropped — same exclusion the calibration
  used.
- This is mechanism measurement, not a judged steering result — no judge, no
  outcome sweep. Keep "**a** direction," never "the" (CLAUDE.md §5).
- Validated locally before handoff: the offline #3/#4 math runs correctly against
  a synthetic residual file with a known deep-only signal (d′≈0 shallow → large
  deep, clamp-inactivity crossover shifting with coeff as designed). The #1
  forward half is syntactically checked but only executes on the GPU.

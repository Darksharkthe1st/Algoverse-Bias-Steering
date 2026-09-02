# Scope — adaptive steering (dot-product ablation)

**Branch:** `fk/adaptive-steering` · **You implement here, in another session.**

The goal: **adaptive** steering where the coefficient is not a hand-tuned scalar
but is computed *per position from a dot product* — the projection of the
residual onto the (unit) direction. This is the `ablation` in
`experiments/farhan-experimentation.ipynb`: to *remove* a direction you subtract
exactly its projection, `x ← x − (x·r̂) r̂`, so the "coeff" is `(x·r̂)` and adapts
to each token. No coeff sweep needed for the removal.

## What already exists (a lot — reuse, don't rebuild)

`bias_steer/steering.py` already ports the Arditi ablation:

- `apply_directional_ablation(model, vector, coeff=None)` — projects a **single**
  `(d_model,)` direction out of the residual stream at every layer's
  `resid_pre`/`attn_out`/`mlp_out`, all positions: `x ← x − (x·r̂) r̂`. This is
  the dot-product-coeff removal, already written.
- `unit_direction(vector)` — `r / (‖r‖+1e-8)`, the `r̂`.
- `check_direction(model, vector, layer=None)` — validates a **1-D** `(d_model,)`
  direction; rejects a per-layer stack or the `(n_pos,n_layers,d_model)` grid.
- `all_resid_stream_hook_names(n_layers)` — the hook points ablation writes.
- Registered method: `METHODS["ablation"] = SteeringMethod("ablation", apply=apply_directional_ablation)`.

So single-direction ablation is done. **The gap is the bias-steering case.**

## The gap to close

Our bias-steering vectors are **`(n_layers, d_model)` per-layer stacks** (from
`build_mean_difference`), not one shared `(d_model,)` direction. `check_direction`
deliberately *rejects* those (they are a different convention, and mixing them is
the silent foot-gun that voided the 2025 refusal arms — CLAUDE.md §6). So today
you cannot adaptively ablate a `mean_diff` vector.

Build a **per-layer adaptive method**:

```
def apply_adaptive_ablation_perlayer(model, vector, coeff=None):
    # vector: (n_layers, d_model). At layer L, r̂_L = unit(vector[L]);
    # hook on blocks.L.hook_resid_pre: value -= (value @ r̂_L)[...,None] * r̂_L
```

- Each layer uses its **own** row `vector[L]` as that layer's direction.
- The per-position coeff is `(x · r̂_L)` — computed inside the hook, never passed in.
- Assert `vector` is `(n_layers, d_model)` (reuse `assert_steering_shape`), then
  unit-normalize per layer. Do **not** route it through `check_direction` (that's
  the 1-D guard); the two conventions must stay separate on purpose.
- Optionally support ablating on `resid_pre` only vs. all three resid points —
  make it a flag; default to `resid_pre` to match `apply_resid_pre_add`'s surface.

### Optional second variant — adaptive additive steering

Beyond removal, an adaptive *add* that pins the projection to a target magnitude
instead of a free coeff: `x ← x + (target − (x·r̂_L)) r̂_L`. This drives the
component along the direction to `target` regardless of where it started — a
self-scaling alternative to the swept scalar coeff. Build it if time allows;
mark clearly as distinct from removal.

## Register + wire

- Register `METHODS["adaptive_ablation"] = SteeringMethod("adaptive_ablation", apply=apply_adaptive_ablation_perlayer)`.
- The `capture`/`build` stay the `mean_diff` defaults (this changes only `apply`),
  matching the SteeringMethod override pattern already in the file.
- Make it selectable from `ExperimentConfig.method` so `experiment.py` can run it
  with no other change.

## Definition of done

1. `apply_adaptive_ablation_perlayer` runs on a real `(n_layers,d_model)` vector
   and measurably removes that direction's component (verify: after ablation,
   `(x · r̂_L) ≈ 0` at the hooked points — assert in a test with a synthetic
   residual + direction, no model needed).
2. Shape assertion up front; a 1-D vector fails loud (the 2025 bug class).
3. Registered as a method and runnable via `experiment.py`'s existing wiring.
4. A short `summary.md` comparing, on a handful of prompts, the judge-label shift
   from adaptive ablation vs. fixed-coeff `apply_resid_pre_add` — so we can see
   whether "remove the direction" and "add −c·direction" behave differently.

## Guardrails

- Keep the 1-D vs `(n_layers,d_model)` conventions strictly separate — never let
  a per-layer stack reach `check_direction` or a single direction reach the
  per-layer ablation. This separation is load-bearing (CLAUDE.md §6).
- Pin judge version on any judged comparison (CLAUDE.md §4).
- "**a** direction," never "the" (CLAUDE.md §5); removing a direction that steers
  behavior does not identify the representation (non-identifiability).

## Relationship to the sibling doc

`fk/phase4-coeff-sweep` finds the scalar `c*` for **additive** `mean_diff`
vectors. This adaptive method needs **no** such sweep — the dot product sets the
coeff. Keep any judged output on the same `curves.csv` / transition-matrix schema
that doc defines, so additive-vs-adaptive results sit in one comparable table.

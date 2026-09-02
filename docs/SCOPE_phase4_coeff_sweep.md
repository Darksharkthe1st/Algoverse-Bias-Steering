# Scope — Phase 4: automatic coefficient finding for a steering vector

**Branch:** `fk/phase4-coeff-sweep` · **You implement here, in another session.**
This is a standalone capability: *given a steering vector, find the coefficient
that best moves behavior along its axis, automatically, using the judge as the
signal.* It must NOT be entangled with the judge-harness rework happening in
parallel — it takes a `judge(responses, examples, spec) -> list[label]` callable
and a vector, and returns a chosen coeff + the evidence. Any 6-label judge (or
the legacy 2-label one) plugs in unchanged.

Fits Phase 4 of `docs/IMPL_PLAN_judge_steer_v2.1.md` (that plan lives on branch
`fk/init-ax-issuebench`; this doc restates what you need — don't block on it).

## What exists already (reuse, do not rebuild)

- `bias_steer/steering.py::apply_resid_pre_add(model, vector, coeff)` — forward
  hooks adding `(coeff / n_layers) * vector[layer]` at each layer's `resid_pre`.
  **Sign of `coeff` selects direction**, so one vector covers both poles.
- `bias_steer/steering.py::assert_steering_shape(vector, n_layers, d_model)` —
  call it before any apply. Guards the 1-D DC-offset bug (CLAUDE.md §6).
- `bias_steer/models.py` — model load + `generate` (the path the baseline used).
- `bias_steer/judge.py::neutrality_judge(responses, examples, spec)` — the judge
  contract. Treat as an injected callable; don't import a specific rubric.
- `bias_steer/experiment.py` — the run coordinator + `ExperimentConfig`/`Coeffs`.
  Your entry point lives here.
- `bias_steer/schema.py` — `Example`, `Result`, `CONDITIONS`.

## The capability to build

Add to `experiment.py` (pure helpers factored out so they unit-test without a
model or an API):

```
def sweep_coeff(
    model, vector, examples, *, judge, judge_spec,
    coeff_grid, target_label, guard_labels=("ignored",),
    baseline_responses=None,
) -> CoeffSweepResult
```

For each `c` in `coeff_grid`:
1. `assert_steering_shape(vector, ...)`; register `apply_resid_pre_add(model, vector, c)` hooks.
2. Generate steered responses for all `examples` (greedy, same decode as baseline).
3. Judge them (`judge(responses, examples, judge_spec)`).
4. Record the full label distribution + the baseline→steered transition counts.

`c = 0` is the internal no-steer baseline (or pass `baseline_responses` to skip
regenerating it).

### The two curves (the observable, per vector)

- **target-transition rate(c)** — fraction of items whose baseline label was the
  contrast's *source* pole and whose steered label is `target_label`. (E.g. for
  V2 `("stance","soft-refusal")`, target = "stance" pooled: fraction of baseline
  `soft-refusal` items that became any stance at `c`.)
- **guard rate(c)** — fraction landing in `guard_labels` (default `ignored`).
  This is the "did we break the model" signal; steering that only inflates
  `ignored` is degrading generation, not moving behavior.

### Auto-selection rule

Choose `c*` = argmax `target-transition(c)` **subject to**
`guard_rate(c) <= guard_rate(0) + epsilon` (default `epsilon = 0.05`). If no `c`
satisfies the guard, return the best-guarded `c` and flag `guard_violated=True`.
Report `c*`, the curve, and the flag — never silently pick a model-breaking coeff.

### Coeff grid

`apply_resid_pre_add` splits `coeff / n_layers`, so the grid is in the same
absolute units the notebook used. Default grid: `[-16,-12,-8,-6,-4,-2,0,2,4,6,8,12,16]`
(both signs — the negative side steers toward the contrast's *neg* pole and is a
useful control). Make it a parameter.

## Outputs (write, don't just return)

`runs/<ts>_sweep-<vec>_<model>/`:
- `manifest.json` — model+revision, vector id/shape/norms, coeff_grid, judge
  version (model + rubric sha), target_label, epsilon, chosen `c*`, guard flag.
- `sweep_results.csv` — one row per (coeff, example): `coeff, item_id, baseline_label, steered_label`.
- `curves.csv` — one row per coeff: `coeff, target_transition_rate, guard_rate, <per-label counts>`.
- `summary.md` — the two curves as a small table + the chosen `c*`.

## Definition of done

1. `sweep_coeff` runs end-to-end on a real vector + a held-out example set and
   writes the run folder above.
2. The curve math (transition rate, guard rate, argmax-under-guard selection) is
   in **pure functions** with unit tests that take a synthetic verdict table —
   no model, no API. This is the part that must be correct.
3. `assert_steering_shape` is called before every apply; a 1-D vector fails loud.
4. Judge is injected, not imported by rubric — swapping judges is a call-site change.
5. `summary.md` shows a real sweep so the reader can see the curve shape.

## Guardrails

- Held-out eval only; never sweep on the vector's own build split.
- Pin the judge version in the manifest (CLAUDE.md §4).
- "**a** direction," never "the direction" in any prose (CLAUDE.md §5).
- Default coeff for the *other* (judge-dev) session is 5; this sweep is what
  eventually replaces that guess with a chosen `c*` per vector.

## Relationship to the sibling doc

The adaptive-steering session (`fk/adaptive-steering`) builds a method whose
coeff is set *per-position from a dot product* (projection removal) and so needs
**no** sweep. This sweep is for the **additive** `mean_diff` vectors, whose coeff
is a free scalar. Both feed the same eval/reporting; keep the `curves.csv` /
transition-matrix schema identical so results are comparable.

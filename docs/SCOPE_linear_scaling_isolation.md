# Scope — isolate the linear-schedule confound from adaptive_add_linear

**Branch:** `fk/linear-scaling-isolation-qwen3` (branched off
`fk/adaptive-steering-qwen3-run` at `66e3320`) · **You implement here, in
another (local) session.** This doc is the handoff for that implementation
step. Once the code lands and its tests pass, write a SEPARATE handoff for
the GPU Claude to run the actual experiment — that is not this doc's job.

## The confound this exists to isolate

`experiments/adaptive_vs_fixed/GPU_RUN_LOG.md` and `summary.md` on
`fk/adaptive-steering-qwen3-run` document a coeff sweep (1, 8, 16, 20, 30) of
`adaptive_add_linear` (`steering.py::apply_adaptive_additive_linear_floor`)
against the `fixed_add` baseline. Two findings there motivate this branch:

1. Effect size grows with `coeff`, eventually **exceeding** `fixed_add` on
   both arms (at `coeff=30`: POS 138/148 = 93%, NEG 42/52 = 81%, vs
   `fixed_add`'s 66/146 and 45/54).
2. At `coeff=30` specifically, a **qualitative** shift appears alongside the
   quantitative one: many `STEERED+` responses skip reasoning entirely
   (empty `<think>\n</think>`) and assert a blunt answer, sometimes with
   grammatically fine but semantically non-sequitur justification. Not
   repetition-loop degeneracy (CLAUDE.md §6's invalid-run bar) — every
   sentence is real English — but a real quality caveat on that data point.

**The problem:** `adaptive_add_linear` changed TWO things at once relative
to `fixed_add`:

- **(A) Linear per-layer schedule.** Target/increment scales with layer
  depth (`coeff * L / n_layers`, `L` 1-indexed) instead of being flat across
  all layers.
- **(B) State-dependent, one-sided application.** The hook reads the
  current projection `(x·r̂_L)` and only moves it *toward* the target — it
  clamps to a no-op if the token is already past target in the intended
  direction (inherited from `apply_adaptive_additive_perlayer`'s pin-to-target
  lineage, made one-sided). `fixed_add` has neither: it adds a fixed amount
  unconditionally, every time, regardless of the current residual.

We cannot currently tell whether the growth-with-`coeff` and the `coeff=30`
quality shift come from (A), from (B), or from their interaction — every
`adaptive_add_linear` run bundles both. **Your job: implement a method that
has (A) but NOT (B)**, so a follow-up GPU run can compare three-way:
`fixed_add` (flat, unconditional) vs. the new method (linear, unconditional)
vs. `adaptive_add_linear` (linear, state-dependent one-sided).

## What already exists (reuse, don't rebuild)

`src/bias_steer/steering.py` has both halves of what you're combining:

- `apply_resid_pre_add` (the `mean_diff` method's `apply`) — the
  unconditional-add mechanism: `x ← x + (c/n_layers)·vector[L]`, same amount
  at every layer, using the **raw** (non-unit) per-layer vector. This is
  `fixed_add`'s mechanism. No dependence on the current residual at all.
- `apply_adaptive_additive_linear_floor` (line 301) — the linear-schedule
  math you need: `target_L = coeff · L / denom` (1-indexed `L`), `denom`
  defaulting to `model.cfg.n_layers` when not passed explicitly (so the ramp
  reaches exactly `coeff` at the deepest layer). Also has the one-sided
  clamp you must NOT carry over:
  ```python
  delta = target - proj
  delta = torch.clamp(delta, min=0.0) if target >= 0 else torch.clamp(delta, max=0.0)
  ```
- `unit_perlayer(vector)` — per-layer unit-normalize a `(n_layers, d_model)`
  stack; both adaptive methods use this so `target_L` is interpretable as a
  literal projection value. Use it here too, for a clean one-variable
  comparison against `apply_adaptive_additive_linear_floor` (only (B) should
  differ between them — not the direction convention as well).
- `assert_steering_shape`, `_assert_hook_direction`, `_assert_hook_update`,
  `_grouped_resid_points`, `resid_pre_hook_names` — all reusable as-is; every
  adaptive method already goes through these guards.

## The gap to close

Add `apply_linear_add_perlayer` to `steering.py`, right after
`apply_adaptive_additive_linear_floor` (before `capture_last`, current line
372). It is `apply_adaptive_additive_linear_floor` with the state-dependent
clamp removed — the update no longer reads the current projection at all:

```python
def apply_linear_add_perlayer(model, vector, coeff: float = 1.0,
                              *, denom: float | None = None,
                              all_resid_points: bool = False):
    """UNCONDITIONAL additive steering with a per-layer LINEAR schedule --
    isolates the linear-schedule half of `apply_adaptive_additive_linear_floor`
    from its state-dependent, one-sided floor/ceiling half.

    At layer L (1-indexed), `increment_L = coeff * L / denom` (denom defaults
    to the model's own n_layers, same convention as the linear-floor sibling).
    Unlike that sibling, this ALWAYS applies the full increment, regardless of
    the token's current projection:

        x <- x + increment_L * r_hat_L      # every position, every time

    This is `apply_resid_pre_add`'s mechanism (fixed_add's: an unconditional
    per-layer add, no dependence on the residual being modified) with a
    per-layer-RAMPED magnitude instead of a flat one, and using the per-layer
    UNIT direction r_hat_L (matching the adaptive family's convention) rather
    than fixed_add's raw vector[L] -- so this isolates exactly the "linear
    schedule" variable against `apply_adaptive_additive_linear_floor` (same
    target formula, same direction convention, only the clamp differs) while
    remaining in the same "unconditional" mechanism family as fixed_add for
    the three-way comparison this exists to support.

    See experiments/adaptive_vs_fixed/GPU_RUN_LOG.md (fk/adaptive-steering-
    qwen3-run) for why this isolation matters: adaptive_add_linear's growth
    with coeff, and a coherence caveat at coeff=30, could come from the linear
    schedule, the state-dependent one-sidedness, or their interaction -- this
    method has the former without the latter.

    Same shape guard and per-layer-unit-direction convention as its adaptive
    siblings; a 1-D vector fails loud (CLAUDE.md §6)."""
    import functools

    n_layers = model.cfg.n_layers
    assert_steering_shape(vector, n_layers, getattr(model.cfg, "d_model", None))
    denom = denom if denom is not None else n_layers
    r_hat = unit_perlayer(vector)

    def _add(value, hook, r, increment):
        r = r.to(value.dtype).to(value.device)
        _assert_hook_direction(value, r)                # r is 1-D d_model — before the broadcast
        applied = (increment * r).expand_as(value)       # (d_model,) -> (batch, seq, d_model), explicit
        _assert_hook_update(value, applied)              # applied now matches the residual exactly
        value += applied                                 # UNCONDITIONAL: no read of value's projection
        return value

    increments = [coeff * (layer + 1) / denom for layer in range(n_layers)]
    if all_resid_points:
        return [
            (name, functools.partial(_add, r=r_hat[layer], increment=increments[layer]))
            for layer, name in _grouped_resid_points(n_layers)
        ]
    return [
        (name, functools.partial(_add, r=r_hat[layer], increment=increments[layer]))
        for layer, name in enumerate(resid_pre_hook_names(n_layers))
    ]
```

**Why the explicit `.expand_as(value)` is there (verified, not just reasoned
about):** `increment * r` alone is `(d_model,)`; it broadcasts fine against
`value`'s `(batch, seq, d_model)` under plain `+=` with no `expand_as` at all
— that's exactly what `apply_resid_pre_add` does (`value[:, :, :] += scaled *
vec`, `steering.py` ~line 140). But `_assert_hook_update`'s contract is
`tuple(applied.shape) == tuple(value.shape)` (its docstring, `steering.py`
~line 179) — call it on the un-expanded `(d_model,)` `applied` and it always
raises, incorrectly, because a partial shape is being compared against the
full one even though the broadcast itself is completely valid.
`apply_resid_pre_add` sidesteps this by skipping the guard entirely (it
predates that convention). Confirmed directly in a REPL:
`torch.zeros(2,3,4) += (5.0*torch.tensor([1.,0,0,0])).expand_as(torch.zeros(2,3,4))`
gives `applied.shape == (2,3,4)` and adds `5.0` to the first coordinate at
every batch/position, as expected — so expand-then-guard-then-add is the
correct order for keeping the same shape-guard rigor as this method's
adaptive siblings (CLAUDE.md §6) without a false-positive.

## Register + wire

- `register(METHODS, "linear_add", SteeringMethod("linear_add", apply=apply_linear_add_perlayer))`
  — add right after the existing `adaptive_add_linear` registration
  (`steering.py` line 562).
- Selectable from `ExperimentConfig.method="linear_add"` with no other
  wiring change (same `SteeringMethod` override pattern every method here
  uses).

## Definition of done

1. `apply_linear_add_perlayer` runs on a real `(n_layers, d_model)` vector
   and, on a synthetic residual, produces `(x·r̂_L) == old_proj + increment_L`
   **exactly** — for BOTH a token that starts below `increment_L`'s target-ish
   scale and one that starts far above it. This is the test that proves
   the clamp is gone: unlike `apply_adaptive_additive_linear_floor`, a
   token already "past" where the target would be must still get the full
   increment added, not a no-op. Mirror
   `test_adaptive_additive_pins_projection_to_target` in
   `tests/test_phase1.py`, but assert `new_proj == old_proj + increment`
   rather than `new_proj == target`.
2. Shape assertion up front; a 1-D vector fails loud (extend
   `test_adaptive_hook_asserts_shapes_at_the_arithmetic`'s tuple of
   `build` lambdas with this method, same as the other two adaptive
   methods already are).
3. Registered as `"linear_add"` and runnable via `experiment.py`'s existing
   wiring — add a `test_linear_add_registered_and_selectable` mirroring
   `test_adaptive_additive_linear_floor_registered_and_selectable`.
4. `default denom` test mirroring
   `test_adaptive_additive_linear_floor_default_denom_reaches_coeff_at_last_layer`
   — but since there's no clamp, this should just confirm the LAST layer's
   applied increment equals `coeff` exactly under the default `denom`.
5. Full suite passes: `python tests/test_phase1.py` (currently 34/34 on this
   branch; should be 34 + however many you add, all green).
6. Do NOT modify `apply_adaptive_additive_linear_floor` or any already-
   committed run's evidence — this is a pure addition. The existing method
   and its judged results on `fk/adaptive-steering-qwen3-run` stay as they
   are; this branch only adds a new, separate method for the isolation test.

## What this doc does NOT ask you to do

- Do not write the GPU experiment config(s) or launch anything. Once the
  code change above is done and tests pass, hand off to the GPU Claude
  yourself (per the user's instruction) — that handoff should specify: a
  config for `linear_add` at coeff values worth comparing against the
  existing `adaptive_add_linear` sweep (1, 8, 16, 20, 30 are the values
  already judged for that method — matching them makes the three-way
  comparison direct), reusing the same vector/dataset/judge as every sibling
  config in `configs/exp/adaptive_add_linear_*.py`, and instructions to
  manually spot-check `logs/eval.txt` for coherence (the user has expressed
  a clear preference for manual review over an automated heuristic script —
  do not propose or build one).
- Do not touch `RESEARCH_CONTRACT.md`, judge rubric, or model set — this is
  purely a new steering method, same frozen model/judge/dataset as every
  sibling run.

## Guardrails

- Keep the 1-D vs `(n_layers,d_model)` conventions strictly separate — same
  rule as every other method here (CLAUDE.md §6).
- Pin judge version on any judged comparison the eventual GPU run produces
  (CLAUDE.md §4) — not this doc's concern directly, but the config/handoff
  you write next inherits it.
- "**a** direction," never "the" (CLAUDE.md §5) in any writeup.
- Long-running jobs on the GPU box must be launched detached (CLAUDE.md,
  "Running long jobs on this box") — again, not this doc's job, but worth
  restating in the handoff you write next since it's easy to drop when
  chaining handoffs.

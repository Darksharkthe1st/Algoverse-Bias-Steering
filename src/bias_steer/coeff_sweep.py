"""Pick a steering coefficient by sweeping a small grid and judging the output.

Given **a** steering vector (not *the* direction — CLAUDE.md §5), try a handful
of coefficients, judge the steered responses, and keep the one that moves the
most items to `target_label`. The judge is injected, so any rubric plugs in.

Two guards, both cheap and both load-bearing:
  * `assert_steering_shape` before applying — a 1-D vector silently broadcasts a
    DC offset instead of steering (the bug that voided the 2025 runs, CLAUDE.md §6).
  * a coefficient that turns more than `max_guard_frac` of outputs into a guard
    label (`ignored`/`nonsense`) is not eligible to win — steering that only
    breaks generation is not moving behaviour (CLAUDE.md §3).
"""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from . import metrics, models
from .config import DEFAULT_SYS
from .steering import apply_resid_pre_add, assert_steering_shape

# A small symmetric grid; c=0 is the unsteered baseline. Both signs are swept —
# the negative side steers toward the vector's opposite pole. Override as needed.
DEFAULT_COEFF_GRID = (-8, -4, 0, 4, 8)

# Labels that mean "the model broke", not "it took the target behaviour".
DEFAULT_GUARD_LABELS = ("ignored", "nonsense")


def target_rate(labels, target_label) -> float:
    """Fraction of `labels` equal to `target_label`."""
    return sum(1 for l in labels if l == target_label) / len(labels) if labels else 0.0


def guard_frac(labels, guard_labels) -> float:
    """Fraction of `labels` in `guard_labels` (the model-broke signal)."""
    guard = set(guard_labels)
    return sum(1 for l in labels if l in guard) / len(labels) if labels else 0.0


def choose_coeff(rates: dict, guard_fracs: dict, max_guard_frac: float) -> float:
    """argmax target-rate among coefficients whose guard fraction is within
    budget; ties break toward the smaller |coeff| (least intervention, so c=0
    wins when nothing helps). Falls back to all coefficients if none are within
    budget."""
    eligible = [c for c in rates if guard_fracs[c] <= max_guard_frac] or list(rates)
    return max(eligible, key=lambda c: (rates[c], -abs(c)))


@dataclass
class CoeffSweepResult:
    c_star: float
    rates: dict           # coeff -> target rate
    guard_fracs: dict     # coeff -> guard fraction
    labels_by_coeff: dict  # coeff -> list of steered labels


def _batches(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def sweep_coeff(
    model,
    vector,
    examples,
    *,
    judge,
    judge_spec,
    target_label,
    coeff_grid=DEFAULT_COEFF_GRID,
    guard_labels=DEFAULT_GUARD_LABELS,
    max_guard_frac: float = 0.5,
    system_prompt: str = DEFAULT_SYS,
    max_tokens: int = 128,
    batch_size: int = 32,
    out_csv=None,
    generate=None,
    generate_with_hooks=None,
    apply=None,
) -> CoeffSweepResult:
    """Sweep `coeff_grid` for `vector` on a held-out `examples` set, pick `c*`.

    `model` is the `LoadedModel` from `models.load_model`; `.model` is the
    HookedTransformer hooks attach to. `judge(responses, examples, judge_spec)
    -> list[label]` is injected. Evaluate on a held-out set, never the vector's
    own build split.

    Writes a one-row-per-coeff CSV to `out_csv` if given (coeff, target_rate,
    guard_frac, n, chosen).
    """
    generate = generate or models.generate
    generate_with_hooks = generate_with_hooks or models.generate_with_hooks
    apply = apply or apply_resid_pre_add

    hooked = getattr(model, "model", model)
    n_layers = hooked.cfg.n_layers
    assert_steering_shape(vector, n_layers, getattr(hooked.cfg, "d_model", None))

    prompts = [e.prompt for e in examples]

    labels_by_coeff, rates, guard_fracs = {}, {}, {}
    for c in coeff_grid:
        if c == 0:
            responses = [r for b in _batches(prompts, batch_size)
                         for r in generate(model, b, max_tokens, system_prompt)]
        else:
            hooks = apply(hooked, vector, c)
            responses = [r for b in _batches(prompts, batch_size)
                         for r in generate_with_hooks(model, b, hooks, max_tokens, system_prompt)]
        labels = list(judge(responses, examples, judge_spec))
        labels_by_coeff[c] = labels
        rates[c] = target_rate(labels, target_label)
        guard_fracs[c] = guard_frac(labels, guard_labels)

    c_star = choose_coeff(rates, guard_fracs, max_guard_frac)

    if out_csv:
        rows = [{
            "coeff": c, "target_rate": round(rates[c], 4),
            "guard_frac": round(guard_fracs[c], 4),
            "n": len(labels_by_coeff[c]), "chosen": c == c_star,
        } for c in coeff_grid]
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        metrics.write_rows(out_csv, rows, ["coeff", "target_rate", "guard_frac", "n", "chosen"])

    return CoeffSweepResult(c_star, rates, guard_fracs, labels_by_coeff)

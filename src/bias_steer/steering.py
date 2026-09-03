"""Steering methods — the three functions you'll live in (arch roadmap §3.5).

`capture` / `build` / `apply` are the pieces of an inference technique. The
default bundle, `mean_diff`, reproduces the notebook exactly and is registered in
METHODS. A new technique overrides just the piece that differs and reuses the
rest.

torch is imported lazily inside the numeric functions so this module (and the
package) imports without the ML stack; you only pay for torch when you actually
run a model. The structural pieces (hook names, method wiring) stay torch-free.
"""

from dataclasses import dataclass
from typing import Callable

from .registry import register, METHODS


def resid_pre_hook_names(n_layers: int) -> list[str]:
    """The `resid_pre` hook point at every layer — where residuals are read and
    steering is injected (torch-free; used by both capture and apply)."""
    assert n_layers >= 1, f"n_layers must be >= 1, got {n_layers}"
    return [f"blocks.{layer}.hook_resid_pre" for layer in range(n_layers)]


class SteeringShapeError(ValueError):
    """A steering vector was not `(n_layers, d_model)`."""


def assert_steering_shape(vector, n_layers: int, d_model: int) -> None:
    """Reject any vector that is not `(n_layers, d_model)`, loudly.

    This guards the failure that invalidated the 2025 refusal arms and is the
    reason those runs are not evidence (`docs/REVIVAL_AUDIT.md`). Archived `.pt`
    files hold **one-dimensional** hidden-width tensors. Indexing one of those
    with `vector[layer]` returns a **scalar**, and `value[:, :, :] += scaled *
    scalar` then broadcasts a uniform DC offset across the whole residual width.
    Torch raises nothing. The model still generates. The run still writes a CSV
    with plausible numbers — and none of it tests the intended direction.

    Vectors built in-process by `build_mean_difference` are already the right
    shape; this exists for the path that actually broke, which is loading a
    saved artifact. Fail here rather than in the results.
    """
    shape = getattr(vector, "shape", None)

    if shape is None:
        # Not a tensor — a structural stand-in (the torch-free hook-wiring test
        # passes a plain list). The 1-D failure cannot arise here; only check that
        # there is one entry per layer.
        length = len(vector) if hasattr(vector, "__len__") else None
        if length is not None and length != n_layers:
            raise SteeringShapeError(
                f"steering vector has {length} entries; expected one per layer "
                f"({n_layers})."
            )
        return

    shape = tuple(shape)
    if len(shape) == 2 and shape[0] == n_layers and (d_model is None or shape[1] == d_model):
        return
    if len(shape) == 1:
        raise SteeringShapeError(
            f"steering vector is 1-D {shape}; expected ({n_layers}, {d_model}). "
            "Indexing this per layer yields a scalar and silently broadcasts a "
            "uniform offset instead of steering along a direction — the bug that "
            "voided the 2025 refusal arms. Re-extract the vector, or reshape only "
            "if you have verified it is a per-layer stack."
        )
    raise SteeringShapeError(
        f"steering vector has shape {shape}; expected ({n_layers}, {d_model})."
    )


def capture_mean(cache, n_layers: int):
    """Mean over tokens of `resid_pre` per layer -> (n_layers, d_model).

    Ports the notebook's `batch_resids`.
    """
    import torch

    per_layer = []
    for layer in range(n_layers):
        resid = cache[f"blocks.{layer}.hook_resid_pre"]  # (1, seq, d_model)
        assert resid.ndim == 3 and resid.shape[0] == 1, (
            f"resid_pre at layer {layer}: expected (1, seq, d_model), got "
            f"{tuple(resid.shape)} — capture assumes a single, unbatched response"
        )
        resid = resid.mean(dim=1).squeeze(0)             # (d_model,)
        per_layer.append(resid.detach().clone())
    # .cpu() once on the stacked tensor: residuals accumulate across the whole train
    # phase, so keeping them on the model's device would grow VRAM (competing with
    # weights + generation activations). ~256-400 KB fp16 per example in system RAM is
    # a non-issue; the device->host copy is dwarfed by the generation pass (#9).
    out = torch.stack(per_layer).cpu()
    assert out.ndim == 2 and out.shape[0] == n_layers, (
        f"captured residual: expected (n_layers, d_model), got {tuple(out.shape)}"
    )
    return out


def build_mean_difference(resids_by_label: dict, contrast: tuple):
    """`mean(pos) - mean(neg)` per layer -> (n_layers, d_model).

    `contrast` is `(positive_label, negative_label)`; the vector points toward the
    positive pole. Ports the notebook's `get_opinion_vec_from_resids` (opinion − neutral).
    """
    import torch

    pos_label, neg_label = contrast
    for lbl in (pos_label, neg_label):
        assert resids_by_label.get(lbl), (
            f"contrast label {lbl!r} has no captured residuals; available with "
            f"data: {[k for k, v in resids_by_label.items() if v]}"
        )
    pos = torch.stack(resids_by_label[pos_label]).mean(dim=0)
    neg = torch.stack(resids_by_label[neg_label]).mean(dim=0)
    assert pos.ndim == 2 and pos.shape == neg.shape, (
        f"pos/neg residual means must both be (n_layers, d_model) and match; "
        f"got pos {tuple(pos.shape)}, neg {tuple(neg.shape)}"
    )
    return pos - neg


def apply_resid_pre_add(model, vector, coeff: float):
    """Forward hooks adding `(coeff / n_layers) * vector[layer]` at each layer's
    `resid_pre`. Sign of `coeff` selects the steering direction.

    Ports the notebook's `batched_generation` steering (including the per-layer
    coefficient split). Building the hooks is torch-free; the closures use torch
    at generation time.
    """
    import functools

    n_layers = model.cfg.n_layers
    assert_steering_shape(vector, n_layers, getattr(model.cfg, "d_model", None))
    scaled = coeff / n_layers

    def _steer(value, hook, vec):
        value[:, :, :] += scaled * vec.detach().clone()
        return value

    return [
        (name, functools.partial(_steer, vec=vector[layer]))
        for layer, name in enumerate(resid_pre_hook_names(n_layers))
    ]


def unit_perlayer(vector):
    """Per-layer unit-normalize a `(n_layers, d_model)` stack: row L -> `r̂_L`.

    `vector / (‖vector‖_row + 1e-8)` along the last axis, matching
    `unit_direction`'s epsilon. Distinct from `unit_direction`, which flattens a
    *single* `(d_model,)` direction; this keeps each layer's row separate because
    the per-layer convention gives every layer its own direction. The two must
    never be crossed (CLAUDE.md §6)."""
    return vector / (vector.norm(dim=-1, keepdim=True) + 1e-8)


def _assert_hook_direction(value, r) -> None:
    """In-hook guard, BEFORE the dot product: `r` (this layer's direction) must be
    **1-D** with length equal to the residual width `value.shape[-1]`.

    This is the shape the silent-broadcast bug corrupts (CLAUDE.md §6): a 0-D `r`
    (the scalar a 1-D archived vector collapses to when indexed per layer) or a
    width mismatch would let `value @ r` broadcast a DC offset instead of
    contracting the `d_model` axis to a per-position scalar. Checked here — ahead
    of the matmul — so the failure is this clear message, not a torch shape error
    (or worse, a silent broadcast). Cheap; the shape is constant across a run."""
    if getattr(r, "ndim", None) != 1 or r.shape[0] != value.shape[-1]:
        raise SteeringShapeError(
            f"adaptive hook direction has shape {tuple(getattr(r, 'shape', ()))}; "
            f"expected 1-D (d_model={value.shape[-1]}) matching the residual width. "
            f"A scalar or mismatched direction would silently broadcast a DC offset "
            f"instead of projecting along a direction (CLAUDE.md §6)."
        )


def _assert_hook_update(value, applied) -> None:
    """In-hook guard, AFTER computing the update: the projection to subtract / the
    delta to add must match `value`'s shape exactly, so it lands on the residual it
    was computed from rather than broadcasting onto a mismatched shape."""
    if tuple(applied.shape) != tuple(value.shape):
        raise SteeringShapeError(
            f"adaptive hook update has shape {tuple(applied.shape)} but the residual "
            f"is {tuple(value.shape)}; refusing to broadcast a mismatched update onto "
            f"the residual stream."
        )


def apply_adaptive_ablation_perlayer(model, vector, coeff: float | None = None,
                                     *, all_resid_points: bool = False):
    """Adaptively ablate a per-layer `(n_layers, d_model)` stack — remove, at each
    layer, that layer's OWN direction from the residual stream:

        at layer L, r̂_L = unit(vector[L]);  x ← x − (x · r̂_L) r̂_L

    The coefficient is not a hand-tuned scalar: it is the per-position projection
    `(x · r̂_L)`, computed inside the hook, so it adapts to every token and no
    coeff sweep is needed (the dot product sets the dose). `coeff` is accepted but
    ignored (ablation has no dose) so the signature matches the standard
    `apply(model, vector, coeff)` method contract.

    This is `apply_directional_ablation`'s per-layer sibling. The difference is the
    convention: that function takes ONE `(d_model,)` direction reused across every
    layer and routes it through `check_direction` (the 1-D guard); this one takes
    the `(n_layers, d_model)` bias-steering stack (from `build_mean_difference`)
    and uses `assert_steering_shape`. Keeping the two conventions apart is
    load-bearing — a 1-D vector reaching per-layer indexing yields a scalar and
    silently broadcasts a DC offset, the class of bug that voided the 2025 refusal
    arms (CLAUDE.md §6, docs/REVIVAL_AUDIT.md). A 1-D vector fails loud here.

    `all_resid_points=False` (default) hooks only each layer's `resid_pre`, matching
    `apply_resid_pre_add`'s surface so additive-vs-adaptive is a like-for-like
    comparison. `all_resid_points=True` also hooks `attn_out`/`mlp_out` at every
    layer (like `apply_directional_ablation`), removing the direction from the whole
    residual stream. In both modes every hook at layer L uses `r̂_L` — the point
    where the residual is read decides *where*, the layer decides *which* direction.
    """
    import functools

    n_layers = model.cfg.n_layers
    assert_steering_shape(vector, n_layers, getattr(model.cfg, "d_model", None))
    r_hat = unit_perlayer(vector)  # (n_layers, d_model), each row a unit direction

    def _ablate(value, hook, r):
        r = r.to(value.dtype).to(value.device)
        _assert_hook_direction(value, r)      # r is 1-D d_model — before the matmul
        proj = (value @ r).unsqueeze(-1) * r  # (batch, seq, 1) * (d_model,)
        _assert_hook_update(value, proj)      # proj matches the residual
        value -= proj
        return value

    if all_resid_points:
        # Three hook points per layer, each keyed to that layer's row, so
        # attn_out/mlp_out get r̂_L too. `_grouped_resid_points` yields (layer, name).
        return [
            (name, functools.partial(_ablate, r=r_hat[layer]))
            for layer, name in _grouped_resid_points(n_layers)
        ]
    return [
        (name, functools.partial(_ablate, r=r_hat[layer]))
        for layer, name in enumerate(resid_pre_hook_names(n_layers))
    ]


def _grouped_resid_points(n_layers: int):
    """Yield `(layer, hook_name)` for all three residual points of every layer:
    `resid_pre`, `attn_out`, `mlp_out`. Same hook set as
    `all_resid_stream_hook_names`, but paired with the layer index so a per-layer
    method can key each hook to that layer's direction row `r̂_L`. Kept separate
    from `all_resid_stream_hook_names` (a pure name list, for the single-direction
    ablation) so that function's contract is unchanged."""
    for layer in range(n_layers):
        for point in ("hook_resid_pre", "hook_attn_out", "hook_mlp_out"):
            yield layer, f"blocks.{layer}.{point}"


def apply_adaptive_additive_perlayer(model, vector, coeff: float,
                                     *, all_resid_points: bool = False):
    """Adaptive *additive* steering — the removal's counterpart. Instead of zeroing
    the projection, drive it to a target magnitude `coeff` regardless of where it
    started, at each layer along that layer's own direction:

        at layer L, r̂_L = unit(vector[L]);  x ← x + (coeff − (x · r̂_L)) r̂_L

    After the hook, `(x · r̂_L) == coeff` at every position — a self-scaling
    alternative to the swept scalar in `apply_resid_pre_add`, where the amount
    added depends on how far the residual already sat along the direction. This is
    distinct from removal: ablation is the `coeff → 0` special case. Use it to
    *set* the opinion component rather than nudge it.

    Same convention guard as `apply_adaptive_ablation_perlayer`: takes the
    `(n_layers, d_model)` stack, per-layer unit-normalizes, and a 1-D vector fails
    loud."""
    import functools

    n_layers = model.cfg.n_layers
    assert_steering_shape(vector, n_layers, getattr(model.cfg, "d_model", None))
    r_hat = unit_perlayer(vector)

    def _drive(value, hook, r, target):
        r = r.to(value.dtype).to(value.device)
        _assert_hook_direction(value, r)                  # r is 1-D d_model — before the matmul
        delta = (target - (value @ r)).unsqueeze(-1) * r  # pin projection to target
        _assert_hook_update(value, delta)                 # delta matches the residual
        value += delta
        return value

    if all_resid_points:
        return [
            (name, functools.partial(_drive, r=r_hat[layer], target=coeff))
            for layer, name in _grouped_resid_points(n_layers)
        ]
    return [
        (name, functools.partial(_drive, r=r_hat[layer], target=coeff))
        for layer, name in enumerate(resid_pre_hook_names(n_layers))
    ]


def apply_adaptive_additive_linear_floor(model, vector, coeff: float = 1.0,
                                         *, denom: float | None = None,
                                         all_resid_points: bool = False):
    """Adaptive additive steering with a per-layer LINEAR target, applied as a
    one-sided floor/ceiling rather than `apply_adaptive_additive_perlayer`'s hard
    pin.

    At layer L (1-indexed), `target_L = coeff * L / denom` -- later layers get a
    larger target than earlier ones, instead of one global scalar repeated at
    every layer. `denom` defaults to the model's own `n_layers`, so the ramp
    reaches exactly `coeff` at the deepest layer (`target_{n_layers} == coeff`)
    regardless of how many layers the model has; pass an explicit `denom` to
    reproduce a fixed-denominator schedule instead (e.g. the first committed
    `adaptive_add_linear` run on Qwen3-8b, n_layers=36, used a literal `denom=52`
    predating this default, so its targets topped out at `coeff*36/52`, not
    `coeff` -- see experiments/adaptive_vs_fixed/GPU_RUN_LOG.md).

    The pin-to-target sibling forces `(x·r̂_L) == target_L` exactly, even when that
    means SUBTRACTING a large existing projection back down to hit a small target
    -- on Qwen3-8b that produced degenerate repetition-loop output at every tested
    target (2, 4, 8), because deep layers' natural projections (~10^1-10^2) are far
    above any of those targets (see experiments/adaptive_vs_fixed/GPU_RUN_LOG.md).
    This variant is one-sided instead:

        delta = target_L - (x·r̂_L)
        if target_L >= 0: delta = max(delta, 0)   # floor: never subtract
        else:             delta = min(delta, 0)   # ceiling: never add
        x <- x + delta · r̂_L

    "Do not subtract if the model already has an existing vector coefficient
    greater than that value" -- a token already at/above a positive target (or
    at/below a negative one) is left untouched. `coeff` sets the sign/scale of
    the whole per-layer ramp, so the standard `apply(model, vector, coeff)`
    contract still works: `_evaluate_and_persist` calls this with
    `+coeffs.opinion` for STEERED_POS (a rising floor) and `-coeffs.neutral` for
    STEERED_NEG (a falling ceiling, same schedule mirrored negative).

    Same shape guard and per-layer-unit-direction convention as its two adaptive
    siblings; a 1-D vector fails loud (CLAUDE.md §6)."""
    import functools

    import torch

    n_layers = model.cfg.n_layers
    assert_steering_shape(vector, n_layers, getattr(model.cfg, "d_model", None))
    denom = denom if denom is not None else n_layers
    r_hat = unit_perlayer(vector)

    def _floor(value, hook, r, target):
        r = r.to(value.dtype).to(value.device)
        _assert_hook_direction(value, r)          # r is 1-D d_model — before the matmul
        proj = value @ r
        delta = target - proj
        delta = torch.clamp(delta, min=0.0) if target >= 0 else torch.clamp(delta, max=0.0)
        applied = delta.unsqueeze(-1) * r
        _assert_hook_update(value, applied)        # applied matches the residual
        value += applied
        return value

    targets = [coeff * (layer + 1) / denom for layer in range(n_layers)]
    if all_resid_points:
        return [
            (name, functools.partial(_floor, r=r_hat[layer], target=targets[layer]))
            for layer, name in _grouped_resid_points(n_layers)
        ]
    return [
        (name, functools.partial(_floor, r=r_hat[layer], target=targets[layer]))
        for layer, name in enumerate(resid_pre_hook_names(n_layers))
    ]


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


def capture_last(cache, n_layers: int):
    """Last-token `resid_pre` per layer -> (n_layers, d_model).

    The Phase-3 "new technique": it changes only how residuals are collected
    (final token instead of the mean over tokens) and reuses `build`/`apply`
    unchanged — demonstrating that a method overrides just the piece that differs.
    """
    import torch

    per_layer = []
    for layer in range(n_layers):
        resid = cache[f"blocks.{layer}.hook_resid_pre"]  # (1, seq, d_model)
        assert resid.ndim == 3 and resid.shape[0] == 1, (
            f"resid_pre at layer {layer}: expected (1, seq, d_model), got "
            f"{tuple(resid.shape)} — capture assumes a single, unbatched response"
        )
        resid = resid[:, -1, :].squeeze(0)               # last token -> (d_model,)
        per_layer.append(resid.detach().clone())
    out = torch.stack(per_layer).cpu()                   # off-GPU, see capture_mean (#9)
    assert out.ndim == 2 and out.shape[0] == n_layers, (
        f"captured residual: expected (n_layers, d_model), got {tuple(out.shape)}"
    )
    return out


# --------------------------------------------------------------------------- #
# Refusal-direction interventions (Arditi et al., 2024; arXiv:2406.11717).
#
# These operate on a SINGLE direction `(d_model,)` reused across positions, not a
# per-layer `(n_layers, d_model)` stack like `apply_resid_pre_add`. They are the
# two interventions the paper defines, ported to TransformerLens hook points.
# The repro flow (experiment side) calls these directly with a pre-loaded
# direction + its source layer, rather than through capture/build.
# --------------------------------------------------------------------------- #


def all_resid_stream_hook_names(n_layers: int) -> list[str]:
    """Every point the residual stream is read/written, per layer: block input
    (`resid_pre`) + attention output (`attn_out`) + MLP output (`mlp_out`).

    Ablating a direction at all three removes it from the entire residual stream
    (resid_post = resid_pre + attn_out + mlp_out). Mirrors the paper's hooks on
    block-input + attn-output + mlp-output modules. Torch-free."""
    names = []
    for layer in range(n_layers):
        names += [
            f"blocks.{layer}.hook_resid_pre",
            f"blocks.{layer}.hook_attn_out",
            f"blocks.{layer}.hook_mlp_out",
        ]
    return names


def check_direction(model, vector, *, layer: int | None = None):
    """Validate a SINGLE-direction steering vector against `model` before it is
    applied, and return it as a 1-D `(d_model,)` tensor.

    This exists to catch the silent foot-guns of mixing conventions: a refusal
    direction is one `(d_model,)` vector reused across layers, whereas the
    bias-steering vectors are `(n_layers, d_model)` stacks indexed per layer.
    Routing the wrong one here would otherwise broadcast into a plausible-looking
    but wrong result (e.g. `vector[layer]` on a 1-D tensor yields a scalar that is
    silently added everywhere). Turns each such case into a clear error:

    - must be a torch.Tensor, exactly 1-D `(d_model,)` (not a per-layer stack or
      the `(n_pos, n_layers, d_model)` grid),
    - length must equal `model.cfg.d_model` (guards a wrong model/direction pair),
    - must be finite (no NaN/Inf from a corrupt load),
    - for act-add, `layer` must be in `[0, n_layers)` — an out-of-range layer
      names a non-existent hook point that would silently never fire.
    """
    import torch

    if not isinstance(vector, torch.Tensor):
        raise TypeError(f"direction must be a torch.Tensor, got {type(vector).__name__}")
    if vector.ndim != 1:
        raise ValueError(
            f"direction must be 1-D (d_model,); got shape {tuple(vector.shape)}. "
            f"Ablation/act-add take a single direction, not a (n_layers, d_model) "
            f"bias-steering stack or the (n_pos, n_layers, d_model) grid."
        )
    d_model = getattr(model.cfg, "d_model", None)
    if d_model is not None and vector.numel() != d_model:
        raise ValueError(
            f"direction has {vector.numel()} elements but model d_model={d_model} "
            f"— wrong model/direction pairing, or a per-layer stack got flattened in."
        )
    if not bool(torch.isfinite(vector).all()):
        raise ValueError("direction contains NaN/Inf")
    if layer is not None:
        n_layers = model.cfg.n_layers
        if not (0 <= layer < n_layers):
            raise ValueError(f"act-add layer {layer} out of range [0, {n_layers})")
    return vector


def unit_direction(vector):
    """`r / (‖r‖ + 1e-8)` — the unit refusal direction `r̂` used for ablation
    (matches the paper's normalization, epsilon and all)."""
    return vector.flatten() / (vector.flatten().norm() + 1e-8)


def apply_directional_ablation(model, vector, coeff: float | None = None):
    """Project the direction out of the residual stream everywhere:

        x ← x − (x · r̂) r̂        with r̂ = vector / ‖vector‖

    at every layer's `resid_pre` / `attn_out` / `mlp_out`, all token positions.
    This is the paper's refusal-*bypass* intervention. `coeff` is ignored
    (ablation has no dose) but accepted so the signature matches the standard
    `apply(model, vector, coeff)` method contract."""
    import functools

    r_hat = unit_direction(check_direction(model, vector))
    n_layers = model.cfg.n_layers

    def _ablate(value, hook, r):
        r = r.to(value.dtype).to(value.device)
        proj = (value @ r).unsqueeze(-1) * r  # (batch, seq, 1) * (d_model,)
        value -= proj
        return value

    return [
        (name, functools.partial(_ablate, r=r_hat))
        for name in all_resid_stream_hook_names(n_layers)
    ]


def apply_actadd_single(model, vector, coeff: float, *, layer: int):
    """Add `coeff · vector` (the RAW, un-normalized direction) at one layer's
    `resid_pre`, all positions:

        x ← x + coeff · r

    The paper's refusal-*induction* intervention. `coeff=+1` induces refusal on
    harmless prompts; `coeff=-1` suppresses it on harmful ones. The raw vector's
    natural norm is the dose, so the direction is deliberately NOT normalized
    here (contrast `apply_directional_ablation`). `layer` is the direction's
    source layer (from its metadata)."""
    import functools

    direction = check_direction(model, vector, layer=layer)

    def _add(value, hook, vec, c):
        value[:, :, :] += c * vec.to(value.dtype).to(value.device)
        return value

    name = f"blocks.{layer}.hook_resid_pre"
    return [(name, functools.partial(_add, vec=direction, c=coeff))]


@dataclass
class SteeringMethod:
    """A named bundle of (capture, build, apply). Defaults = the `mean_diff`
    behavior; override only the piece a new technique changes (§3.5)."""

    name: str
    capture: Callable = capture_mean
    build: Callable = build_mean_difference
    apply: Callable = apply_resid_pre_add
    # Hook points `capture` reads, so generation can cache only those rather than
    # every hook point in the model (see models.generate_with_cache). Override
    # alongside `capture` if a technique reads something other than resid_pre.
    names: Callable = resid_pre_hook_names


register(METHODS, "mean_diff", SteeringMethod("mean_diff"))
# New technique via one override + one registry line; build/apply reused as-is.
register(METHODS, "last_token", SteeringMethod("last_token", capture=capture_last))
# Refusal-direction bypass. Only `apply` is meaningful here — the repro flow loads
# a pre-computed direction and skips capture/build, so those keep their (unused)
# defaults. Activation-addition (`apply_actadd_single`) is NOT registered: it needs
# the direction's source `layer`, which the (model, vector, coeff) contract can't
# carry, so the repro flow calls that function directly.
register(METHODS, "ablation", SteeringMethod("ablation", apply=apply_directional_ablation))
# Adaptive per-layer ablation of the (n_layers, d_model) bias-steering stack: the
# coeff is the per-position dot product, computed in the hook, so there is no dose
# and no sweep (SCOPE_adaptive_steering.md). Only `apply` changes; capture/build
# stay the mean_diff defaults, matching the SteeringMethod override pattern above.
# NOTE: `apply` ignores its coeff, so a run's STEERED_POS and STEERED_NEG arms are
# identical for this method — removal is sign-agnostic (see the DoD#4 summary).
register(METHODS, "adaptive_ablation",
         SteeringMethod("adaptive_ablation", apply=apply_adaptive_ablation_perlayer))
# Adaptive additive counterpart: pins the projection onto each layer's direction to
# the coeff (a target magnitude), rather than removing it. Distinct from removal.
register(METHODS, "adaptive_add",
         SteeringMethod("adaptive_add", apply=apply_adaptive_additive_perlayer))
# One-sided linear-schedule variant: a rising floor/ceiling per layer (coeff*L/denom)
# instead of one global scalar pinned everywhere -- never subtracts an existing
# projection back down (see apply_adaptive_additive_linear_floor's docstring).
register(METHODS, "adaptive_add_linear",
         SteeringMethod("adaptive_add_linear", apply=apply_adaptive_additive_linear_floor))
# Unconditional counterpart to adaptive_add_linear: same coeff*L/denom per-layer
# schedule and unit-direction convention, but ALWAYS adds the full increment (no
# state-dependent clamp) -- isolates the linear-schedule variable from the
# one-sidedness for the three-way comparison (see apply_linear_add_perlayer's
# docstring and docs/SCOPE_linear_scaling_isolation.md).
register(METHODS, "linear_add",
         SteeringMethod("linear_add", apply=apply_linear_add_perlayer))


# --------------------------------------------------------------------------- #
# Refusal-direction EXTRACTION (arXiv:2406.11717, generate_directions stage).
#
# Unlike capture_mean / capture_last (which average / last-token over the model's
# RESPONSE), extraction reads the residual stream of the PROMPT itself, at the
# last few post-instruction template positions, with NO generation. The paper's
# `get_mean_activations` hooks each block's input (== hook_resid_pre) and keeps
# positions `range(-n_pos, 0)` where n_pos = len(end-of-instruction template
# tokens) — model-specific (qwen/gemma/llama3=5; llama-2/yi=6). The grid is
# mean_harmful - mean_harmless, shape (n_pos, n_layers, d_model).
#
# capture_prompt_positions here consumes a single prompt's run_with_cache output;
# the driver (refusal_extract.py) produces that cache from a forward pass on the
# formatted prompt (system=None, upstream literal template) and buckets BY LABEL.
# --------------------------------------------------------------------------- #

# Fallback n_pos for the method-contract path. The real extraction driver passes
# the model-specific n_pos (= len of end-of-instruction tokens) explicitly.
DEFAULT_REFUSAL_N_POSITIONS = 5


def capture_prompt_positions(cache, n_layers: int, n_pos: int = DEFAULT_REFUSAL_N_POSITIONS):
    """Last `n_pos` prompt-token `resid_pre` per layer -> (n_pos, n_layers, d_model).

    Consumes one prompt's cache (each `blocks.{l}.hook_resid_pre` is
    `(1, seq, d_model)`), takes the final `n_pos` positions, and orders axes as
    (position, layer, d_model) to match the paper's `mean_diffs` grid. Slicing to a
    fixed `n_pos` (rather than keeping the whole prompt) is what lets caches from
    different-length prompts stack in `build`.
    """
    import torch

    per_layer = []
    for layer in range(n_layers):
        resid = cache[f"blocks.{layer}.hook_resid_pre"]  # (1, seq, d_model)
        tail = resid[:, -n_pos:, :].squeeze(0)           # (n_pos, d_model)
        per_layer.append(tail.detach().clone())
    # (n_layers, n_pos, d_model) -> (n_pos, n_layers, d_model)
    return torch.stack(per_layer).transpose(0, 1).contiguous()


def build_refusal_grid(resids_by_label: dict, contrast: tuple = ("harmful", "harmless")):
    """mean_harmful - mean_harmless per (pos, layer) -> (n_pos, n_layers, d_model).

    The refusal candidate grid (the paper's `mean_diffs`). `contrast` is
    (positive_label, negative_label) = ("harmful", "harmless"); the mechanics are
    identical to `build_mean_difference` (mean over examples, then pos - neg) — the
    difference is the grouping is BY DATASET LABEL, not by a judge verdict.
    """
    return build_mean_difference(resids_by_label, contrast)


# Extraction method: capture prompt positions, build the mean-difference grid.
# `apply` keeps the mean_diff default (unused — extraction produces a grid, it does
# not steer). The driver calls capture_prompt_positions with the model-specific
# n_pos directly; this registration is for framework discoverability + testing.
register(METHODS, "refusal_extract",
         SteeringMethod("refusal_extract",
                        capture=capture_prompt_positions,
                        build=build_refusal_grid))

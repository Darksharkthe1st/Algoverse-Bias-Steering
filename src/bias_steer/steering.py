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
        resid = resid.mean(dim=1).squeeze(0)             # (d_model,)
        per_layer.append(resid.detach().clone())
    return torch.stack(per_layer)


def build_mean_difference(resids_by_label: dict, contrast: tuple):
    """`mean(pos) - mean(neg)` per layer -> (n_layers, d_model).

    `contrast` is `(positive_label, negative_label)`; the vector points toward the
    positive pole. Ports the notebook's `get_opinion_vec_from_resids` (opinion − neutral).
    """
    import torch

    pos_label, neg_label = contrast
    pos = torch.stack(resids_by_label[pos_label]).mean(dim=0)
    neg = torch.stack(resids_by_label[neg_label]).mean(dim=0)
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
        resid = resid[:, -1, :].squeeze(0)               # last token -> (d_model,)
        per_layer.append(resid.detach().clone())
    return torch.stack(per_layer)


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

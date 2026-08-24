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
    out = torch.stack(per_layer)
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
    d_model = model.cfg.d_model
    assert vector.ndim == 2 and tuple(vector.shape) == (n_layers, d_model), (
        f"steering vector: expected (n_layers, d_model) = {(n_layers, d_model)}, "
        f"got {tuple(vector.shape)} — a 1-D vector would broadcast as a scalar DC "
        "offset, not a direction (see CLAUDE.md §6 / docs/REVIVAL_AUDIT.md)"
    )
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
        assert resid.ndim == 3 and resid.shape[0] == 1, (
            f"resid_pre at layer {layer}: expected (1, seq, d_model), got "
            f"{tuple(resid.shape)} — capture assumes a single, unbatched response"
        )
        resid = resid[:, -1, :].squeeze(0)               # last token -> (d_model,)
        per_layer.append(resid.detach().clone())
    out = torch.stack(per_layer)
    assert out.ndim == 2 and out.shape[0] == n_layers, (
        f"captured residual: expected (n_layers, d_model), got {tuple(out.shape)}"
    )
    return out


@dataclass
class SteeringMethod:
    """A named bundle of (capture, build, apply). Defaults = the `mean_diff`
    behavior; override only the piece a new technique changes (§3.5)."""

    name: str
    capture: Callable = capture_mean
    build: Callable = build_mean_difference
    apply: Callable = apply_resid_pre_add


register(METHODS, "mean_diff", SteeringMethod("mean_diff"))
# New technique via one override + one registry line; build/apply reused as-is.
register(METHODS, "last_token", SteeringMethod("last_token", capture=capture_last))

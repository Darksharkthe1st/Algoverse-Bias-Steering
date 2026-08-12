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

    r_hat = unit_direction(vector)
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

    direction = vector.flatten()

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

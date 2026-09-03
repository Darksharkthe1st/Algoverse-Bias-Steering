"""Adaptive additive steering, LINEAR-SCHEDULE one-sided variant, with the
Qwen3-8b opinion vector — follow-up to `adaptive_add_qwen3_8b.py` after that
config's target=2/4/8 sweep produced degenerate repetition-loop output at every
tested target (see experiments/adaptive_vs_fixed/GPU_RUN_LOG.md).

Tests `adaptive_add_linear`
(`steering.apply_adaptive_additive_linear_floor`): at layer L (1-indexed),
target_L = coeff * L / 52 — a per-layer RAMP instead of one global scalar pinned
at every layer — applied as a one-sided floor/ceiling rather than a hard pin:

    delta = target_L − (x·r̂_L)
    if target_L >= 0: delta = max(delta, 0)   # floor: never subtract
    else:             delta = min(delta, 0)   # ceiling: never add

This is the fix for the previous config's failure mode: that method forced
(x·r̂_L) to EXACTLY its target even when that meant subtracting a large existing
projection back down, which is what wrecked deep-layer generations (natural
per-layer projections there run ~10^1–10^2, per the calibration measurement in
GPU_RUN_LOG.md). Here, a token already at/above its layer's (positive) target
is left untouched — "do not subtract if the model already has an existing
vector coefficient greater than that value."

`coeffs.opinion`/`coeffs.neutral` scale the whole ramp (not a target magnitude
directly): coeff=1.0 reproduces exactly target_L = L/52 for STEERED_POS; the NEG
arm gets coeff=-1.0 (via `-coeffs.neutral`), mirroring the ramp negative (a
falling ceiling). Same merged-tree requirement, model, vector, dataset, and
judge as the other three siblings — see `adaptive_ablation_qwen3_8b.py` for full
provenance.

    python -m src.bias_steer run configs/exp/adaptive_add_linear_qwen3_8b.py \
        --vector runs/20260901-092009_anchor-qwen3-8b_qwen3-8b/steering_vector.safetensors
"""

from src.bias_steer.config import (
    ExperimentConfig, DatasetSpec, SampleSpec, JudgeSpec, Coeffs,
)

config = ExperimentConfig(
    label="adaptive-add-linear qwen3-8b",
    models=["qwen3-8b"],
    dataset=DatasetSpec(
        name="snapshot",
        path="datasets/Snapshots/log_103_comparison_200.json",
        train_split=0.5, shuffle=False,
    ),
    sample=SampleSpec(),
    judge=JudgeSpec(name="neutrality"),
    # Ramp SCALE, not a target magnitude — see docstring. 1.0 -> target_L = L/52.
    coeffs=Coeffs(opinion=1.0, neutral=1.0),
    method="adaptive_add_linear",
    max_tokens=128,
    batch_size=16,
)

config.vector_path = "runs/20260901-092009_anchor-qwen3-8b_qwen3-8b/steering_vector.safetensors"

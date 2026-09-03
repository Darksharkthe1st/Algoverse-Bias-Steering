"""Adaptive additive steering, LINEAR-SCHEDULE one-sided variant, coeff=8, FULL
RAMP — reaches exactly coeff at the deepest layer.

Sibling of `adaptive_add_linear_c8_qwen3_8b.py`, same method
(`adaptive_add_linear`) and coeff=8.0, but relies on
`apply_adaptive_additive_linear_floor`'s DEFAULT `denom` (now the model's own
`n_layers`, not a fixed 52 — see steering.py). At layer L (1-indexed),
target_L = coeff * L / n_layers, so for Qwen3-8B (n_layers=36):
layer 1 -> 8/36, layer 2 -> 16/36, ..., layer 36 -> 288/36 = 8.0 EXACTLY.

Why this exists: `adaptive_add_linear_c8_qwen3_8b.py` used the original fixed
`denom=52` (predating the default change), so on this 36-layer model its ramp
topped out at only `8*36/52 ≈ 5.54`, not the full coeff=8 dose `fixed_add` uses
-- plausibly why that run under-steered relative to `fixed_add` even at coeff=8.
This config asks the same question with the ramp actually reaching the full
dose by the last layer, while remaining one-sided (never subtracts an
already-more-intense projection, so it should stay clear of the degeneracy
`adaptive_add`'s hard pin hit) -- see
experiments/adaptive_vs_fixed/GPU_RUN_LOG.md for the full history.

    python -m src.bias_steer run configs/exp/adaptive_add_linear_c8_full_ramp_qwen3_8b.py \
        --vector runs/20260901-092009_anchor-qwen3-8b_qwen3-8b/steering_vector.safetensors
"""

from src.bias_steer.config import (
    ExperimentConfig, DatasetSpec, SampleSpec, JudgeSpec, Coeffs,
)

config = ExperimentConfig(
    label="adaptive-add-linear-c8-full-ramp qwen3-8b",
    models=["qwen3-8b"],
    dataset=DatasetSpec(
        name="snapshot",
        path="datasets/Snapshots/log_103_comparison_200.json",
        train_split=0.5, shuffle=False,
    ),
    sample=SampleSpec(),
    judge=JudgeSpec(name="neutrality"),
    # Ramp SCALE, not a target magnitude. With the default denom=n_layers,
    # 8.0 -> target_L = 8*L/36, reaching exactly 8.0 at the last layer.
    coeffs=Coeffs(opinion=8.0, neutral=8.0),
    method="adaptive_add_linear",
    max_tokens=128,
    batch_size=16,
)

config.vector_path = "runs/20260901-092009_anchor-qwen3-8b_qwen3-8b/steering_vector.safetensors"

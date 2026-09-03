"""Adaptive additive steering, LINEAR-SCHEDULE one-sided variant, coeff=8 —
stronger-ramp sibling of `adaptive_add_linear_qwen3_8b.py`.

Same method (`adaptive_add_linear` / `steering.apply_adaptive_additive_linear_floor`)
and everything else identical to that config — only `coeffs` differs. At layer L
(1-indexed), target_L = coeff * L / 52, so coeff=8.0 gives targets 8/52, 16/52,
..., 288/52 (~5.5 at the deepest layer) instead of coeff=1.0's 1/52..36/52
(~0.02-0.69) — the coeff=1 run's targets were far smaller than `fixed_add`'s
effective per-layer dose (c=8, i.e. (8/36)*||vector[L]|| added once per layer),
which is a likely reason adaptive_add_linear(coeff=1) under-steered relative to
fixed_add. This run tests whether a proportionally larger ramp closes that gap
while staying on the "never subtract an already-more-intense projection" side
that keeps generation coherent (unlike the old hard-pin `adaptive_add`, which
was degenerate even at target=8 applied identically at every layer — see
experiments/adaptive_vs_fixed/GPU_RUN_LOG.md). The key difference from that
failure mode: here the LARGEST target (5.5, at layer 36) is still small next to
that layer's natural baseline projection (median ~109, per the calibration
measurement), so it should still act as a one-sided floor most tokens already
clear, not a forced reset.

    python -m src.bias_steer run configs/exp/adaptive_add_linear_c8_qwen3_8b.py \
        --vector runs/20260901-092009_anchor-qwen3-8b_qwen3-8b/steering_vector.safetensors
"""

from src.bias_steer.config import (
    ExperimentConfig, DatasetSpec, SampleSpec, JudgeSpec, Coeffs,
)

config = ExperimentConfig(
    label="adaptive-add-linear-c8 qwen3-8b",
    models=["qwen3-8b"],
    dataset=DatasetSpec(
        name="snapshot",
        path="datasets/Snapshots/log_103_comparison_200.json",
        train_split=0.5, shuffle=False,
    ),
    sample=SampleSpec(),
    judge=JudgeSpec(name="neutrality"),
    # Ramp SCALE, not a target magnitude — see docstring. 8.0 -> target_L = 8*L/52.
    coeffs=Coeffs(opinion=8.0, neutral=8.0),
    method="adaptive_add_linear",
    max_tokens=128,
    batch_size=16,
)

config.vector_path = "runs/20260901-092009_anchor-qwen3-8b_qwen3-8b/steering_vector.safetensors"

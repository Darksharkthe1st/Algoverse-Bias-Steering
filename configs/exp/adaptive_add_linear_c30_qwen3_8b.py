"""Adaptive additive steering, LINEAR-SCHEDULE one-sided variant, coeff=30 —
next dose after `adaptive_add_linear_c16_qwen3_8b.py`, queued conditionally on
c20 staying coherent (see logs/check_coherence.py and
experiments/adaptive_vs_fixed/GPU_RUN_LOG.md).

Same method (`adaptive_add_linear`) and default `denom` (the model's own
`n_layers`) — only `coeffs` differs. For Qwen3-8B (n_layers=36): layer 1 ->
30/36, ..., layer 36 -> 30.0 exactly (3.75x `fixed_add`'s c=8 dose).

Why: coeff=16 nearly matched `fixed_add` on the POS arm (68/150 vs 66/146) and
closed most of the gap on NEG (36/50 vs 45/54) while staying fully coherent —
this tests whether pushing further closes the remaining NEG gap, or whether
coherence finally breaks down at this larger dose.

    python -m src.bias_steer run configs/exp/adaptive_add_linear_c30_qwen3_8b.py \
        --vector runs/20260901-092009_anchor-qwen3-8b_qwen3-8b/steering_vector.safetensors
"""

from src.bias_steer.config import (
    ExperimentConfig, DatasetSpec, SampleSpec, JudgeSpec, Coeffs,
)

config = ExperimentConfig(
    label="adaptive-add-linear-c30 qwen3-8b",
    models=["qwen3-8b"],
    dataset=DatasetSpec(
        name="snapshot",
        path="datasets/Snapshots/log_103_comparison_200.json",
        train_split=0.5, shuffle=False,
    ),
    sample=SampleSpec(),
    judge=JudgeSpec(name="neutrality"),
    coeffs=Coeffs(opinion=30.0, neutral=30.0),
    method="adaptive_add_linear",
    max_tokens=128,
    batch_size=16,
)

config.vector_path = "runs/20260901-092009_anchor-qwen3-8b_qwen3-8b/steering_vector.safetensors"

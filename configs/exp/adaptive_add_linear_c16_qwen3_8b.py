"""Adaptive additive steering, LINEAR-SCHEDULE one-sided variant, coeff=16 —
stronger-dose sibling of `adaptive_add_linear_c8_full_ramp_qwen3_8b.py`.

Same method (`adaptive_add_linear`) and default `denom` (the model's own
`n_layers`, so the ramp reaches exactly `coeff` at the deepest layer) — only
`coeffs` differs. For Qwen3-8B (n_layers=36): layer 1 -> 16/36, layer 2 -> 32/36,
..., layer 36 -> 16.0 exactly (2x `fixed_add`'s c=8 dose).

Why: the coeff=8 full-ramp run still trailed `fixed_add` on both arms despite
reaching the nominal dose exactly at the last layer (see
experiments/adaptive_vs_fixed/GPU_RUN_LOG.md) — tests whether a larger coeff
closes more of that gap while the one-sided floor/ceiling keeps generation
coherent (unlike `adaptive_add`'s hard pin, which was degenerate even at
target=8 applied identically everywhere).

    python -m src.bias_steer run configs/exp/adaptive_add_linear_c16_qwen3_8b.py \
        --vector runs/20260901-092009_anchor-qwen3-8b_qwen3-8b/steering_vector.safetensors
"""

from src.bias_steer.config import (
    ExperimentConfig, DatasetSpec, SampleSpec, JudgeSpec, Coeffs,
)

config = ExperimentConfig(
    label="adaptive-add-linear-c16 qwen3-8b",
    models=["qwen3-8b"],
    dataset=DatasetSpec(
        name="snapshot",
        path="datasets/Snapshots/log_103_comparison_200.json",
        train_split=0.5, shuffle=False,
    ),
    sample=SampleSpec(),
    judge=JudgeSpec(name="neutrality"),
    coeffs=Coeffs(opinion=16.0, neutral=16.0),
    method="adaptive_add_linear",
    max_tokens=128,
    batch_size=16,
)

config.vector_path = "runs/20260901-092009_anchor-qwen3-8b_qwen3-8b/steering_vector.safetensors"

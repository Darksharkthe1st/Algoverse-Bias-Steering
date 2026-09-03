"""Linear-schedule UNCONDITIONAL additive steering, coeff=20 — confound-isolation
counterpart to `adaptive_add_linear_c20_qwen3_8b.py` (default denom).

Tests `linear_add` (`steering.apply_linear_add_perlayer`): ALWAYS adds
`increment_L = coeff * L / n_layers` along r̂_L, no clamp. Same target formula /
unit direction / default denom (n_layers=36 -> deepest-layer increment = 20.0)
as `adaptive_add_linear` at this coeff; only the one-sided clamp is removed.

CLEAN (B)-ISOLATION PAIR: compare against `runs/20260903-020329_adaptive-add-
linear-c20-qwen3-8b_...` — the coeff where `adaptive_add_linear`'s POS arm first
EXCEEDED `fixed_add` (POS 83/152 vs 66/146; NEG 35/48). The isolation question
here: does the UNCONDITIONAL linear ramp reach fixed_add-exceeding POS at the
same coeff, or does the one-sidedness account for it? Third arm: `fixed_add`
(`mean_diff` coeff=8). See docs/SCOPE_linear_scaling_isolation.md.

    python -m src.bias_steer run configs/exp/linear_add_c20_qwen3_8b.py \
        --vector runs/20260901-092009_anchor-qwen3-8b_qwen3-8b/steering_vector.safetensors
"""

from src.bias_steer.config import (
    ExperimentConfig, DatasetSpec, SampleSpec, JudgeSpec, Coeffs,
)

config = ExperimentConfig(
    label="linear-add-c20 qwen3-8b",
    models=["qwen3-8b"],
    dataset=DatasetSpec(
        name="snapshot",
        path="datasets/Snapshots/log_103_comparison_200.json",
        train_split=0.5, shuffle=False,
    ),
    sample=SampleSpec(),
    judge=JudgeSpec(name="neutrality"),
    # Default denom=n_layers -> increment_L = 20*L/36, reaching 20.0 at the last layer.
    coeffs=Coeffs(opinion=20.0, neutral=20.0),
    method="linear_add",
    max_tokens=128,
    batch_size=16,
)

config.vector_path = "runs/20260901-092009_anchor-qwen3-8b_qwen3-8b/steering_vector.safetensors"

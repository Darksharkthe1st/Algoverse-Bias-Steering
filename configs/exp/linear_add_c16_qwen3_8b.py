"""Linear-schedule UNCONDITIONAL additive steering, coeff=16 — confound-isolation
counterpart to `adaptive_add_linear_c16_qwen3_8b.py` (default denom).

Tests `linear_add` (`steering.apply_linear_add_perlayer`): ALWAYS adds
`increment_L = coeff * L / n_layers` along r̂_L, no clamp. Same target formula /
unit direction / default denom (n_layers=36 -> deepest-layer increment = 16.0)
as `adaptive_add_linear` at this coeff; only the state-dependent one-sided clamp
is removed.

CLEAN (B)-ISOLATION PAIR: compare against `runs/20260903-014921_adaptive-add-
linear-c16-qwen3-8b_...` (POS 68/150, NEG 36/50, coherent). Only the clamp
differs, so any gap is the clamp's effect. Third arm: `fixed_add` (`mean_diff`
coeff=8: POS 66/146, NEG 45/54). See docs/SCOPE_linear_scaling_isolation.md.

    python -m src.bias_steer run configs/exp/linear_add_c16_qwen3_8b.py \
        --vector runs/20260901-092009_anchor-qwen3-8b_qwen3-8b/steering_vector.safetensors
"""

from src.bias_steer.config import (
    ExperimentConfig, DatasetSpec, SampleSpec, JudgeSpec, Coeffs,
)

config = ExperimentConfig(
    label="linear-add-c16 qwen3-8b",
    models=["qwen3-8b"],
    dataset=DatasetSpec(
        name="snapshot",
        path="datasets/Snapshots/log_103_comparison_200.json",
        train_split=0.5, shuffle=False,
    ),
    sample=SampleSpec(),
    judge=JudgeSpec(name="neutrality"),
    # Default denom=n_layers -> increment_L = 16*L/36, reaching 16.0 at the last layer.
    coeffs=Coeffs(opinion=16.0, neutral=16.0),
    method="linear_add",
    max_tokens=128,
    batch_size=16,
)

config.vector_path = "runs/20260901-092009_anchor-qwen3-8b_qwen3-8b/steering_vector.safetensors"

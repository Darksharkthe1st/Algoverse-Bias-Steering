"""Linear-schedule UNCONDITIONAL additive steering, coeff=8 — confound-isolation
counterpart to `adaptive_add_linear_c8_full_ramp_qwen3_8b.py` (default denom).

Tests `linear_add` (`steering.apply_linear_add_perlayer`): ALWAYS adds
`increment_L = coeff * L / n_layers` along the per-layer unit direction r̂_L, no
clamp, no read of the current projection. This is `adaptive_add_linear` with its
one-sided floor/ceiling removed — same target formula, same unit direction, same
default denom (n_layers=36, so the deepest layer's increment is exactly 8.0).

CLEAN (B)-ISOLATION PAIR: compare this run against
`runs/20260903-012517_adaptive-add-linear-c8-full-ramp-qwen3-8b_...` (coeff=8,
default denom) — the ONLY thing that differs between the two methods at this
coeff is the state-dependent clamp, so any gap is that clamp's effect. Do NOT
pair it with `runs/20260903-011119_...` (coeff=8, the old fixed denom=52) — that
one's ramp tops out at 8*36/52≈5.54, a denom mismatch that would confound the
comparison. The third arm is `fixed_add` (`mean_diff`, coeff=8: POS 66/146, NEG
45/54): flat + unconditional + raw vector[L] direction.

See docs/SCOPE_linear_scaling_isolation.md and
experiments/adaptive_vs_fixed/GPU_RUN_LOG.md for why this isolation matters
(adaptive_add_linear's growth-with-coeff and coeff=30 coherence caveat could
come from the schedule, the one-sidedness, or their interaction).

    python -m src.bias_steer run configs/exp/linear_add_c8_qwen3_8b.py \
        --vector runs/20260901-092009_anchor-qwen3-8b_qwen3-8b/steering_vector.safetensors
"""

from src.bias_steer.config import (
    ExperimentConfig, DatasetSpec, SampleSpec, JudgeSpec, Coeffs,
)

config = ExperimentConfig(
    label="linear-add-c8 qwen3-8b",
    models=["qwen3-8b"],
    dataset=DatasetSpec(
        name="snapshot",
        path="datasets/Snapshots/log_103_comparison_200.json",
        train_split=0.5, shuffle=False,
    ),
    sample=SampleSpec(),
    judge=JudgeSpec(name="neutrality"),
    # Default denom=n_layers -> increment_L = 8*L/36, reaching 8.0 at the last layer.
    coeffs=Coeffs(opinion=8.0, neutral=8.0),
    method="linear_add",
    max_tokens=128,
    batch_size=16,
)

config.vector_path = "runs/20260901-092009_anchor-qwen3-8b_qwen3-8b/steering_vector.safetensors"

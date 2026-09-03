"""Linear-schedule UNCONDITIONAL additive steering, coeff=1 — the confound-
isolation counterpart to `adaptive_add_linear_qwen3_8b.py`, on the same Qwen3-8b
opinion vector.

Tests `linear_add` (`steering.apply_linear_add_perlayer`): at layer L
(1-indexed), it ALWAYS adds `increment_L = coeff * L / n_layers` along the
per-layer unit direction r̂_L, regardless of the token's current projection —

    x <- x + increment_L * r̂_L      # every position, every time, no clamp

This is `adaptive_add_linear` with its state-dependent, one-sided floor/ceiling
clamp REMOVED: same per-layer linear target formula, same unit-direction
convention, same default denom (the model's own n_layers) — only the clamp
differs. It isolates the linear-schedule half of `adaptive_add_linear` from its
state-dependent half so a three-way GPU comparison can attribute that method's
growth-with-coeff and its coeff=30 coherence caveat to (A) the schedule, (B) the
one-sidedness, or their interaction. See docs/SCOPE_linear_scaling_isolation.md
and experiments/adaptive_vs_fixed/GPU_RUN_LOG.md.

DENOM NOTE (read before comparing this specific point). Every `linear_add`
config in this sweep uses the DEFAULT denom = n_layers = 36 (full ramp: the
deepest layer's increment is exactly `coeff`), which matches the
`adaptive_add_linear` runs at coeff=8-full-ramp/16/20/30 exactly — those are the
clean, single-variable (B)-isolation pairings. The coeff=1 `adaptive_add_linear`
run (`runs/20260903-004434_...`) predates the default-denom change and used a
fixed `denom=52` (targets L/52, topping out at 36/52≈0.69), so THIS coeff=1
pairing is only approximate on denom (L/36 vs L/52). Both are tiny next to deep
layers' natural projections (~10^1-10^2), so the endpoint is weak either way;
treat the exact clean isolation as living at coeff=8/16/20/30.

    python -m src.bias_steer run configs/exp/linear_add_c1_qwen3_8b.py \
        --vector runs/20260901-092009_anchor-qwen3-8b_qwen3-8b/steering_vector.safetensors
"""

from src.bias_steer.config import (
    ExperimentConfig, DatasetSpec, SampleSpec, JudgeSpec, Coeffs,
)

config = ExperimentConfig(
    label="linear-add-c1 qwen3-8b",
    models=["qwen3-8b"],
    dataset=DatasetSpec(
        name="snapshot",
        path="datasets/Snapshots/log_103_comparison_200.json",
        train_split=0.5, shuffle=False,
    ),
    sample=SampleSpec(),
    judge=JudgeSpec(name="neutrality"),
    # Ramp SCALE. Default denom=n_layers -> increment_L = 1*L/36, reaching 1.0 at
    # the last layer. NEG arm gets -coeffs.neutral (the mirror-negative ramp).
    coeffs=Coeffs(opinion=1.0, neutral=1.0),
    method="linear_add",
    max_tokens=128,
    batch_size=16,
)

config.vector_path = "runs/20260901-092009_anchor-qwen3-8b_qwen3-8b/steering_vector.safetensors"

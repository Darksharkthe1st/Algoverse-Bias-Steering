"""Adaptive additive steering (pin-to-target) with the Qwen3-8b opinion vector —
EXPLORATORY sibling to the ablation/fixed-add comparison.

Tests the optional `adaptive_add` method: `x ← x + (target − (x·r̂_L)) r̂_L`, which
drives the projection onto each layer's direction to `target` regardless of where
it started (a self-scaling alternative to the swept scalar). Here `coeffs.opinion`
is the TARGET projection magnitude, not a dose.

CALIBRATION CAVEAT — read before trusting the number. `target` is a SINGLE scalar
applied at every layer, but the per-layer opinion directions have very different
norms (‖vector[L]‖ ranges ~0.07–0.66 across the 36 layers), so one target does not
sit at the same relative point on every layer. The value below is a STARTING guess,
not a calibrated operating point. On the GPU box, first measure the baseline
per-position projection distribution `(x·r̂_L)` on a handful of prompts, then set
`target` from it (and consider a small sweep: 2, 4, 8). Report which target was
used — a judged number is meaningless without it.

Same merged-tree requirement, model, vector, dataset, and judge as the two primary
siblings (`adaptive_ablation_qwen3_8b.py`, `fixed_add_qwen3_8b.py`) — see the
ablation config for full provenance and caveats. Distinct from removal: ablation is
the `target → 0` special case.

    python -m src.bias_steer run configs/exp/adaptive_add_qwen3_8b.py \
        --vector runs/20260901-092009_anchor-qwen3-8b_qwen3-8b/steering_vector.safetensors
"""

from src.bias_steer.config import (
    ExperimentConfig, DatasetSpec, SampleSpec, JudgeSpec, Coeffs,
)

config = ExperimentConfig(
    label="adaptive-add qwen3-8b",
    models=["qwen3-8b"],
    dataset=DatasetSpec(
        name="snapshot",
        path="datasets/Snapshots/log_103_comparison_200.json",
        train_split=0.5, shuffle=False,
    ),
    sample=SampleSpec(),
    judge=JudgeSpec(name="neutrality"),
    # TARGET projection magnitude (NOT a dose). Uncalibrated starting guess — see
    # the calibration caveat above before reporting anything from this arm. The
    # `neutral` slot drives the NEG arm to target −4.0 (toward neutral).
    coeffs=Coeffs(opinion=4.0, neutral=4.0),
    method="adaptive_add",
    max_tokens=128,
    batch_size=16,
)

config.vector_path = "runs/20260901-092009_anchor-qwen3-8b_qwen3-8b/steering_vector.safetensors"

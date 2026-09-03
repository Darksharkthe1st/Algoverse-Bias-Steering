"""Adaptive additive steering, LINEAR-SCHEDULE one-sided variant, coeff=20 —
stronger-dose sibling of `adaptive_add_linear_c8_full_ramp_qwen3_8b.py` /
`adaptive_add_linear_c16_qwen3_8b.py`.

Same method (`adaptive_add_linear`) and default `denom` (the model's own
`n_layers`) — only `coeffs` differs. For Qwen3-8B (n_layers=36): layer 1 ->
20/36, ..., layer 36 -> 20.0 exactly (2.5x `fixed_add`'s c=8 dose).

See `adaptive_add_linear_c16_qwen3_8b.py` for why: closing the gap to
`fixed_add` by pushing coeff further, since the floor/ceiling stays one-sided
(never subtracts an already-more-intense projection) so larger coeff should
still avoid the hard-pin's degeneracy.

    python -m src.bias_steer run configs/exp/adaptive_add_linear_c20_qwen3_8b.py \
        --vector runs/20260901-092009_anchor-qwen3-8b_qwen3-8b/steering_vector.safetensors
"""

from src.bias_steer.config import (
    ExperimentConfig, DatasetSpec, SampleSpec, JudgeSpec, Coeffs,
)

config = ExperimentConfig(
    label="adaptive-add-linear-c20 qwen3-8b",
    models=["qwen3-8b"],
    dataset=DatasetSpec(
        name="snapshot",
        path="datasets/Snapshots/log_103_comparison_200.json",
        train_split=0.5, shuffle=False,
    ),
    sample=SampleSpec(),
    judge=JudgeSpec(name="neutrality"),
    coeffs=Coeffs(opinion=20.0, neutral=20.0),
    method="adaptive_add_linear",
    max_tokens=128,
    batch_size=16,
)

config.vector_path = "runs/20260901-092009_anchor-qwen3-8b_qwen3-8b/steering_vector.safetensors"

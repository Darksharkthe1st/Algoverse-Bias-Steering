"""Adaptive additive steering, LINEAR-SCHEDULE one-sided variant, coeff=40 —
next dose after `adaptive_add_linear_c30_qwen3_8b.py`.

Same method (`adaptive_add_linear`) and default `denom` (the model's own
`n_layers`) — only `coeffs` differs. For Qwen3-8B (n_layers=36): layer 1 ->
40/36, ..., layer 36 -> 40.0 exactly (5x `fixed_add`'s c=8 dose).

Why: coeff=30 pushed both arms past `fixed_add` (POS 138/148=93%, NEG
42/52=81%) but also produced a coherence caveat — several STEERED+ responses
skip reasoning (empty <think></think>) and assert semantically non-sequitur
justifications (see experiments/adaptive_vs_fixed/summary.md's coeff=30
section for verbatim examples). This run checks whether coeff=40 continues
that quality drift, saturates numerically without getting worse, or finally
produces the repetition-loop degeneracy `adaptive_add`'s hard pin hit at much
lower (but flat, non-ramped) targets. Manually spot-check logs/eval.txt
end-to-end (beginning/middle/end) before trusting any number from this run,
per the established practice on this branch (CLAUDE.md, "Working style on a
GPU box").

    python -m src.bias_steer run configs/exp/adaptive_add_linear_c40_qwen3_8b.py \
        --vector runs/20260901-092009_anchor-qwen3-8b_qwen3-8b/steering_vector.safetensors
"""

from src.bias_steer.config import (
    ExperimentConfig, DatasetSpec, SampleSpec, JudgeSpec, Coeffs,
)

config = ExperimentConfig(
    label="adaptive-add-linear-c40 qwen3-8b",
    models=["qwen3-8b"],
    dataset=DatasetSpec(
        name="snapshot",
        path="datasets/Snapshots/log_103_comparison_200.json",
        train_split=0.5, shuffle=False,
    ),
    sample=SampleSpec(),
    judge=JudgeSpec(name="neutrality"),
    coeffs=Coeffs(opinion=40.0, neutral=40.0),
    method="adaptive_add_linear",
    max_tokens=128,
    batch_size=16,
)

config.vector_path = "runs/20260901-092009_anchor-qwen3-8b_qwen3-8b/steering_vector.safetensors"

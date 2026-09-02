"""Fixed-coeff additive steering with the Qwen3-8b opinion vector — the BASELINE
comparator for the adaptive methods.

This is the swept-scalar path (`method="mean_diff"` -> `apply_resid_pre_add`):
`x ← x + (c/n_layers) r`, a uniform per-layer translation along the raw direction.
It is the "add −c·direction" side of the DoD #4 question — run it alongside
`adaptive_ablation_qwen3_8b.py` ("remove the direction") and
`adaptive_add_qwen3_8b.py` ("pin the projection") on the SAME model, vector,
prompts, and judge so all three sit in one comparable table.

Same merged-tree requirement, vector, dataset, and judge as its two siblings — see
`adaptive_ablation_qwen3_8b.py` for the full provenance and caveats. The only
differences here are `method` and `coeffs`.

Dose: opinion=neutral=8.0 — the coefficient the vector was VALIDATED at when it was
fit (configs/exp/anchor_qwen3_8b.py) and the dose the downstream apply configs use.
Using the same c keeps this baseline honest: it is the fixed-scalar operating point
the project already settled on, not a fresh sweep. (For the c* the sweep selects,
see the sibling fk/phase4-coeff-sweep doc; adaptive needs no such sweep.)

    python -m src.bias_steer run configs/exp/fixed_add_qwen3_8b.py \
        --vector runs/20260901-092009_anchor-qwen3-8b_qwen3-8b/steering_vector.safetensors
"""

from src.bias_steer.config import (
    ExperimentConfig, DatasetSpec, SampleSpec, JudgeSpec, Coeffs,
)

config = ExperimentConfig(
    label="fixed-add qwen3-8b",
    models=["qwen3-8b"],
    dataset=DatasetSpec(
        name="snapshot",
        path="datasets/Snapshots/log_103_comparison_200.json",
        train_split=0.5, shuffle=False,
    ),
    sample=SampleSpec(),
    judge=JudgeSpec(name="neutrality"),
    coeffs=Coeffs(opinion=8.0, neutral=8.0),         # the validated dose (anchor_qwen3_8b.py)
    method="mean_diff",                              # apply_resid_pre_add: fixed scalar coeff
    max_tokens=128,
    batch_size=16,
)

config.vector_path = "runs/20260901-092009_anchor-qwen3-8b_qwen3-8b/steering_vector.safetensors"

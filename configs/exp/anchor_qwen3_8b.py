"""Fit a real opinion vector on qwen3-8b — the frozen submission model.

The two generalization-test configs (`configs/apply_opinion_axbench.py`,
`configs/apply_opinion_issuebench.py`) both pin `models=["qwen3-8b"]` and need a
saved `steering_vector.safetensors` with shape (36, 4096) to apply — the shape
guard rejects anything fit on another model (see `docs/PREREG.md` §3b / contract
§12 A4 for why qwen3-8b is the pinned frozen submission model). No such vector
existed anywhere in the repo or its history, so this config extracts one the
normal way: TRAIN split -> generate + judge + bucket -> mean_diff direction ->
TEST split eval.

Same battery (`snapshot`/log_103_comparison_200.json) and method as the other
`anchor_*` configs, sibling to `configs/exp/anchor_qwen7b.py`. Coeffs match the
dose already fixed in the two apply configs (opinion=8.0, neutral=8.0) rather
than the qwen-7b notebook coeffs, so the fitted vector is validated at the same
dose it will actually be applied at downstream.

    python -m src.bias_steer run configs/exp/anchor_qwen3_8b.py
"""

from src.bias_steer.config import (
    ExperimentConfig, DatasetSpec, SampleSpec, JudgeSpec, Coeffs,
)

config = ExperimentConfig(
    label="anchor qwen3-8b",
    models=["qwen3-8b"],
    dataset=DatasetSpec(name="snapshot", path="datasets/Snapshots/log_103_comparison_200.json", train_split=0.5, shuffle=False),
    sample=SampleSpec(),
    judge=JudgeSpec(name="neutrality"),
    coeffs=Coeffs(opinion=8.0, neutral=8.0),
    method="mean_diff",
    max_tokens=128,
    batch_size=16,  # 8B in fp16; drop to 8 if OOM (matches g1_qwen3_8b.py note)
)

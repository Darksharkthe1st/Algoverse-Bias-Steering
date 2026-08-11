"""Coefficient sweep point c=18 on qwen-1.8b (needed-experiments.md §7).

One point on a dose-response curve. Both directions use the same magnitude so
the curve is symmetric and readable; the notebook's tuned 14/15 sits inside the
swept range as a reference point.

Why re-sweep at all: the notebook's coeffs were tuned against a pipeline whose
steered arms were judged on chat-template-contaminated text (docs/04-parity.md
rung 4), so those tuned values do not necessarily transfer to the fixed pipeline.
Everything downstream depends on picking coeffs under the clean one.

Same snapshot prompt set as the parity anchor, so these are directly comparable
to the validated Log_103 run.
"""

from src.bias_steer.config import (
    ExperimentConfig, DatasetSpec, SampleSpec, JudgeSpec, Coeffs,
)

config = ExperimentConfig(
    label="sweep qwen18 c18",
    models=["qwen-1.8b"],
    dataset=DatasetSpec(name="snapshot", path="datasets/Snapshots/log_103_comparison_200.json", train_split=0.5, shuffle=False),
    sample=SampleSpec(),
    judge=JudgeSpec(name="neutrality"),
    coeffs=Coeffs(opinion=18, neutral=18),
    method="mean_diff",
    max_tokens=128,
    batch_size=32,
)

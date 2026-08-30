"""Clean baseline for qwen-14b at its notebook-tuned coeffs.

needed-experiments.md §0 asks for the anchors to be re-run so everything sits on
one scale. That is now doubly necessary: every archived steered number was
produced by judging chat-template-contaminated text (docs/04-parity.md rung 4),
so the archive cannot serve as the baseline it was meant to be.

Coeffs are the notebook's tuned values for this model (cell 37), kept so these
runs are comparable to the archive on everything except the defect. The sweep
branch is what re-tunes them.

Only ungated models are queued: gemma and llama-3 need an approved HF token.
"""

from src.bias_steer.config import (
    ExperimentConfig, DatasetSpec, SampleSpec, JudgeSpec, Coeffs,
)

config = ExperimentConfig(
    label="anchor qwen-14b",
    models=["qwen-14b"],
    dataset=DatasetSpec(name="snapshot", path="datasets/Snapshots/log_103_comparison_200.json", train_split=0.5, shuffle=False),
    sample=SampleSpec(),
    judge=JudgeSpec(name="neutrality"),
    coeffs=Coeffs(opinion=13, neutral=12),
    method="mean_diff",
    max_tokens=128,
    # 14B in fp16 is ~28GB of weights on a 40GB card, so the KV cache is what
    # decides whether this fits. Halved from the usual 32 to leave headroom; if it
    # still OOMs the coordinator soft-lands this run and continues with the rest.
    batch_size=16,
)

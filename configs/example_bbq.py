"""Starter experiment config — a small, fast smoke run on BBQ (Age).

Run it with:  python -m src.bias_steer run configs/example_bbq.py

Requires (on the GPU box): torch + transformer_lens installed, `HF_TOKEN` set for
model download, and `OPENAI_API_KEY` set for the neutrality judge.

This is deliberately tiny (small model, 40 sampled prompts) to prove the pipeline
end-to-end quickly. For a real parity check, match a historical run's exact model /
dataset / split / coeffs instead (see docs/03-gpu-bringup.md §5).
"""

from src.bias_steer.config import (
    ExperimentConfig, DatasetSpec, SampleSpec, JudgeSpec, Coeffs,
)

config = ExperimentConfig(
    label="smoke bbq age",
    models=["qwen-1.8b"],                       # smallest catalog model -> fastest first run
    dataset=DatasetSpec(
        name="bbq",
        path="datasets/BBQ_Prompt_Sets/Age.jsonl",
        train_split=0.5,
    ),
    sample=SampleSpec(limit=40, seed=0),        # keep the smoke run quick
    judge=JudgeSpec(name="neutrality"),         # gpt-4o-mini; needs OPENAI_API_KEY
    coeffs=Coeffs(opinion=14, neutral=15),      # qwen-1.8b values from the notebook
    method="mean_diff",                         # = coeff/n_layers, all layers, raw vector
    max_tokens=128,
    batch_size=16,
)

"""Phase-4-style eval — apply V2 (stance <- soft-refusal) to the held-out test
split from the contrast-vector build (runs/20260903-062503_..._qwen3-8b).

Sibling of eval_v1_contrast.py — same setup, different vector. See that file's
docstring for the full explanation (extraction-skipped `run` path, strip_reasoning
parity with the build phase, and the caveat about the printed `quality` summary's
mismatched pos/neg labels for a 9-way judge).

    python -m src.bias_steer run configs/eval_v2_contrast.py \
        --vector runs/20260903-062503_contrast-vectors-qwen3-8b_qwen3-8b/V2.safetensors
"""

from src.bias_steer.config import ExperimentConfig, DatasetSpec, SampleSpec, Coeffs
from src.bias_steer.judges.v2 import judge_v2_spec

config = ExperimentConfig(
    label="eval V2 on contrast test split",
    models=["qwen3-8b"],
    dataset=DatasetSpec(
        name="plain",
        path="/tmp/claude-1000/-home-ubuntu-Algoverse-Bias-Steering/bfb356f8-d45b-4411-a211-1bc1fbabff1e/scratchpad/test_split_prompts.txt",
        train_split=0.5,
        shuffle=True,
    ),
    sample=SampleSpec(seed=0, limit=50),
    judge=judge_v2_spec(model="gpt-4o-mini"),
    coeffs=Coeffs(opinion=8.0, neutral=8.0),
    method="mean_diff",
    max_tokens=2048,
    batch_size=64,
    strip_reasoning=True,
)
config.vector_path = "runs/20260903-062503_contrast-vectors-qwen3-8b_qwen3-8b/V2.safetensors"

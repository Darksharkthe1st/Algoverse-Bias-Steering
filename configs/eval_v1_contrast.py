"""Phase-4-style eval — apply V1 (soft-refusal <- hard-refusal) to the held-out
test split from the contrast-vector build (runs/20260903-062503_..._qwen3-8b).

This is the normal `run` path with a vector SUPPLIED (extraction skipped): every
sampled prompt from `test_split.csv` is evaluated at coeff=0 (INITIAL), +coeff
(STEERED_POS, toward V1's positive pole = soft-refusal), and -coeff (STEERED_NEG,
toward hard-refusal). Judged with the same judge v2.1 rubric used to build the
vector; `strip_reasoning=True` so the judge sees the post-</think> answer, same
as the build phase (see experiment.py:steer_and_judge's answer_of param).

    python -m src.bias_steer run configs/eval_v1_contrast.py \
        --vector runs/20260903-062503_contrast-vectors-qwen3-8b_qwen3-8b/V1.safetensors

`sample.limit` caps the eval to keep wall-clock time down for a pilot — raise it
for a fuller eval once this looks sane. `coeffs.opinion`/`coeffs.neutral` are set
to the same magnitude (8.0) used by this repo's other mean_diff-built vectors
(extract_axbench.py, extract_issuebench.py) as a starting point; there is no
established "right" value for these brand-new contrasts, so this is not a tuned
choice — treat the printed `quality` summary with that in mind (it also uses
`_contrast(config)`'s default (judge.labels[1], judge.labels[0]) pair, which is
NOT V1's actual poles for a 9-way judge — read `results.csv`'s per-condition
verdict counts directly instead of trusting that summary number).
"""

from src.bias_steer.config import ExperimentConfig, DatasetSpec, SampleSpec, Coeffs
from src.bias_steer.judges.v2 import judge_v2_spec

config = ExperimentConfig(
    label="eval V1 on contrast test split",
    models=["qwen3-8b"],
    dataset=DatasetSpec(
        name="plain",
        path="/tmp/claude-1000/-home-ubuntu-Algoverse-Bias-Steering/bfb356f8-d45b-4411-a211-1bc1fbabff1e/scratchpad/test_split_prompts.txt",
        train_split=0.5,   # irrelevant when a vector is supplied — train+test are folded into eval either way
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
config.vector_path = "runs/20260903-062503_contrast-vectors-qwen3-8b_qwen3-8b/V1.safetensors"

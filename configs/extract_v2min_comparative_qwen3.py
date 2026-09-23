"""Minimized judge-v2.1 extraction (V2 only) on the COMPARATIVE toy set — qwen3-8b.

Retry of extract_v2min_issuebench_qwen3.py after that run starved the soft-refusal
pole (IssueBench essay prompts -> the model writes the essay = stance 187/200, only
1 soft-refusal, so V2 could not build). Comparative "which do you prefer, X or Y?"
prompts are the opposite regime: the model naturally HEDGES on them (~150/200 neutral
under the old binary judge), so soft-refusal is populated.

Caveat (read the printed bucket table): this flips the risk — STANCE is now the
minority pole (~50/200 naturally). Hence train_split=0.85 (maximise train items) and
run with a lower floor if stance lands in the 30s:

    python -m src.bias_steer vectors configs/extract_v2min_comparative_qwen3.py \
        --contrasts V2 --n-floor 30

Everything else is unchanged from the IssueBench config: FULL 9-way judge labels every
response (audit trail + grow-out), only V2 = mean(stance) - mean(soft-refusal) is built.
DEFAULT_SYS is kept deliberately — its "give the clear, definitive answer" nudge lifts
the (scarce) stance pole. Qwen3 thinking OFF, max_tokens 512.
"""

from src.bias_steer.config import ExperimentConfig, DatasetSpec, Coeffs, SampleSpec
from src.bias_steer.judges.v2 import judge_v2_spec

config = ExperimentConfig(
    label="v2min extract V2 comparative qwen3-8b",
    models=["qwen3-8b"],                         # Qwen/Qwen3-8B @ b968826d9c46 (pinned)
    dataset=DatasetSpec(
        name="snapshot",
        path="datasets/Snapshots/log_103_comparison_200.json",  # 200 comparative prompts
        train_split=0.85,                        # 170 train / 30 test — stance is the tight pole
        shuffle=True,
    ),
    sample=SampleSpec(seed=0),
    judge=judge_v2_spec(model="gpt-4o-mini"),    # FULL 9-way v2.1 judge; only V2 is built
    coeffs=Coeffs(opinion=0.0, neutral=0.0),     # unused in `vectors` (no steering eval)
    method="mean_diff",
    max_tokens=512,
    batch_size=16,
    strip_reasoning=True,
    enable_thinking=False,
)

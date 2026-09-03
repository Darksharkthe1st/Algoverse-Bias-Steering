"""Build the 3 judge-v2.1 contrast vectors (V1/V2/V3) on qwen3-8b.

Run on the GPU box (needs torch + transformer_lens, HF_TOKEN, and OPENAI_API_KEY
for the judge):

    python -m src.bias_steer vectors configs/contrast_vectors_qwen3.py

This is the first-class Phase 2-3 path (`experiment.build_contrast_vectors`): the
TRAIN split's residuals are captured, judged with judge v2.1 (fine 9-way), bucketed
and collapsed, then V1/V2/V3 are built and saved with a norm profile. The held-out
TEST split is written for Phase 4 (the coeff sweep). No steering eval here.

Pool: point `dataset.path` at a combined `plain` file (one prompt per line) that
mixes the behaviors the three contrasts need — crucially some hard-refusal-eliciting
prompts (Do_Not_Answer) so V1 (soft <- hard) clears the floor, plus opinion
comparisons / BBQ / issuebench for stance vs soft vs non-engagement. EXCLUDE the 40
calibration items (datasets/Calibration/calibration_v2_prompts.csv). If a pole is
under `--n-floor` the gate skips that vector (see the printed table).
"""

from src.bias_steer.config import ExperimentConfig, DatasetSpec, SampleSpec, Coeffs
from src.bias_steer.judges.v2 import judge_v2_spec

config = ExperimentConfig(
    label="contrast-vectors qwen3-8b",
    models=["qwen3-8b"],                         # Qwen/Qwen3-8B @ b968826d9c46 (pinned in catalog)
    dataset=DatasetSpec(
        name="plain",
        path="datasets/Calibration/vector_pool.txt",   # combined pool; assemble per the docstring
        train_split=0.5,                                # half builds vectors, half held out for Phase 4
        shuffle=True,
    ),
    sample=SampleSpec(seed=0),
    judge=judge_v2_spec(model="gpt-4o-mini"),    # judge v2.1: 9 fine labels, collapsed in code
    coeffs=Coeffs(opinion=0.0, neutral=0.0),     # unused for vector extraction (no steering eval)
    method="mean_diff",
    max_tokens=2048,                             # room for qwen3 <think> + a full answer
    batch_size=8,
    strip_reasoning=True,                        # qwen3 emits <think>...</think>; judge the answer
)

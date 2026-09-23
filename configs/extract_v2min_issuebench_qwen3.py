"""Minimized judge-v2.1 extraction: build ONLY V2 (stance <- soft-refusal) on qwen3-8b.

The "minimized" pass of the updated-judge program. The FULL 9-way judge v2.1 still
labels every training response (the audit trail, and what a later grow-out to V1/V3
reuses) — but we only build the ONE contrast we intend to steer on:

    V2 = mean(stance-factual + stance-evaluative) - mean(soft-refusal)

Because V1 (needs a hard-refusal pole) and V3 (needs a non-engagement pole) are NOT
built, no hard-refusal-eliciting / topic-avoidant prompts are needed and only two
behaviour buckets have to reach the floor -> far fewer generations than the full
V1/V2/V3 program (configs/contrast_vectors_qwen3.py).

Run on the GPU box (torch + transformer_lens, HF_TOKEN, OPENAI_API_KEY), restricting
to V2 with --contrasts:

    python scripts/fetch_issuebench.py --split sample
    python -m src.bias_steer vectors configs/extract_v2min_issuebench_qwen3.py --contrasts V2

Produces runs/<id>/V2.safetensors (+ test_split.csv for the held-out eval, which is
configs/apply_v2min_heldout_qwen3.py). If a pole is under the floor the printed table
says so — bump `sample.limit` (more train items) or lower --n-floor and re-run.

Qwen3 thinking: enable_thinking=False pre-fills an empty <think></think> so the model
answers directly (no mid-<think> truncation at a small max_tokens — the bug that
voided the earlier prompt-baseline run, docs/HANDOFF_prompt_control.md §0.3).
strip_reasoning stays on as a belt-and-suspenders in case a trace leaks.
"""

from src.bias_steer.config import ExperimentConfig, DatasetSpec, Coeffs, SampleSpec
from src.bias_steer.judges.v2 import judge_v2_spec

config = ExperimentConfig(
    label="v2min extract V2 issuebench qwen3-8b",
    models=["qwen3-8b"],                         # Qwen/Qwen3-8B @ b968826d9c46 (pinned)
    dataset=DatasetSpec(
        name="issuebench",
        path="sample",                           # fetch first: scripts/fetch_issuebench.py --split sample
        train_split=0.5,                         # half builds V2, half held out for eval
        shuffle=True,
    ),
    sample=SampleSpec(limit=400, seed=0),        # 400 -> ~200 train / ~200 test
    judge=judge_v2_spec(model="gpt-4o-mini"),    # FULL 9-way v2.1 judge; only V2 is built
    coeffs=Coeffs(opinion=0.0, neutral=0.0),     # unused here (no steering eval in `vectors`)
    method="mean_diff",
    max_tokens=512,                              # enough for a direct answer with thinking OFF
    batch_size=16,
    strip_reasoning=True,
    enable_thinking=False,                        # the truncation-bug fix (chosen 2026-09-23)
)
# Cap the load (636k rows in the `sample` split) before SampleSpec sampling.
config.dataset.max_rows = 3000

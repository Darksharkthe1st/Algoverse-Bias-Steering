"""Forced-contrast opinion direction on IssueBench prompts (the trustworthy vector).

Same real IssueBench prompts as configs/extract_issuebench.py, but fit with
FORCED-CONTRAST extraction (contrast_mode="forced") instead of natural variation.

Why (FK, 2026-09-23): the natural-variation run (20260923-080817) bucketed 160
neutral / 40 opinionated, and the 40 opinionated prompts were almost all ones whose
TEMPLATE already embeds a stance ("X being a bad thing", "... being morally
justifiable"). So the natural contrast was fitting a PROMPT-FRAMING feature ("did
the input contain a stance word") rather than the model's opinionation behaviour —
confounded and unusable. See algoverse_iclr_readiness.md §"Session addendum".

Forced-contrast fixes this: every prompt is generated TWICE on the SAME text —
once under POS_SYS ("take a clear side"), once under NEG_SYS ("stay strictly
neutral") — and the two poles are bucketed BY CONSTRUCTION. Paired on the identical
prompt, the prompt content cancels in mean(pos) - mean(neg), leaving the
opinionated-vs-neutral behavioural axis; buckets are balanced 200/200 by design.
This is the CAA / RepE / Arditi-style contrastive method.

    python -m src.bias_steer run configs/extract_issuebench_forced.py

Cost note: forced mode runs 2 passes over the 200 train prompts (~400 generations)
but consults NO judge in the train phase (bucketing is by construction), so it
trades a little GPU for zero train-phase OpenAI cost. The TEST phase still judges
the self-eval. Needs OPENAI_API_KEY for that TEST eval.

The two system prompts are DEFAULT_POS_SYS / DEFAULT_NEG_SYS, frozen in
src/bias_steer/config.py — they are part of the method here, recorded in the
manifest. enable_thinking=False (avoids <think> truncation, as everywhere else).
"""

from src.bias_steer.config import ExperimentConfig, DatasetSpec, JudgeSpec, Coeffs, SampleSpec

config = ExperimentConfig(
    label="extract issuebench opinion (forced-contrast)",
    models=["qwen3-8b"],                              # frozen submission model, shape (36, 4096)
    dataset=DatasetSpec(name="issuebench", path="sample", train_split=0.667, shuffle=True),
    judge=JudgeSpec(name="neutrality"),               # used only by the TEST-phase self-eval
    coeffs=Coeffs(opinion=8.0, neutral=8.0),          # validated at the dose it will be applied at
    sample=SampleSpec(limit=300, seed=0),             # 300 sampled -> 200 train / 100 test
    method="mean_diff",
    max_tokens=256,  # writing tasks; enable_thinking off so no <think> trace to budget for
    batch_size=16,   # 8B in fp16; drop to 8 if OOM
)
config.dataset.max_rows = 5000        # cap the 636k-row `sample` load before sampling
config.enable_thinking = False        # no mid-<think> truncation of the induced generations
config.contrast_mode = "forced"       # pos_system_prompt vs neg_system_prompt, bucketed by construction

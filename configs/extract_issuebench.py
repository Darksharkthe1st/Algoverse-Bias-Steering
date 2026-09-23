"""Extract an opinion(neutrality) direction ON IssueBench prompts (real data).

Fits a diff-of-means "opinion" vector from the IssueBench writing-assistance
prompts (Röttger et al., 2025; arXiv:2502.08395) instead of the artificial GPT
comparison prompts the shipped vector (20260903-105600) was fit on. Motivation
(FK, 2026-09-23): the direction should come from a real dataset, and an
in-distribution IssueBench vector is the upper-bound control that separates
"the toy direction didn't transfer" from "opinionation isn't linearly steerable
on realistic prompts" (see algoverse_iclr_readiness.md §"Session addendum").

Produces runs/<id>/steering_vector.safetensors, shape (36, 4096).

    python scripts/fetch_issuebench.py --split sample   # already fetched on the box
    python -m src.bias_steer run configs/extract_issuebench.py

HOW THE CONTRAST IS BUILT (read before trusting the vector): the TRAIN phase
generates on each prompt under the DEFAULT system prompt, the judge labels each
response opinionated/neutral, residuals are bucketed by that verdict, and the
vector is mean(opinionated) - mean(neutral) (src/bias_steer/experiment.py
_extract_vector). It therefore depends on the model NATURALLY producing BOTH
buckets on IssueBench. On the toy comparison prompts the split is ~75/25; on
IssueBench writing tasks the model may lean heavily one way, leaving a tiny
minority bucket and a noisy direction. VERIFY the run's bucket sizes in
`logs/run.log` ("building steering vector (buckets: {...})") -- if one bucket is
< ~30% of the other, this natural-variation extraction is not sound and we should
switch to a forced-contrast extraction (generate under POS_SYS vs NEG_SYS), which
needs a small pipeline change, not just a config.

TRAIN/TEST HYGIENE: this fits on a train split of `sample`. When the resulting
vector is applied downstream (e.g. a prompt-vs-steer run on the `debug` split),
confirm the eval prompts are DISJOINT from these train prompts, or the comparison
is circular. `debug`'s 150 curated prompts may be a subset of `sample`.

<think>-truncation: FIXED here (enable_thinking=False). The prior extract run
(20260901-091212) predated this, ran thinking-ON at max_tokens=128, produced only
residuals.safetensors (no vector), and is contaminated -- do not reuse it.
Needs OPENAI_API_KEY (the neutrality judge).
"""

from src.bias_steer.config import ExperimentConfig, DatasetSpec, JudgeSpec, Coeffs, SampleSpec

config = ExperimentConfig(
    label="extract issuebench opinion",
    models=["qwen3-8b"],                              # frozen submission model, shape (36, 4096)
    dataset=DatasetSpec(name="issuebench", path="sample", train_split=0.667, shuffle=True),
    judge=JudgeSpec(name="neutrality"),
    coeffs=Coeffs(opinion=8.0, neutral=8.0),          # validated at the dose it will be applied at
    sample=SampleSpec(limit=300, seed=0),             # 300 sampled -> 200 train / 100 test (train_split 2/3)
    method="mean_diff",                               # diff-of-means direction (explicit)
    max_tokens=256,  # IssueBench is a writing task, not a one-line answer; the judge
                     # needs enough response to label opinionated vs neutral. 128
                     # truncates that. enable_thinking is off (below) so there is no
                     # <think> trace to budget for.
    batch_size=16,   # 8B in fp16; drop to 8 if OOM
)
# Cap the 636k-row `sample` load before sampling (avoids one Example per row).
config.dataset.max_rows = 5000
# Turn thinking OFF so extraction generations don't truncate mid-<think> and poison
# the bucketed residuals (docs/HANDOFF_prompt_control.md §0.3) -- the fix the
# 20260901-091212 run lacked.
config.enable_thinking = False

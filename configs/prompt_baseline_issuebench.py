"""System-prompt control baseline on IssueBench prompts (EXP-A, 2026-09-23).

Same steer-vs-prompt head-to-head as `configs/prompt_baseline_opinion.py`, but the
eval set is IssueBench (Röttger et al., 2025; arXiv:2502.08395) instead of the toy
GPT forced-choice comparison prompts.

Motivation: on the toy prompts the system-prompt baseline hit 100% in BOTH
directions (run `20260923-030712_prompt-baseline-opinion_qwen3-8b`) — a CEILING
regime, because the target behaviour ("take a side vs. hedge") is trivially
instructable on trivially side-able questions, so it cannot test the method's
limits. IssueBench prompts are realistic writing-assistance requests on contested
political/social issues, where "take a clear side" is NOT guaranteed to succeed —
breaking the ceiling and giving the steer-vs-prompt `beat_rate` room to be
informative. See `algoverse_iclr_readiness.md` §"Session addendum" (EXP-A).

Runs the SAME five arms and the SAME neutrality judge as the opinion config, so the
two are directly comparable (INITIAL, PROMPT_POS/NEG, STEERED_POS/NEG):

    python -m src.bias_steer run configs/prompt_baseline_issuebench.py

CAVEAT (transfer): the applied opinion vector was extracted on the toy comparison
prompts, NOT on IssueBench — whether it steers this prompt distribution at all is
part of what EXP-A measures. A weak steer arm here is a finding (the opinion
direction does not transfer to realistic issue-writing), not a bug. Likewise
fixed-add c=8 barely steered on the toy set (+11 items); a coeff sweep / the
adaptive-linear schedule is the natural follow-up if the direction does transfer.

The two behaviour-inducing system prompts are FROZEN in src/bias_steer/config.py
(DEFAULT_POS_SYS / DEFAULT_NEG_SYS) — edit them there, not here. Needs
OPENAI_API_KEY (the neutrality judge).
"""

from src.bias_steer.config import ExperimentConfig, DatasetSpec, JudgeSpec, Coeffs, SampleSpec

config = ExperimentConfig(
    label="prompt-baseline issuebench",
    models=["qwen3-8b"],                              # frozen submission model
    dataset=DatasetSpec(
        name="issuebench",
        path="debug",   # 150 curated IssueBench prompts (arXiv:2502.08395); the
                         # "sample" split (636k) is also fetched on the box if a
                         # larger, topic-polarity-stratified n is wanted later.
    ),
    judge=JudgeSpec(name="neutrality"),
    coeffs=Coeffs(opinion=8.0, neutral=8.0),          # match the opinion run's fixed-add dose
    sample=SampleSpec(limit=150, seed=0),             # all 150 debug prompts
    max_tokens=256,  # IssueBench prompts are writing-assistance requests (short
                     # prose), not one-line forced-choice answers; max_tokens=128
                     # truncates a written stance before the judge can read it.
                     # enable_thinking is off (below) so there is no <think> trace
                     # to budget for -- 256 is generation headroom for the answer.
    batch_size=16,
)
# qwen3-8b defaults to thinking mode; turn it off so it answers directly and does
# not truncate mid-<think> (docs/HANDOFF_prompt_control.md §0.3) -- identical to
# the opinion config.
config.enable_thinking = False
# Full head-to-head: INITIAL + PROMPT_POS/NEG + STEERED_POS/NEG, with the native
# per-item steer-vs-prompt beat_rate (item-bootstrap CI) emitted automatically.
config.intervention = "both"
# The clean, re-extracted Qwen3-8B opinion vector (enable_thinking=False,
# shape-verified (36, 4096)); same vector as the opinion head-to-head.
config.vector_path = "runs/20260903-105600_anchor-qwen3-8b_qwen3-8b/steering_vector.safetensors"

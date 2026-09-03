"""System-prompt control baseline on the old opinion prompts (needed-experiments §14).

Runs the SAME eval prompts under three (or five) arms and judges them identically,
so "did the steering vector beat simply *asking* the model?" is a per-item
comparison (AGENTS.md §5a; AxBench, arXiv:2501.17148):

  - INITIAL     — default system prompt, no steering (the control)
  - PROMPT_POS  — opinion-inducing system prompt, no steering
  - PROMPT_NEG  — neutrality-inducing system prompt, no steering
  - STEERED_POS/NEG (only in intervention="both") — our diff-of-means vector

The eval set is the 2025 GPT opinion/comparison prompts, read one-per-line by the
`plain` loader — "the old opinion prompts" this control is meant to validate.

    # (a) pure prompt baseline — NO vector, nothing to fetch or fit:
    #     set config.intervention = "prompt" (default here) and run:
    python -m src.bias_steer run configs/prompt_baseline_opinion.py

    # (b) head-to-head vector vs prompt — set intervention="both" and supply a vector:
    python -m src.bias_steer run configs/prompt_baseline_opinion.py \
        --vector runs/<opinion_run>/steering_vector.safetensors

The two behaviour-inducing system prompts are FROZEN in src/bias_steer/config.py
(DEFAULT_POS_SYS / DEFAULT_NEG_SYS) and recorded verbatim in every manifest — they
are part of the method, like a judge version; edit them there, not here, and
document the change. Needs OPENAI_API_KEY (the neutrality judge).

§0.1 injection convention is already locked to the farhan-batch-coeffs rule
(coeff / n_layers per layer, raw vector — src/bias_steer/steering.py); §0.2
(k-repeat judge) is intentionally NOT applied here to keep API cost down, so treat
small per-item differences within the judge's ±1–2/100 drift with due caution.
"""

from src.bias_steer.config import ExperimentConfig, DatasetSpec, JudgeSpec, Coeffs, SampleSpec

config = ExperimentConfig(
    label="prompt-baseline opinion",
    models=["qwen3-8b"],                              # frozen submission model
    dataset=DatasetSpec(
        name="plain",
        path="datasets/GPT_Prompts/all_data_1000_prompts.txt",
    ),
    judge=JudgeSpec(name="neutrality"),
    coeffs=Coeffs(opinion=8.0, neutral=8.0),          # only used by the steer arms (intervention="both")
    sample=SampleSpec(limit=200, seed=0),
    max_tokens=128,  # no reasoning trace to budget for (enable_thinking=False
                      # below) -- same headroom as the other non-reasoning models
                      # this control was designed for.
    batch_size=16,
)
# qwen3-8b defaults to thinking mode; the empty-<think> template pre-fill turns it
# off so the model answers directly, like the other (non-reasoning) models in this
# project's set. Measured before this: qwen3-8b's <think> traces run ~250-860
# tokens on this dataset before ever reaching an answer (median 381, p95 762 over
# n=32) -- genuine reasoning, not padding, but it means a naive max_tokens=128 run
# truncates ~1 in 4 examples mid-thought with no verdict at all (see git history:
# two prior fix attempts here -- raising max_tokens to 512 then 1024, then porting
# fk/init-better-rubric's strip_reasoning -- before landing on turning thinking off
# entirely, which is simpler and matches the non-reasoning baseline everywhere else
# in this project). strip_reasoning is left False: with thinking off there's no
# <think> block in the generated continuation to strip.
config.enable_thinking = False
# Default to the pure prompt baseline (no vector needed). Flip to "both" AND pass
# --vector (or set config.vector_path) to run the steer-vs-prompt head-to-head.
config.intervention = "steer"  # INITIAL + STEERED_POS/NEG only -- PROMPT_POS/NEG
                                # already exist (committed run
                                # runs/20260903-093536_prompt-baseline-opinion_qwen3-8b,
                                # same dataset/sample/seed/model), merged offline
                                # instead of regenerated (see the merge script next
                                # to this run's output).
# RE-EXTRACTED 2026-09-03 (configs/exp/anchor_qwen3_8b.py, enable_thinking=False)
# to replace the original (runs/20260901-092009_anchor-qwen3-8b_qwen3-8b), which
# predated enable_thinking and ran thinking-ON at max_tokens=128 -- likely the
# same truncation contamination fixed for the prompt arms. This vector's own
# TRAIN/TEST battery (log_103_comparison_200, same as the original) is clean: 0
# <think> tags, shape verified (36, 4096). Its own self-test already shows the
# neutral direction is weak (27 good / 26 bad -- near chance) against a baseline
# that's 70% opinionated by default under non-thinking mode.
config.vector_path = "runs/20260903-105600_anchor-qwen3-8b_qwen3-8b/steering_vector.safetensors"

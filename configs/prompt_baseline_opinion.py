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
    max_tokens=128,
    batch_size=16,
)
# Default to the pure prompt baseline (no vector needed). Flip to "both" AND pass
# --vector (or set config.vector_path) to run the steer-vs-prompt head-to-head.
config.intervention = "prompt"
# config.vector_path = "runs/REPLACE_WITH_OPINION_RUN/steering_vector.safetensors"

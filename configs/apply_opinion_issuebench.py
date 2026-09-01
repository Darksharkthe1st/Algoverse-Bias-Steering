"""GPU-B / Mode 2 — apply an EXISTING opinion direction to IssueBench prompts.

The generalization test on IssueBench: normal `run` with a vector SUPPLIED, so
extraction is skipped and the saved direction is evaluated on IssueBench's
writing-assistance prompts. Sibling of apply_opinion_axbench.py.

    python scripts/fetch_issuebench.py --split sample
    python -m src.bias_steer run configs/apply_opinion_issuebench.py \
        --vector runs/<opinion_run>/steering_vector.safetensors

The supplied vector MUST be a qwen3-8b (n_layers, d_model) steering_vector.safetensors
— the shape guard rejects a mismatched one. Pass --vector or set config.vector_path.
A loud warning notes the train split is not used to fit. Needs OPENAI_API_KEY.
"""

from src.bias_steer.config import ExperimentConfig, DatasetSpec, JudgeSpec, Coeffs, SampleSpec

config = ExperimentConfig(
    label="apply opinion vec on issuebench",
    models=["qwen3-8b"],
    dataset=DatasetSpec(name="issuebench", path="sample"),
    judge=JudgeSpec(name="neutrality"),
    coeffs=Coeffs(opinion=8.0, neutral=8.0),
    sample=SampleSpec(limit=200, seed=0),
    max_tokens=128,
    batch_size=16,
)
config.dataset.max_rows = 3000
# The saved opinion direction to apply. Pass --vector, or set the real path here.
config.vector_path = "runs/REPLACE_WITH_OPINION_RUN/steering_vector.safetensors"

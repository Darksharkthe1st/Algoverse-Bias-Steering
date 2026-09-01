"""FK-5(b) — apply our EXISTING opinion direction to AxBench prompts.

Generalization test: does a difference-of-means opinion direction (fit on our
stance battery) still steer opinionation on AxBench's realistic, mostly-factual
prompt distribution? This is the normal `run` path with a vector SUPPLIED, so
extraction is skipped and the saved direction is evaluated instead. Swap `dataset`
to `issuebench` (path='debug') to run the same generalization test on IssueBench.

    # fetch the eval prompts first
    python scripts/fetch_axbench.py --variant 2b/l20
    # apply a saved opinion vector to them (needs OPENAI_API_KEY for the judge)
    python -m src.bias_steer run configs/apply_opinion_axbench.py \
        --vector runs/<prior_opinion_run>/steering_vector.safetensors

Pass --vector, or set config.vector_path below to pin it in the config. When a
vector is supplied, the TRAIN split is not used to fit one — every sampled prompt
is evaluated instead (a loud warning notes the train split was skipped).
"""

from src.bias_steer.config import ExperimentConfig, DatasetSpec, JudgeSpec, Coeffs, SampleSpec

config = ExperimentConfig(
    label="apply opinion vec on axbench",
    models=["qwen3-8b"],                            # the frozen submission model
    dataset=DatasetSpec(name="axbench", path="2b/l20/test"),  # AxBench prompts (labels unused here)
    judge=JudgeSpec(name="neutrality"),             # our opinionation judge
    coeffs=Coeffs(opinion=8.0, neutral=8.0),        # dose magnitude; tune per the coeff sweep
    sample=SampleSpec(limit=200, seed=0),           # cap the eval set for a pilot
    max_tokens=128,
    batch_size=16,
)
# The saved opinion direction to apply. Leave as a placeholder and pass --vector,
# or point it at a prior run's artifact.
config.vector_path = "runs/REPLACE_WITH_OPINION_RUN/steering_vector.safetensors"

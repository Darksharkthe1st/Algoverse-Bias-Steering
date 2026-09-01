"""GPU-A / Mode 1 — extract an opinion(neutrality) direction ON IssueBench prompts.

Same pipeline as extract_axbench.py, on the IssueBench writing-assistance prompts.
Produces runs/<id>/steering_vector.safetensors.

    python scripts/fetch_issuebench.py --split sample
    python -m src.bias_steer run configs/extract_issuebench.py

`dataset.max_rows` caps how many of the 636k `sample` prompts are materialised
(the loader would otherwise build an Example per row). Same bucket-balance gate as
extract_axbench.py applies. Needs OPENAI_API_KEY.
"""

from src.bias_steer.config import ExperimentConfig, DatasetSpec, JudgeSpec, Coeffs, SampleSpec

config = ExperimentConfig(
    label="extract issuebench opinion",
    models=["qwen3-8b"],
    dataset=DatasetSpec(name="issuebench", path="sample"),
    judge=JudgeSpec(name="neutrality"),
    coeffs=Coeffs(opinion=8.0, neutral=8.0),
    sample=SampleSpec(limit=400, seed=0),
    max_tokens=128,
    batch_size=16,
)
# Cap the load (636k rows in the `sample` split) before sampling. Use the `debug`
# split instead (150 prompts, no --split needed) for the cheapest smoke test.
config.dataset.max_rows = 3000

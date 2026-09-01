"""GPU-A / Mode 1 — extract an opinion(neutrality) direction ON AxBench prompts.

The standard `run` pipeline: generate on the TRAIN split, bucket each response with
the neutrality judge (opinionated vs neutral), build the mean-difference vector, and
evaluate it on TEST. Produces runs/<id>/steering_vector.safetensors.

    python scripts/fetch_axbench.py --variant 2b/l20
    python -m src.bias_steer run configs/extract_axbench.py

⚠️ Gate before trusting the vector: AxBench prompts are mostly factual, so the
opinionated/neutral contrast can be weak. Check the run.log "building steering
vector (buckets: {...})" line — BOTH buckets must be non-trivially populated, else
the mean-diff is meaningless (needed-experiments §0). Needs OPENAI_API_KEY.
"""

from src.bias_steer.config import ExperimentConfig, DatasetSpec, JudgeSpec, Coeffs, SampleSpec

config = ExperimentConfig(
    label="extract axbench opinion",
    models=["qwen3-8b"],                              # frozen submission model
    dataset=DatasetSpec(name="axbench", path="2b/l20/train"),
    judge=JudgeSpec(name="neutrality"),
    coeffs=Coeffs(opinion=8.0, neutral=8.0),          # TEST-phase dose; tune per the coeff sweep
    sample=SampleSpec(limit=400, seed=0),             # cap for a pilot; raise for a real fit
    max_tokens=128,
    batch_size=16,
)

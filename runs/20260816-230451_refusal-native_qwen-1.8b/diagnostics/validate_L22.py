"""Layer sweep cell: layer 22.

Phase-2 sweep cell: same as configs/refusal_native_validate.py at layer 15
(the paper's own layer), for the layer sweep in docs/needed-experiments.md §12."""

from src.bias_steer.config import ExperimentConfig, DatasetSpec, JudgeSpec, Coeffs

config = ExperimentConfig(
    label="refusal native validate L22",
    models=["qwen-1.8b"],
    dataset=DatasetSpec(name="refusal_eval"),
    judge=JudgeSpec(name="refusal_substring", labels=["compliance", "refusal"]),
    coeffs=Coeffs(opinion=1.0, neutral=1.0),
    method="ablation",
    system_prompt="",
    max_tokens=512,
    batch_size=16,
)
config.direction_path = "runs/20260816-230451_refusal-native_qwen-1.8b/steering_vector.safetensors"
config.direction_layer = 22

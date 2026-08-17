"""Phase 2 — validate OUR native refusal vector via ablation + act-add.

Takes the vector produced by Phase 1 (configs/refusal_native.py) and runs it
through the same refusal harness used for the paper's published direction:
ablation should DROP harmful refusal, act-add(+) should RAISE harmless refusal.
This is the test that our native (response-mean, judge-bucketed) vector encodes
the refusal *decision* and not merely harmful-topic content.

    python -m src.bias_steer refuse configs/refusal_native_validate.py

BEFORE RUNNING: set `direction_path` to the Phase-1 run's steering vector, and
`direction_layer` to the layer you want to test (pick the layer that Phase 3,
scripts/refusal_native_compare.py, reports as best-aligned to the paper — or
sweep a few). No OpenAI key needed.
"""

from src.bias_steer.config import ExperimentConfig, DatasetSpec, JudgeSpec, Coeffs

config = ExperimentConfig(
    label="refusal native validate",
    models=["qwen-1.8b"],
    dataset=DatasetSpec(name="refusal_eval"),
    judge=JudgeSpec(name="refusal_substring", labels=["compliance", "refusal"]),
    coeffs=Coeffs(opinion=1.0, neutral=1.0),  # act-add dose magnitude
    method="ablation",
    system_prompt="",
    max_tokens=512,  # paper's value; the load track found 128 truncates real refusals
    batch_size=16,
)
# ---- FILL THESE IN from the Phase-1 run ----
config.direction_path = "runs/20260816-230451_refusal-native_qwen-1.8b/steering_vector.safetensors"
# Phase 3 (scripts/refusal_native_compare.py) put our best alignment to the paper's
# direction at layer 19 (cos +0.370; +0.358 at the paper's own layer 15).
config.direction_layer = 19

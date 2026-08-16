"""Phase 1 — extract a refusal vector with OUR native pipeline (arXiv:2406.11717).

Reproduces refusal using the SAME `mean_diff` machinery that produces our bias /
opinion vectors: generate on a harmful+harmless train mix, bucket each response
by the deterministic `refusal_substring` judge (refuse vs comply), take the
mean-difference of the per-layer residuals. The result is a refusal direction in
our own convention, directly comparable to any future vector we extract.

Run (NO OpenAI key needed — the judge is substring-based):

    python -m src.bias_steer run configs/refusal_native.py

Output: `runs/<run_id>/steering_vector.safetensors`, shape (n_layers, d_model).
Feed that path to Phase 2 (configs/refusal_native_validate.py) and Phase 3
(scripts/refusal_native_compare.py). See docs/needed-experiments.md.

CONVENTION NOTE: this deliberately uses our recipe (response-token mean, bucketed
by the model's actual verdict), NOT the paper's (prompt last-token, bucketed by
known label). That makes the refuse bucket also the harmful-topic bucket, so the
vector could encode topic rather than the refusal decision — Phase 2 (does
ablating it bypass refusal?) is the test that tells them apart.
"""

from src.bias_steer.config import ExperimentConfig, DatasetSpec, SampleSpec, JudgeSpec, Coeffs

config = ExperimentConfig(
    label="refusal native",
    models=["qwen-1.8b"],
    dataset=DatasetSpec(name="refusal_contrast", train_split=0.85, shuffle=True),
    # 128 harmful + 128 harmless (balanced), matching the paper's n_train; harmless_train
    # is ~70x larger, so per_group balancing is essential.
    sample=SampleSpec(per_group=("label", 128), seed=0),
    judge=JudgeSpec(name="refusal_substring", labels=["compliance", "refusal"]),
    coeffs=Coeffs(opinion=4.0, neutral=4.0),  # only the (ignored) eval half uses these
    method="mean_diff",
    system_prompt="",  # our chat rendering with no injected instruction; keep buckets natural
    max_tokens=64,      # enough to trigger a refusal prefix; keeps extraction cheap
    batch_size=16,
)
# Ad-hoc field read by load_refusal_contrast:
config.dataset.split = "train"

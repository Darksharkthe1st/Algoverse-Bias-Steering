"""Forced-contrast opinion vector via adaptive-LINEAR steering, coeff=16, qwen3-8b.

Experiment #1: apply the FORCED-CONTRAST opinion vector (run 20260923-090514, built
with POS_SYS/NEG_SYS on IssueBench, contrast_mode=forced) using the adaptive per-layer
LINEAR-schedule additive method (`adaptive_add_linear`, target_L = coeff·L/n_layers,
one-sided floor) — instead of the fixed single-scalar add the forced run self-evaluated
with. This is the dose-sweep sibling: c16 / c20 / c30.

Held-out eval: the EXACT 100 IssueBench prompts the forced run held out (last 100 rows of
its examples.csv → datasets/Snapshots/issuebench_forced_heldout_100.json), so results are
directly comparable to the forced run's own fixed-add c=8 self-eval.

Judge `neutrality` (binary) — same judge as the forced self-eval and the anchor-vector
adaptive runs; keeps #1 about the STEERING METHOD (the 9-way judge is experiment #2).
Qwen3 thinking OFF + max_tokens=512 (the truncation-bug fix).

    python -m src.bias_steer run configs/forced_adaptive_linear_c16_qwen3_8b.py

(--vector is set via config.vector_path below; pass --vector to override.)
"""

from src.bias_steer.config import ExperimentConfig, DatasetSpec, SampleSpec, JudgeSpec, Coeffs

FORCED_VECTOR = ("runs/20260923-090514_extract-issuebench-opinion-forced-contrast_qwen3-8b"
                 "/steering_vector.safetensors")   # (36, 4096) F16, md5 dab6cec…

config = ExperimentConfig(
    label="forced-adaptive-linear-c16 qwen3-8b",
    models=["qwen3-8b"],
    dataset=DatasetSpec(
        name="snapshot",
        path="datasets/Snapshots/issuebench_forced_heldout_100.json",
        train_split=0.5, shuffle=False,   # vector supplied -> all 100 evaluated; split unused
    ),
    sample=SampleSpec(),
    judge=JudgeSpec(name="neutrality"),
    coeffs=Coeffs(opinion=16.0, neutral=16.0),
    method="adaptive_add_linear",
    max_tokens=512,
    batch_size=16,
    enable_thinking=False,
)
config.vector_path = FORCED_VECTOR

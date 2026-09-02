"""Adaptive per-layer ablation of the Qwen3-8b opinion vector.

Tests the new `adaptive_ablation` method (docs/SCOPE_adaptive_steering.md) on the
frozen submission model, reusing the ALREADY-EXTRACTED Qwen3-8b opinion direction
rather than refitting one. The coefficient is the per-position dot product
`(x·r̂_L)`, computed in the hook — there is no dose and no coeff sweep; the vector
is simply projected out at every layer.

REQUIRES A MERGED TREE. This config only runs where BOTH are present:
  - the adaptive methods           (branch fk/adaptive-steering — this branch), and
  - the vector-supply `run` path + the committed vector + the snapshot dataset
                                    (branch fk/qwen3-8b-opinion-vector).
Merge fk/qwen3-8b-opinion-vector into this branch (or vice-versa) before running.
On the merged tree the supply path is method-agnostic: it loads `vector_path` and
calls `method.apply(model, vector, coeff)`, so `adaptive_ablation` consumes the
supplied (36, 4096) opinion vector directly.

Vector provenance: `runs/20260901-092009_anchor-qwen3-8b_qwen3-8b/` — a mean_diff
opinion direction fit on the `snapshot` battery (configs/exp/anchor_qwen3_8b.py),
shape (36, 4096) fp16 = (n_layers, d_model) for Qwen/Qwen3-8B @ b968826d9c46. The
per-layer shape guard (`assert_steering_shape`) accepts it; a 1-D or wrong-model
vector fails loud (CLAUDE.md §6).

Note this is an IN-DISTRIBUTION technique comparison on the same battery the vector
was fit on, not a generalization claim — all three sibling configs
(adaptive_ablation / fixed_add / adaptive_add) evaluate the SAME items with the
SAME judge, so the method contrast is fair. For a generalization test, swap the
dataset to `axbench` (needs `scripts/fetch_axbench.py`). Pin the judge version on
any judged number (CLAUDE.md §4); the manifest records judge model + rubric.
"a" direction, never "the" — removing it does not identify the representation.

    # on the merged tree, with OPENAI_API_KEY set for the neutrality judge:
    python -m src.bias_steer run configs/exp/adaptive_ablation_qwen3_8b.py \
        --vector runs/20260901-092009_anchor-qwen3-8b_qwen3-8b/steering_vector.safetensors
"""

from src.bias_steer.config import (
    ExperimentConfig, DatasetSpec, SampleSpec, JudgeSpec, Coeffs,
)

config = ExperimentConfig(
    label="adaptive-ablation qwen3-8b",
    models=["qwen3-8b"],                              # frozen submission model @ b968826d9c46
    dataset=DatasetSpec(
        name="snapshot",
        path="datasets/Snapshots/log_103_comparison_200.json",
        train_split=0.5, shuffle=False,              # matches anchor_qwen3_8b.py
    ),
    sample=SampleSpec(),
    judge=JudgeSpec(name="neutrality"),              # opinionation judge (needs OPENAI_API_KEY)
    coeffs=Coeffs(opinion=1.0, neutral=1.0),         # IGNORED by adaptive_ablation (no dose);
                                                     # both steered arms are identical (removal
                                                     # is sign-agnostic). Kept only to satisfy
                                                     # the config contract.
    method="adaptive_ablation",
    max_tokens=128,
    batch_size=16,                                   # 8B in fp16; drop to 8 if OOM
)

# Reuse the committed Qwen3-8b opinion vector; skip TRAIN (extraction). Pass the
# same path via --vector, or rely on this. On the merged tree `run(vector_path=...)`
# honors it and logs a loud "vector supplied — NOT extracting" warning.
config.vector_path = "runs/20260901-092009_anchor-qwen3-8b_qwen3-8b/steering_vector.safetensors"
